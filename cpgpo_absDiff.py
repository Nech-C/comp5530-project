import torch
from torch import nn as nn
from math import floor
from utils import get_discounted_rewards


def nstep_cumulative_prob_from_logs(n_step, log_probs):
    """
    Calculate n-step cumulative probabilities using log probabilities from trajectory.
    Args:
        n_step (int): Number of steps to consider for n-step cumulative probability.
        log_probs (Tensor): Tensor of log probabilities for each step.
    Returns:
        Tensor of n-step cumulative probabilities(log).
    """
    nstep_cumulative_probs = []
    #print(f"log probs: {log_probs}")
    for i in range(len(log_probs)):
        cumulative_log_prob = sum(log_probs[max(0, i - n_step + 1):i + 1])
        nstep_cumulative_probs.append(cumulative_log_prob)
        #print(f"cum log prob: {cumulative_log_prob}, length: {len(log_probs[max(0, i - n_step + 1):i + 1])}")
    return torch.stack(nstep_cumulative_probs).squeeze()


def nstep_cumulative_prob_from_states(model, n_step, states, actions):
    """
    Calculate n-step cumulative probabilities based on a model, states, and actions.
    Args:
        model (nn.Module): The model to use for probability calculation.
        n_step (int): Number of steps to consider for n-step cumulative probability.
        states (list): Tensor of states.
        actions (list): Tensor of actions.
    Returns:
        Tensor of n-step cumulative probabilities.
    """
    # Convert states to a tensor and apply squeeze if necessary
    states_tensor = torch.stack(states).squeeze(1)
    # print(f"States tensor shape: {states_tensor.shape}")

    # Convert actions to a tensor
    actions_tensor = torch.stack(actions)
    # print(f"Actions tensor shape: {actions_tensor.shape}")

    # Pass the states to the model
    action_probs, _ = model(states_tensor)
    # print(f"Action probabilities shape: {action_probs.shape}")

    # Create a distribution for the entire batch
    action_dists = torch.distributions.Categorical(action_probs)

    # Calculate log probabilities for all actions in the batch
    log_probs = action_dists.log_prob(actions_tensor.squeeze())
    # print(f"Log probabilities shape: {log_probs.shape}")

    return nstep_cumulative_prob_from_logs(n_step, log_probs)

def bprop_with_cumulative_prob_absDiff(log_probs, values, returns, log_nstep_cp, log_nstep_cp_old, epsilon, optimizer, device):
    actor_loss = []
    critic_loss = []
    counter = 0
    updated = 0
    #print(f"log_nstep_cp.shape: {log_nstep_cp.shape}, log_nstep_cp_old.shape: {log_nstep_cp_old.shape}")
    for log_prob, value, R, log_cp, log_cp_old in zip(log_probs, values, returns, log_nstep_cp, log_nstep_cp_old):
        advantage = R - value.item()
        cp_diff = torch.exp(log_cp) - torch.exp(log_cp_old)  # difference of new CP to old CP
        # cp_diff > 0 new policy more likely; < 0 old policy more likely
        #print(f"Ratio shape: {ratio.shape}")

        if len(log_probs) != len(log_nstep_cp) or len(log_probs) != len(log_nstep_cp_old):
            exit(-1)
        # Determine if an update should happen
        update = False
        if cp_diff < - epsilon or cp_diff > epsilon:
            update = True
        elif advantage < 0 and cp_diff < 0:
            update = True
        elif advantage > 0 and cp_diff > 0:
            update = True
        if update:
            updated += 1
        counter += 1
        # Calculate surrogate loss
        if update:
            surr = -log_prob * advantage
        else:
            surr = torch.zeros_like(log_prob)
        # if counter < 10:
        #     print(f"new: {log_cp}, old: {log_cp_old}, update: {update}, ratio: {cp_diff}")
        actor_loss.append(surr)
        # Move R to the same device as value before calculating mse_loss
        R_tensor = torch.tensor([R], device=device)
        critic_loss.append(nn.functional.mse_loss(value.flatten(), R_tensor, reduction='sum'))

    # Sum up the losses
    actor_loss = torch.sum(torch.stack(actor_loss))
    critic_loss = torch.sum(torch.stack(critic_loss))
    total_loss = actor_loss + critic_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print(f"update rate: {updated/counter}")
    print(f"actor loss: {actor_loss.item()}, critic_loss: {critic_loss.item()}, reward: {returns[-1]}")
    return actor_loss.item(), critic_loss.item()

def single_CPGPO_absDiff(env, model, reference_model, config, device):
    """
    Train the model using the CPGPO algorithm with cumulative probability in the objective function,
    growing n, and decreasing epsilon.
    """
    # Move models to the specified device
    model = model.to(device)
    reference_model = reference_model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    for episode in range(config['epoch']):
        print(episode)
        # Adjust n and epsilon as training progresses
        n_step = min(floor(config['starting_n'] + episode * config['n_growth']), config['max_n'])
        epsilon = max(config['epsilon'] * (1. / (1 + config['epsilon_decay'] * episode)), config['min_epsilon'])

        state = env.reset()
        done = False
        model.reset_trajectory()

        while not done:
            state_tensor = torch.FloatTensor(state).to(device).reshape(1, -1)
            action_prob, value = model(state_tensor)
            action_dist = torch.distributions.Categorical(action_prob)
            action = action_dist.sample()
            next_state, reward, done, _ = env.step(action.item())
            log_prob = action_dist.log_prob(action)

            # Storing trajectories
            model.log_probs.append(log_prob)
            model.values.append(value)
            model.rewards.append(reward)
            model.state_trajectory.append(state_tensor)
            model.action_trajectory.append(action)

            state = next_state

        # Calculating returns and cumulative probabilities
        returns = get_discounted_rewards(model.rewards, config['gamma']).to(device)
        log_nstep_cp_new = nstep_cumulative_prob_from_logs(n_step, model.log_probs)
        log_nstep_cp_old= nstep_cumulative_prob_from_states(reference_model, n_step, model.state_trajectory, model.action_trajectory)
        #print(f"old algo: {log_nstep_cp_old}, shape: {log_nstep_cp_old.shape}\n-------------------------")

        #print(f"new algo: {new_ouput}, shape: {new_ouput.shape}")
        # Backpropagation with custom update rules
        actor_loss, critic_loss = bprop_with_cumulative_prob_absDiff(
            model.log_probs,
            model.values,
            returns,
            log_nstep_cp_new,
            log_nstep_cp_old,
            epsilon,
            optimizer,
            device
        )

    return model