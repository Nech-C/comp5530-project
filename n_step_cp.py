import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from collections import deque
from utils import ActorCritic, load_reference_models
from gym import spaces
from grid_world import SimpleGridEnv 
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
pretrained_model_list = ["./trained_models/a2c_model1.pth"]
learning_rate = 0.002
gamma = 0.99
epoch = 1000
n_step = 4
epsilon = 0.2  # Clipping parameter for PPO

# Initialize environment and models
env = SimpleGridEnv()
model = ActorCritic()
reference_model = load_reference_models(pretrained_model_list, 5)[0].eval()  # Load reference model
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
writer = SummaryWriter(log_dir=f'./runs/1', flush_secs=1)

# Setup logging
def setup_logging():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = f"training_log_{timestamp}.txt"
    return open(log_filename, 'w')

log_file = setup_logging()

def log(message, file_object=None):
    """Utility function to log a message to a file and stdout."""
    if file_object:
        print(message, file=file_object)
        file_object.flush()

def get_discounted_rewards(rewards):
    """Calculate the discounted rewards with normalization."""
    returns = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float)
    # Normalize returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)
    return returns

def report_old_model_distribution(model, env, file_object):
    """Report action probability distribution of the old model."""
    log("Old Model Action Probability Distribution:", file_object)
    for state in env.all_states():
        state_tensor = torch.FloatTensor(state).reshape(1, 3).float()
        action_prob, _ = model(state_tensor)
        log(f"State: {state} - Action Probabilities: {action_prob.detach().numpy()}", file_object)

def save_model(model, file_object):
    """Save the current model state."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f"trained_model_{timestamp}.pth"
    torch.save(model.state_dict(), model_filename)
    log(f"Model saved as {model_filename}", file_object)

def get_nstep_cp(model, n_step, states, actions):
    """Compute n-step counterfactual probabilities for actions taken."""
    log_nstep_cp = []
    for i in range(len(states)):
        log_cp = 0.0  # log(1) = 0
        for j in range(max(0, i-n_step+1), i+1):
            state = states[j]
            action = actions[j]
            action_prob, _ = model(state)
            action_dist = torch.distributions.Categorical(action_prob)
            log_prob = action_dist.log_prob(action)
            log_cp += log_prob
        log_nstep_cp.append(log_cp)
    return log_nstep_cp

def update_model(log_probs, values, returns, log_nstep_cp, log_nstep_cp_old, epsilon, optimizer):
    """Update the model using the PPO algorithm with clipping."""
    actor_loss = []
    critic_loss = []

    # Compute actor and critic loss using the PPO loss function
    for log_prob, value, R, log_cp, log_cp_old in zip(log_probs, values, returns, log_nstep_cp, log_nstep_cp_old):
        advantage = R - value.item()
        ratio = torch.exp(log_cp - log_cp_old)  # PPO's probability ratio
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
        actor_loss.append(-torch.min(surr1, surr2))
        critic_loss.append(nn.functional.mse_loss(value.flatten(), torch.tensor([R], dtype=torch.float)))

    # Sum up the losses
    actor_loss = torch.stack(actor_loss).sum() if actor_loss else torch.tensor(0.0)
    critic_loss = torch.stack(critic_loss).sum() if critic_loss else torch.tensor(0.0)
    total_loss = actor_loss + critic_loss

    # Perform backpropagation and optimization step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return actor_loss.item(), critic_loss.item()

def should_update(log_cp_new, log_cp_old, epsilon):
    """Check if model's counterfactual probabilities suggest an update."""
    ratio = torch.exp(log_cp_new - log_cp_old)
    return ratio < (1 - epsilon) or ratio > (1 + epsilon)

def train():
    log_file = setup_logging()

    try:
        # Report action probability distribution of the old model at the start
        report_old_model_distribution(reference_model, env, log_file)
        log("-" * 60, log_file)  # Visual separator for the beginning of training

        for episode in range(epoch):
            state = env.reset()
            done = False
            model.reset_trajectory()

            while not done:
                # Interact with the environment and store trajectories
                state_tensor = torch.FloatTensor(state).reshape(1, 3).float()
                action_prob, value = model(state_tensor)
                action_dist = torch.distributions.Categorical(action_prob)
                action = action_dist.sample()
                next_state, reward, done, _ = env.step(action.item())
                log_prob = action_dist.log_prob(action)

                # Store log probabilities, values, and rewards
                model.log_probs.append(log_prob)
                model.values.append(value)
                model.rewards.append(reward)
                model.state_trajectory.append(state_tensor)
                model.action_trajectory.append(action)

                state = next_state

            returns = get_discounted_rewards(model.rewards)
            log_nstep_cp_new = get_nstep_cp(model, n_step, model.state_trajectory, model.action_trajectory)
            log_nstep_cp_old = get_nstep_cp(reference_model, n_step, model.state_trajectory, model.action_trajectory)

            # Log before update
            log(f"\nEpisode {episode} - Before Update:", log_file)
            log(f"{'State':>30} | {'Action':>6} | {'New CP':>10} | {'Old CP':>10} | {'Update':>6}", log_file)
            log("-" * 70, log_file)
            for state_tensor, action, log_cp_new, log_cp_old in zip(model.state_trajectory, model.action_trajectory, log_nstep_cp_new, log_nstep_cp_old):
                state_repr = ' '.join(f"{x:.2f}" for x in state_tensor.detach().numpy().flatten())
                cp_new = torch.exp(log_cp_new).item()
                cp_old = torch.exp(log_cp_old).item()
                update_decision = "Yes" if should_update(log_cp_new, log_cp_old, epsilon) else "No"
                log(f"{state_repr:>30} | {action.item():>6} | {cp_new:>10.4f} | {cp_old:>10.4f} | {update_decision:>6}", log_file)

            # Perform model update
            actor_loss, critic_loss = update_model(
                model.log_probs,
                model.values,
                returns,
                log_nstep_cp_new,
                log_nstep_cp_old,
                epsilon,
                optimizer
            )

            # Log after update
            log(f"Episode {episode} - After Update:", log_file)
            log(f"{'State':>30} | {'Action Probs':>20}", log_file)
            log("-" * 70, log_file)
            for state in env.all_states():
                state_tensor = torch.FloatTensor(state).reshape(1, 3).float()
                action_prob, _ = model(state_tensor)
                action_prob = action_prob.detach().numpy().squeeze()
                state_repr = ' '.join(f"{x:.2f}" for x in state)
                log(f"{state_repr:>30} | [{action_prob[0]:.4f}, {action_prob[1]:.4f}]", log_file)

            last_reward = model.rewards[-1].item()
            log(f"Reward: {last_reward}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss}", log_file)
            log("-" * 70, log_file)

        save_model(model, log_file)

    finally:
        log_file.close()

if __name__ == "__main__":
    train()
