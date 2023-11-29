# utils.py:
import torch
import torch.nn as nn
import numpy as np
import random
from gym import spaces

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(3, 16)
        self.actor = nn.Linear(16, 2)
        self.critic = nn.Linear(16, 1)
        self.reset_trajectory()

    def forward(self, x):
        x = torch.relu(self.fc(x))
        action_prob = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_prob, value

    def reset_trajectory(self):
        self.state_trajectory = []  # To store states
        self.action_trajectory = []  # To store actions
        self.log_probs = []  # Log probabilities
        self.values = []  # Value estimates
        self.rewards = []  # Observed rewards

def log(message, file_object=None):
    """Utility function to log a message to a file and stdout."""
    if file_object:
        print(message, file=file_object)
        file_object.flush()

def get_discounted_rewards(rewards, gamma):
    """Calculate the discounted rewards with normalization."""
    returns = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float)
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)
    return returns

def report_old_model_distribution(model, env, file_object):
    """Report action probability distribution of the old model."""
    log("Old Model Action Probability Distribution:", file_object)
    for state in env.all_states():
        state_tensor = torch.FloatTensor(state).reshape(1, 3).float()
        action_prob, _ = model(state_tensor)
        log(f"State: {state} - Action Probabilities: {action_prob.detach().numpy()}", file_object)

def save_model(model, model_dir, filename):
    """Save the current model state."""
    model_path = f"{model_dir}/{filename}"
    torch.save(model.state_dict(), model_path)

def nstep_cumulative_prob_from_logs(n_step, log_probs):
    """
    Calculate n-step cumulative probabilities using log probabilities from trajectory.
    Args:
        n_step (int): Number of steps to consider for n-step cumulative probability.
        log_probs (Tensor): Tensor of log probabilities for each step.
    Returns:
        Tensor of n-step cumulative probabilities.
    """
    nstep_cumulative_probs = []
    for i in range(len(log_probs)):
        cumulative_log_prob = sum(log_probs[max(0, i - n_step + 1):i + 1])
        cumulative_prob = torch.exp(cumulative_log_prob)
        nstep_cumulative_probs.append(cumulative_prob)
    return torch.stack(nstep_cumulative_probs)

def nstep_cumulative_prob_from_states(model, n_step, states, actions):
    """
    Calculate n-step cumulative probabilities based on a model, states, and actions.
    Args:
        model (nn.Module): The model to use for probability calculation.
        n_step (int): Number of steps to consider for n-step cumulative probability.
        states (Tensor): Tensor of states.
        actions (Tensor): Tensor of actions.
    Returns:
        Tensor of n-step cumulative probabilities.
    """
    log_probs = []
    for state, action in zip(states, actions):
        action_prob, _ = model(state.unsqueeze(0))
        action_dist = torch.distributions.Categorical(action_prob)
        log_prob = action_dist.log_prob(action)
        log_probs.append(log_prob)
    
    return nstep_cumulative_prob_from_logs(n_step, torch.stack(log_probs))

def bprop_with_cumulative_prob(log_probs, values, returns, log_nstep_cp, log_nstep_cp_old, epsilon, optimizer):
    """Update the model using the PPO algorithm with clipping."""
    actor_loss = []
    critic_loss = []

    for log_prob, value, R, log_cp, log_cp_old in zip(log_probs, values, returns, log_nstep_cp, log_nstep_cp_old):
        advantage = R - value.item()
        ratio = torch.exp(log_cp - log_cp_old)  # PPO's probability ratio
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
        actor_loss.append(-torch.min(surr1, surr2))
        critic_loss.append(nn.functional.mse_loss(value.flatten(), torch.tensor([R], dtype=torch.float)))

    actor_loss = torch.stack(actor_loss).sum() if actor_loss else torch.tensor(0.0)
    critic_loss = torch.stack(critic_loss).sum() if critic_loss else torch.tensor(0.0)
    total_loss = actor_loss + critic_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return actor_loss.item(), critic_loss.item()

def bprop_with_log_prob():
    ...

def should_update(log_cp_new, log_cp_old, epsilon):
    """Check if model's counterfactual probabilities suggest an update."""
    ratio = torch.exp(log_cp_new - log_cp_old)
    return ratio < (1 - epsilon) or ratio > (1 + epsilon)

def load_reference_models(model_list, n):
    """Load reference models for comparative analysis."""
    reference_models = []
    random.shuffle(model_list)
    for i in range(min(len(model_list), n)):
        model = ActorCritic()
        model.load_state_dict(torch.load(model_list[i]))
        model.eval()
        reference_models.append(model)
    return reference_models


def train_a2c(env, num_episodes, learning_rate, gamma, save_path=None):
    """
    Train an Actor-Critic model.
    
    Args:
        env: The environment to train on.
        num_episodes (int): The number of episodes to train for.
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor for rewards.
        save_path (str, optional): Path to save the trained model.

    Returns:
        The trained ActorCritic model.
    """
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        model.reset_trajectory()

        while not done:
            state_tensor = torch.FloatTensor(state).reshape(1, 3).float()
            action_probs, value = model(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            next_state, reward, done, _ = env.step(action.item())
            log_prob = action_dist.log_prob(action)

            model.log_probs.append(log_prob)
            model.values.append(value)
            model.rewards.append(reward)

            state = next_state

        # Update the model at the end of each episode
        returns = get_discounted_rewards(model.rewards, gamma)
        bprop_with_log_prob(model.log_probs, model.values, returns, optimizer)

    if save_path:
        torch.save(model.state_dict(), save_path)

    return model

def train_ppo(env, num_episodes, learning_rate, gamma, epsilon, beta, save_path=None):
    """
    Train a PPO model.
    
    Args:
        env: The environment to train on.
        num_episodes (int): The number of episodes to train for.
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor for rewards.
        epsilon (float): Clipping parameter for PPO.
        beta (float): Coefficient for the entropy bonus.
        save_path (str, optional): Path to save the trained model.

    Returns:
        The trained ActorCritic model.
    """
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        model.reset_trajectory()

        while not done:
            state_tensor = torch.FloatTensor(state).reshape(1, 3).float()
            action_probs, value = model(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            next_state, reward, done, _ = env.step(action.item())
            log_prob = action_dist.log_prob(action)
            entropy = action_dist.entropy()

            model.log_probs.append(log_prob)
            model.values.append(value)
            model.rewards.append(reward)
            model.entropies.append(entropy)

            state = next_state

        # Update the model at the end of each episode
        returns = get_discounted_rewards(model.rewards, gamma)
        bprop_with_ppo(model.log_probs, model.values, model.entropies, returns, epsilon, beta, optimizer)

    if save_path:
        torch.save(model.state_dict(), save_path)

    return model

def bprop_with_ppo(log_probs, values, entropies, returns, epsilon, beta, optimizer):
    """
    Perform backpropagation using the PPO loss function.
    Args:
        log_probs (list): Log probabilities of actions.
        values (list): Value estimates from the critic.
        entropies (list): Entropy values for each action.
        returns (Tensor): Discounted returns.
        epsilon (float): Clipping parameter for PPO.
        beta (float): Coefficient for the entropy bonus.
        optimizer (Optimizer): The optimizer to use.
    """
    advantages = returns - values.detach()
    actor_loss = 0
    critic_loss = 0
    entropy_loss = 0

    for log_prob, value, R, entropy in zip(log_probs, values, returns, entropies):
        advantage = R - value.item()
        ratio = torch.exp(log_prob - log_prob.detach())
        surr1 = ratio
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
        actor_loss += -torch.min(surr1, surr2)
        critic_loss += nn.functional.mse_loss(value.flatten(), torch.tensor([R], dtype=torch.float))
        entropy_loss += -entropy

    total_loss = actor_loss + 0.5 * critic_loss + beta * entropy_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

def evaluate_model(model, env, num_runs):
    """
    Evaluate the given model in the environment.

    Args:
        model (nn.Module): The trained model to evaluate.
        env: The environment to evaluate the model on.
        num_runs (int): Number of runs to perform the evaluation.

    Returns:
        float: The average reward over the specified number of runs.
    """
    total_reward = 0.0

    for _ in range(num_runs):
        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            state_tensor = torch.FloatTensor(state).reshape(1, 3).float()
            action_probs, _ = model(state_tensor)
            action = torch.argmax(action_probs).item()  # Choose the action with highest probability
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward
            state = next_state

        total_reward += episode_reward

    average_reward = total_reward / num_runs
    return average_reward

