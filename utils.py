# utils.py:

import torch
import torch.nn as nn
import numpy as np
import random
import os
from scipy.stats import entropy
from gym import spaces
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from math import floor
import time

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.rewards = None
        self.values = None
        self.log_probs = None
        self.action_trajectory = None
        self.state_trajectory = None
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


def load_models_from_directory(directory):
    models = []
    for filename in os.listdir(directory):
        if filename.endswith(".pth"):
            model_path = os.path.join(directory, filename)
            model = ActorCritic()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            models.append(model)
    return models


def get_discounted_rewards(rewards, gamma):
    """Calculate the discounted rewards with normalization."""
    returns = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float)
    #returns = (returns - returns.mean()) / (returns.std() + 1e-5)
    # print(returns)
    return returns


def report_old_model_distribution(model, env, file_object):
    """Report action probability distribution of the old model."""
    log("Old Model Action Probability Distribution:", file_object)
    for state in env.all_states():
        state_tensor = torch.FloatTensor(state).reshape(1, env.get_observation_size()).float()
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


def bprop_with_cumulative_prob(log_probs, values, returns, log_nstep_cp, log_nstep_cp_old, epsilon, optimizer, device):
    actor_loss = []
    critic_loss = []
    counter = 0
    updated = 0
    for log_prob, value, R, log_cp, log_cp_old in zip(log_probs, values, returns, log_nstep_cp, log_nstep_cp_old):
        advantage = R - value.item()
        ratio = torch.exp(log_cp - log_cp_old)  # ratio of new CP to old CP
        if len(log_probs) != len(log_nstep_cp) or len(log_probs) != len(log_nstep_cp_old):
            exit(-1)
        # Determine if an update should happen
        update = False
        if ratio < 1 - epsilon or ratio > 1 + epsilon:
            update = True
        elif advantage < 0 and ratio < 1:
            update = True
        elif advantage > 0 and ratio > 1:
            update = True
        if update:
            updated +=1
        counter += 1
        # Calculate surrogate loss
        if update:
            surr = -log_prob * advantage
        else:
            surr = torch.zeros_like(log_prob)

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
    # time.sleep(2.5)
    return actor_loss.item(), critic_loss.item()



def should_update(log_cp_new, log_cp_old, epsilon, advantage):
    """Check if model's counterfactual probabilities suggest an update."""
    ratio = torch.exp(log_cp_new - log_cp_old)
    update = False
    if ratio < 1 - epsilon or ratio > 1 + epsilon:
        update = True  # Rule 1: ratio is significantly different
    elif advantage < 0 and ratio < 1:
        update = True  # Rule 2: Advantage negative, ratio less than 1
    elif advantage > 0 and ratio > 1:
        update = True  # Rule 3: Advantage positive, ratio more than 1
    return update

def single_CPGPO(env, model, reference_model, config, device):
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
        log_nstep_cp_old = nstep_cumulative_prob_from_states(reference_model, n_step, model.state_trajectory, model.action_trajectory)

        # Backpropagation with custom update rules
        actor_loss, critic_loss = bprop_with_cumulative_prob(
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

def multi_CPGPO(env, model, reference_model, config, device):
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
        log_nstep_cp_old = nstep_cumulative_prob_from_states(reference_model, n_step, model.state_trajectory, model.action_trajectory)

        # Backpropagation with custom update rules
        actor_loss, critic_loss = bprop_with_cumulative_prob(
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

# baseline algo related
def bprop_with_log_prob(log_probs, values, returns, optimizer):
    """
    Perform backpropagation with log probabilities for A2C model.

    Args:
        log_probs (list): List of log probabilities from the policy network.
        values (list): List of value estimates from the value network.
        returns (list): List of discounted return values.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
    """
    policy_losses = []
    value_losses = []

    for log_prob, value, R in zip(log_probs, values, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage)  # Policy gradient loss
        value_losses.append(nn.functional.mse_loss(value.squeeze(), R))  # Value loss

    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()  # Total loss
    loss.backward()
    optimizer.step()


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


def train_a2c(env, model, num_episodes, learning_rate, gamma, device=None):
    model = model.to(device)  # Move model to specified device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        model.reset_trajectory()
        print(episode)
        while not done:
            state_tensor = torch.FloatTensor(state).reshape(1, env.get_observation_size()).to(device)
            action_probs, value = model(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            next_state, reward, done, _ = env.step(action.item())
            log_prob = action_dist.log_prob(action)

            model.log_probs.append(log_prob)
            model.values.append(value)
            model.rewards.append(reward)

            state = next_state

        returns = get_discounted_rewards(model.rewards, gamma).to(device)
        bprop_with_log_prob(model.log_probs, model.values, returns, optimizer)



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
            state_tensor = torch.FloatTensor(state).reshape(1, env.get_observation_size()).float()
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


# Policy Evaluation:

def evaluate_model(model, env, num_runs, device):
    """
        for visitation and average_reward
    """
    total_reward = 0.0
    visitation_matrix = np.zeros(env.get_grid_size())

    for _ in range(num_runs):
        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            state_tensor = torch.FloatTensor(state.ravel()).unsqueeze(0).to(device)  # Flatten the state
            action_probs, _ = model(state_tensor)
            action = torch.argmax(action_probs).item()
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward
            state = next_state
            update_visitation_matrix(visitation_matrix, env.current_position)  # Assuming this function is defined

        total_reward += episode_reward

    average_reward = total_reward / num_runs
    return average_reward, visitation_matrix


def evaluate_group(models, num_episodes, env_class, utils_module):
    total_reward = 0
    total_visitation = np.zeros((10, 10))  # Assuming a 10x10 grid
    prob_dists = []

    for model in models:
        env = env_class()
        average_reward = utils_module.evaluate_model(model, env, num_episodes)
        total_reward += average_reward

        # Assuming you have a function to compute visitation frequency
        visitation_matrix = utils_module.compute_visitation_frequency(model, env, num_episodes)
        total_visitation += visitation_matrix

        # Assuming you have a function to get all action probability distributions
        action_prob_dists = utils_module.all_action_prob_dists(env.all_states(), model)
        prob_dists.append(action_prob_dists)

    return total_reward, total_visitation, prob_dists


def grid_state_visitation_eval(policy, env, num_runs):
    """
    Evaluate state visitation frequency of a policy in a gridworld environment.

    Args:
        policy (callable): A function that takes a state and returns an action.
        env: The gridworld environment.
        num_runs (int): Number of trajectories to generate.

    Returns:
        numpy.ndarray: A matrix representing the visitation frequency of each grid cell.
    """
    # Determine the dimensionality of the grid
    grid_size = env.get_grid_size()
    if isinstance(grid_size, int):  # 1D grid
        visitation_matrix = np.zeros(grid_size)
    elif isinstance(grid_size, tuple) and len(grid_size) == 2:  # 2D grid
        visitation_matrix = np.zeros(grid_size)
    else:
        raise ValueError("Unsupported grid size")

    for _ in range(num_runs):
        state = env.reset()
        position = env.get_current_position()  # You might need to implement this method in env
        update_visitation_matrix(visitation_matrix, position)

        done = False
        while not done:
            action = policy(state)
            state, _, done, info = env.step(action)
            update_visitation_matrix(visitation_matrix, info['position'])

    return visitation_matrix


def compute_visitation_matrix(model, env, num_runs):
    visitation_matrix = np.zeros(env.get_grid_size())
    for _ in range(num_runs):
        state = env.reset()
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).reshape(1, -1)
            action_probs, _ = model(state_tensor)
            action = torch.argmax(action_probs).item()
            _, _, done, info = env.step(action)
            x, y = info['position']
            visitation_matrix[x, y] += 1
    return visitation_matrix


def update_visitation_matrix(matrix, position):
    """
    Update the visitation matrix based on the agent's position.

    Args:
        matrix (numpy.ndarray): The visitation matrix.
        position (int or tuple): The agent's position.
    """
    if isinstance(position, int):
        # 1D position
        matrix[position] += 1
    elif isinstance(position, (list, tuple)) and len(position) == 2:
        # 2D position
        x, y = position
        matrix[x, y] += 1
    else:
        raise ValueError("Unsupported position format")


def all_action_prob_dists(states, model):
    """
    Generate action probability distributions for a list of states using a model.

    Args:
        states (list): List of states.
        model (nn.Module): The neural network model.

    Returns:
        numpy.ndarray: Array of action probability distributions for each state.
    """
    prob_dists = []
    for state in states:
        # Flatten the state before passing it to the model
        state_tensor = torch.FloatTensor(state.ravel()).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = model(state_tensor)
            prob_dists.append(action_probs.numpy())
    return np.array(prob_dists)


def kl_divergence(dists):
    """
    Calculate average Kullback-Leibler divergence between pairs of probability distributions.

    Args:
        dists (numpy.ndarray): Array of probability distributions.

    Returns:
        float: Average KL divergence.
    """
    num_dists = dists.shape[0]
    total_kl_div = 0
    count = 0
    for i in range(num_dists):
        for j in range(i + 1, num_dists):
            kl_div_ij = entropy(dists[i], dists[j])
            total_kl_div += kl_div_ij
            count += 1
    return total_kl_div / count if count > 0 else 0


def mean_variance(dists):
    """
    Calculate mean and variance of probability distributions.

    Args:
        dists (numpy.ndarray): Array of probability distributions.

    Returns:
        Tuple(numpy.ndarray, numpy.ndarray): Mean and variance of distributions.
    """
    mean = np.mean(dists, axis=0)
    variance = np.var(dists, axis=0)
    return mean, variance


def mean_variance(dists):
    """
    Calculate mean and variance of probability distributions.

    Args:
        dists (numpy.ndarray): Array of probability distributions.

    Returns:
        Tuple(numpy.ndarray, numpy.ndarray): Mean and variance of distributions.
    """
    mean = np.mean(dists, axis=0)
    variance = np.var(dists, axis=0)
    return mean, variance


def average_cosine_similarity(dists):
    """
    Calculate average cosine similarity between pairs of probability distributions.

    Args:
        dists (numpy.ndarray): Array of probability distributions.

    Returns:
        float: Average cosine similarity.
    """
    sim_matrix = cosine_similarity(dists)
    upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    return np.mean(upper_triangle)


def jensen_shannon_divergence(dists):
    """
    Calculate average Jensen-Shannon divergence between pairs of probability distributions.

    Args:
        dists (numpy.ndarray): Array of probability distributions.

    Returns:
        float: Average Jensen-Shannon divergence.
    """
    num_dists = dists.shape[0]
    total_js_div = 0
    count = 0
    for i in range(num_dists):
        for j in range(i + 1, num_dists):
            js_div_ij = jensenshannon(dists[i], dists[j])
            total_js_div += js_div_ij
            count += 1
    return total_js_div / count if count > 0 else 0


def load_models_from_directory(directory):
    models = []
    for filename in os.listdir(directory):
        if filename.endswith(".pth"):
            model_path = os.path.join(directory, filename)
            model = ActorCritic()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            models.append(model)
    return models
