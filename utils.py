# utils.py:

import torch
import torch.nn as nn
import random
import os


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

def save_model(model, model_dir, filename):
    """Save the current model state."""
    model_path = f"{model_dir}/{filename}"
    torch.save(model.state_dict(), model_path)


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



