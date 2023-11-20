# n_step_cp.py:
import gym
import torch
import torch.nn as nn
import torch.optim as optim
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
n_step = 5
epsilon = 0.1  # Clipping parameter for PPO

# Initialize environment and models
env = SimpleGridEnv()
model = ActorCritic()
reference_model = load_reference_models(pretrained_model_list)[0]  # Load reference model
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
writer = SummaryWriter(log_dir=f'./runs/{__file__}', flush_secs=1)

# Initialize CP queues
old_model_cp_queue = deque([1.0] * n_step, maxlen=n_step)
new_model_cp_queue = deque([1.0] * n_step, maxlen=n_step)

def get_discounted_rewards(rewards):
    returns = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns, dtype=torch.float)
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)
    return returns

def update_model(log_probs, values, returns):
    actor_loss = []
    critic_loss = []

    # Update CP and calculate actor and critic losses
    for log_prob, value, R in zip(log_probs, values, returns):
        old_model_cp = np.prod(old_model_cp_queue)
        new_model_cp = np.prod(new_model_cp_queue)

        # Only update if CPs are sufficiently different
        if abs(new_model_cp - old_model_cp) > epsilon:
            advantage = R - value.item()
            old_prob = old_model_cp_queue[-1]
            ratio = torch.exp(log_prob - old_prob)  # Ratio of new and old probabilities
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
            actor_loss.append(-torch.min(surr1, surr2))  # Clipped loss
            critic_loss.append(nn.functional.mse_loss(value.flatten(), torch.tensor([R], dtype=torch.float)))

            # Update CP queues
            old_model_cp_queue.append(old_prob)
            new_model_cp_queue.append(log_prob.item())

    if actor_loss and critic_loss:
        total_loss = torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

def train():
    for episode in range(epoch):
        state = env.reset()
        done = False
        model.reset_trajectory()

        while not done:
            state_tensor = torch.FloatTensor(state).reshape(1, 3).float()
            action_prob, value = model(state_tensor)

            # Sample action from the new model
            action_dist = torch.distributions.Categorical(action_prob)
            action = action_dist.sample()

            next_state, reward, done, _ = env.step(action.item())

            # Store log probabilities, values, and rewards
            log_prob = action_dist.log_prob(action)
            model.log_probs.append(log_prob)
            model.values.append(value)
            model.rewards.append(reward)

            state = next_state

        returns = get_discounted_rewards(model.rewards)
        update_model(model.log_probs, model.values, returns)

if __name__ == "__main__":
    train()
