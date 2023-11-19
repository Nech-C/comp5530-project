import gym
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import numpy as np

from gym import spaces
from grid_world import SimpleGridEnv 

pretrained_model_list = ["./trained_moddels/a2c_model1.pth"]
n_models = 5

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(3, 16)
        self.actor = nn.Linear(16, 2)
        self.critic = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        action_prob = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_prob, value


def run():
    env = SimpleGridEnv()
    obs = env.reset()

    while True:
        print("Observation:", obs)
        action_input = input("Enter 'l' to move left or 'r' to move right: ")
        
        if action_input == 'l':
            action = 0
        elif action_input == 'r':
            action = 1
        else:
            print("Invalid input. Try again.")
            continue
        
        obs, reward, done, _ = env.step(action)
        
        if done:
            print("Game Over! Final Reward:", reward)
            break

def train_model():
    # Initialize environment and model
    env = SimpleGridEnv()
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    gamma = 0.99

    # Training loop
    for episode in range(1000):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        done = False

        while not done:
            #print(state)
            state_tensor = torch.FloatTensor(state).reshape(1, 3).float()

            action_prob, value = model(state_tensor)

            # Sample action from the distribution
            action_dist = torch.distributions.Categorical(action_prob)
            action = action_dist.sample()
            
            next_state, reward, done, _ = env.step(action.item())

            log_prob = action_dist.log_prob(action)
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state

        # Calculate discounted rewards
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Compute loss and backpropagate
        actor_loss = []
        critic_loss = []
        for log_prob, value, R in zip(log_probs, values, returns):
            advantage = R - value.item()
            
            actor_loss.append(-log_prob * advantage)
            critic_loss.append(nn.functional.mse_loss(value.flatten(), torch.tensor([R], dtype=torch.float)))

        loss = torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode}, Total Reward: {sum(rewards)}")
    torch.save(model.state_dict(), "actor_critic_model.pth")
    print("Model saved.")

def cumulative_similarity_penalty(pretrained_model, state_trajectory, action_trajectory):
    cum_penalty = 1  # Initialize cumulative penalty
    decay_factor = 0.5  # Decay factor for the exponential term
    with torch.no_grad():
        for state, action in zip(state_trajectory, action_trajectory):
            state_tensor = torch.FloatTensor(state).reshape(1, 3).float()
            old_action_prob, _ = pretrained_model(state_tensor)
            penalty = torch.exp(-decay_factor * (1 - old_action_prob[0][action]))  # Exponential decay
            cum_penalty *= penalty  # Accumulate penalty
    return cum_penalty


def update_params(reference_models, model):
    pass

# Function to train a new model with cumulative similarity penalty
def train_model_with_custom_algo():
    # Load the pretrained model(s)
    reference_models = []
    random.shuffle(pretrained_model_list)
    
    for i in min(len(pretrained_model_list), 5):
        model = ActorCritic()
        model.load_state_dict(torch.load(pretrained_model_list[i]))
        model.eval()
        reference_models.append(model)
    
    # Initialize a new ActorCritic model for training
    env = SimpleGridEnv()
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    gamma = 0.99 # discount value
    lambda_penalty = 0  # Weight for the similarity penalty term
    lambda_penalty_decay = 0.002
    starting_lambda = 1
    
    for episode in range(3000):
        if episode % 100 == 0:
            lambda_penalty = max(starting_lambda * (1. / (1 + lambda_penalty_decay * episode)), 0.01)
        state = env.reset()
        state_trajectory = []  # To store states
        action_trajectory = []  # To store actions
        penalties = []
        log_probs = []
        values = []
        rewards = []
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).reshape(1, 3).float()
            action_prob, value = model(state_tensor)

            # Sample action from the new model
            action_dist = torch.distributions.Categorical(action_prob)
            action = action_dist.sample()

            next_state, reward, done, _ = env.step(action.item())
            
            # Store state and action for cumulative penalty
            state_trajectory.append(state)
            action_trajectory.append(action.item())

            log_prob = action_dist.log_prob(action)
            
            # Compute the cumulative similarity penalty
            # !!!!!!!!!penalty should be calculated per step not as a cumulated value FIX THIS!
            penalties.append(cumulative_similarity_penalty(reference_models, state_trajectory, action_trajectory))

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state

        # Calculate discounted rewards
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Compute loss and backpropagate
        actor_loss = []
        critic_loss = []
        for log_prob, value, R, penalty in zip(log_probs, values, returns, penalties):
            advantage = R - value.item()
            actor_loss.append(-log_prob * advantage * (1 - lambda_penalty * penalty))  # Modified loss with cumulative penalty
            critic_loss.append(nn.functional.mse_loss(value.flatten(), torch.tensor([R], dtype=torch.float)))

        loss = torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode}, Total Reward: {sum(rewards)}, lambda_penalty: {lambda_penalty}")

if __name__ == "__main__":
    # Uncomment to train the initial model
    # train_model()

    # Uncomment to train the model with the custom algorithm
    train_model_with_custom_algo()