# grid_world_2d.py
import gym
import torch
from gym import spaces
import numpy as np
from torch import nn


class ActorCritic2D(nn.Module):
    def __init__(self):
        super(ActorCritic2D, self).__init__()
        self.rewards = None
        self.values = None
        self.log_probs = None
        self.action_trajectory = None
        self.state_trajectory = None
        self.fc = nn.Linear(9, 32)
        self.actor = nn.Linear(32, 4)
        self.critic = nn.Linear(32, 1)
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


class SimpleGridEnv2D(gym.Env):
    def __init__(self):
        super(SimpleGridEnv2D, self).__init__()

        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right
        self.grid_size = (10, 10)  # 10x10 grid
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3, 3), dtype=np.float32)

        self.grid = np.array([
            [0.5, -0.2, 0.1, 0.4, -0.3, 0.0, -0.5, 0.2, 0.0, -0.1],
            [0.3, -0.4, 0.2, 0.1, -0.2, 0.4, -0.2, 1.0, 0.3, -0.5],
            [-0.1, 0.0, 0.5, -0.2, 0.1, 0.4, -0.3, 0.0, -0.5, 0.2],
            [-0.0, -0.1, -0.3, -0.4, 0.2, 0.4, -0.2, 0.1, 0.5, -0.1],
            [-0.2, -0.4, -0.2, -0.1, 0.3, -0.5, 0.1, 0.0, -0.4, 0.2],
            [0.5, 0.6, 0.0, -0.1, 0.5, -0.2, 0.1, 0.4, -0.3, 0.0],
            [0.5, 0.4, 0.3, 0.0, -0.5, 0.2, 0.0, -0.1, 0.3, -0.4],
            [0.4, 0.2, 0.1, 0.3, -0.4, 0.2, 0.1, -0.2, 0.4, -0.2],
            [0.2, 0.1, 0.5, 0.1, 0.0, -0.5, 0.2, 0.4, -0.2, 0.1],
            [0.1, 0.3, -0.4, 0.2, 0.4, -0.2, 0.1, 0.5, -0.1, 0.0]
        ])
        self.current_position = [1, 1]  # Starting position
        self.step_count = 0

    def reset(self):
        self.current_position = [1, 1]
        self.step_count = 0
        return self.get_local_observation()

    def step(self, action):
        # Restrict movement so that the agent stays within the bounds where a 3x3 observation is possible
        if action == 0 and self.current_position[0] > 1:
            self.current_position[0] -= 1
        elif action == 1 and self.current_position[0] < self.grid_size[0] - 2:
            self.current_position[0] += 1
        elif action == 2 and self.current_position[1] > 1:
            self.current_position[1] -= 1
        elif action == 3 and self.current_position[1] < self.grid_size[1] - 2:
            self.current_position[1] += 1

        self.step_count += 1
        reward = self.grid[self.current_position[0], self.current_position[1]]
        done = self.step_count >= 10 # Adjust as per your requirement

        return self.get_local_observation(), reward, done, {'position': self.current_position}

    def get_local_observation(self):
        x, y = self.current_position
        observation = self.grid[x - 1:x + 2, y - 1:y + 2]
        return observation

    def all_states(self):
        # Return all possible 3x3 observations
        all_observations = []
        for i in range(1, self.grid_size[0] - 1):
            for j in range(1, self.grid_size[1] - 1):
                observation = self.grid[i - 1:i + 2, j - 1:j + 2]
                all_observations.append(observation)
        return all_observations

    def get_grid_size(self):
        return self.grid_size

    def get_observation_size(self):
        return 9



def run():
    env = SimpleGridEnv2D()
    obs = env.reset()

    while True:
        print("Observation:", obs)
        action_input = input("Enter 'u' to move up, 'd' to move down, 'l' to move left, or 'r' to move right: ")

        action = {'u': 0, 'd': 1, 'l': 2, 'r': 3}.get(action_input)
        if action is not None:
            obs, reward, done, _ = env.step(action)
            if done:
                print("Game Over! Final Reward:", reward)
                break
        else:
            print("Invalid input. Try again.")


if __name__ == "__main__":
    run()
