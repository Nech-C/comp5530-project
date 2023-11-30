# grid_world.py:
import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class SimpleGridEnv(gym.Env):
    def __init__(self):
        super(SimpleGridEnv, self).__init__()

        self.action_space = spaces.Discrete(2)  # 0: Left, 1: Right
        self.observation_space = spaces.Box(low=-10, high=10, shape=(1, 3), dtype=np.int64)

        self.current_position = 6
        self.grid = np.array([0, 0, 0.5, 0.4, 0.2, 0.1, 0, -0.1, -0.2, -0.5, 1, 0, 0])
        self.move_count = 0

    def reset(self):
        self.current_position = 6
        self.move_count = 0
        return self.observe()

    def step(self, action):
        if action == 0:
            self.current_position = self.current_position - 1
        elif action == 1:
            self.current_position = self.current_position + 1

        self.move_count += 1

        reward = 0
        done = False

        if self.move_count >= 4:
            reward = self.grid[self.current_position]
            done = True

        return self.observe(), reward, done, {'position:': self.current_position}

    def observe(self):
        left = self.current_position - 1
        right = self.current_position + 1
        return self.grid[left:right + 1]

    def all_states(self):
        return [self.grid[i - 1:i + 2] for i in range(1, len(self.grid) - 1)]

    def get_grid_size(self):
        return len(self.grid)


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


if __name__ == "__main__":
    run()
