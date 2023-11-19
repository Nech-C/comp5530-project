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
        self.grid = np.array([0, 0, 0.5, 0.4, 0.2, 0.1, 0, -0.1, -0.2, -0.5, 1, 0 ,0 ])
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
        
        if self.move_count >= 5:
            reward = self.grid[self.current_position]
            done = True

        return self.observe(), reward, done, {}

    def observe(self):
        left = self.current_position - 1
        right = self.current_position + 1
        return self.grid[left:right+1]