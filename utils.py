# utils.py: 
import gym
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

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
        self.penalties = []
        self.log_probs = []
        self.values = []
        self.rewards = []
    
def update_tensorboard(writer, episode: int, loss: float, reward: any):
        """ the method updates the tensorboard

        Args:
            episode (int): the eposode number
            loss (float): policy update loss 
        """
        
        if loss is not None:
            writer.add_scalar('Loss/train', loss, episode)
        if reward is not None:
            writer.add_scalar('Reward/train', reward, episode)

def load_reference_models(model_list):
    reference_models = []
    random.shuffle(model_list)
    for i in min(len(model_list), 5):
        model = ActorCritic()
        model.load_state_dict(torch.load(model_list[i]))
        model.eval()
        reference_models.append(model)
    return reference_models