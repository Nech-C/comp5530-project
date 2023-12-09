# maze.py
import numpy as np
import random
import gym
from gym import spaces
from torch import nn
import torch

# Define constants for the maze and portal pairs
DEF_MAZE = np.array([
    [1, 1, 1, 2, 1, 1, 1, 2, 1, 4],
    [1, 2, 1, 1, 1, 2, 2, 1, 1, 1],
    [1, 1, 1, 2, 1, 2, 1, 1, 2, 3],
    [2, 2, 1, 1, 1, 1, 2, 1, 2, 1],
    [1, 1, 1, 5, 3, 1, 3, 1, 1, 1],
    [1, 1, 1, 1, 3, 1, 2, 1, 1, 1],
    [2, 3, 1, 2, 1, 2, 1, 5, 1, 1],
    [1, 2, 3, 1, 2, 1, 1, 2, 2, 1],
    [4, 1, 3, 1, 1, 1, 2, 2, 1, 1],
    [2, 1, 1, 1, 2, 1, 1, 1, 1, 6]
])
DEF_PORTAL_PAIRS = {(0, 9): (8, 0), (8, 0): (0, 9)}


# Actor-Critic Neural Network for Maze Environment
class ActorCriticMaze(nn.Module):
    def __init__(self):
        super(ActorCriticMaze, self).__init__()
        self.rewards = None
        self.values = None
        self.log_probs = None
        self.action_trajectory = None
        self.state_trajectory = None
        self.fc = nn.Linear(25, 64)
        self.actor = nn.Linear(64, 6)
        self.critic = nn.Linear(64, 1)
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


class ActorCriticMazeV2(nn.Module):
    def __init__(self):
        super(ActorCriticMazeV2, self).__init__()
        self.rewards = None
        self.values = None
        self.log_probs = None
        self.action_trajectory = None
        self.state_trajectory = None
        self.fc1 = nn.Linear(25, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, 6)
        self.critic = nn.Linear(64, 1)
        self.reset_trajectory()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_prob = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_prob, value

    def reset_trajectory(self):
        self.state_trajectory = []  # To store states
        self.action_trajectory = []  # To store actions
        self.log_probs = []  # Log probabilities
        self.values = []  # Value estimates
        self.rewards = []  # Observed rewards


# Maze Environment
class MazeEnv(gym.Env):
    def __init__(self, maze=DEF_MAZE, start=(0, 0), goal=(9, 9), portal_pairs=DEF_PORTAL_PAIRS):
        self.maze = np.array(maze, dtype=int)
        self.current_position = start
        self.goal = goal
        self.portal_pairs = portal_pairs
        self.size = len(maze)
        self.observation_size = 5
        self.facing = 'up'
        self.num_steps = 0
        self.invalid_moves = 0

        # for evaluation:
        self.visitation_matrix = np.zeros_like(self.maze)
        self.action_sequence = []
        self.cumulative_reward = 0
        # Define action space and mapping
        self.actions = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1),
            'activate': 'activate',
            'jump': 'jump'
        }
        self.action_mapping = {
            0: 'up',
            1: 'down',
            2: 'left',
            3: 'right',
            4: 'activate',
            5: 'jump'
        }

        # Define observation and action spaces
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0, high=6, shape=(self.observation_size ** 2,), dtype=int)

    def step(self, action):
        # self.print_maze_with_agent()
        self.num_steps += 1
        done = False
        reward = 0
        # Handle action
        action = self.action_mapping[action]
        if action in ['up', 'down', 'left', 'right']:
            if not self.move(self.actions[action]):
                self.invalid_moves += 1
                reward = -0.5
                if self.invalid_moves >= 10:
                    done = True
            self.facing = action
        elif action == 'activate':
            self.activate_portal()
        elif action == 'jump':
            self.jump_over()

        # Get observation, reward, done, info
        observation = self.get_observation()
        if not done:
            if self.current_position == self.goal:
                reward = 1.0
                print("goal!")
                done = True
            else:
                if self.num_steps > 150:
                    done = True
                    reward = -0.5

        # Increment the visitation count
        self.visitation_matrix[self.current_position] += 1
        # Record the action taken
        self.action_sequence.append(self.action_mapping[action])
        info = {"pos:": self.current_position}
        if done:
            info['visitation_matrix'] = self.visitation_matrix
            info['action_sequence'] = self.action_sequence
            info['cumulative_reward'] = self.cumulative_reward
            info['total_steps'] = self.num_steps

        return observation, reward, done, info

    def move(self, direction):
        new_position = (self.current_position[0] + direction[0], self.current_position[1] + direction[1])
        if not self.is_within_bounds(new_position):
            return False
        else:
            grid_type = self.maze[new_position]
            if grid_type == 2 or grid_type == 5:  # Wall or gap
                return True
            elif grid_type == 3:  # Breakable wall
                self.maze[new_position] = 1
                return True
            self.current_position = new_position
            return True

    def is_valid_position(self, position):
        x, y = position
        return 0 <= x < self.size and 0 <= y < self.size and self.maze[x, y] in [1, 4]

    def activate_portal(self):
        if self.maze[self.current_position] == 4:
            self.current_position = self.portal_pairs.get(self.current_position, self.current_position)

    def jump_over(self):
        dx, dy = self.actions[self.facing]
        over_position = (self.current_position[0] + 2 * dx, self.current_position[1] + 2 * dy)
        between_position = (self.current_position[0] + dx, self.current_position[1] + dy)
        if self.is_within_bounds(between_position) and self.is_within_bounds(over_position):
            grid_type_between = self.maze[between_position]
            if grid_type_between == 5 and self.is_valid_position(over_position):
                self.current_position = over_position
            elif grid_type_between == 2 and random.random() < 0.35 and self.is_valid_position(over_position):
                self.current_position = over_position

    def is_within_bounds(self, position):
        x, y = position
        return 0 <= x < self.size and 0 <= y < self.size

    def get_observation(self):
        obs = np.zeros((self.observation_size, self.observation_size), dtype=int)

        # Calculate the bounds of the observation window centered around the agent
        start_x = self.current_position[0] - 2
        start_y = self.current_position[1] - 2

        for i in range(self.observation_size):
            for j in range(self.observation_size):
                # Map the observation window to the maze coordinates
                maze_x = start_x + i
                maze_y = start_y + j

                if 0 <= maze_x < self.size and 0 <= maze_y < self.size:
                    # Copy the maze value if within bounds
                    obs[i, j] = self.maze[maze_x, maze_y]
                else:
                    # Fill with 0 if out of bounds
                    obs[i, j] = 0

        return obs.flatten()

    def reset(self):
        self.current_position = (0, 0)
        self.facing = 'up'
        self.num_steps = 0
        self.maze = np.array(DEF_MAZE, dtype=int)
        self.invalid_moves = 0
        self.visitation_matrix = np.zeros_like(self.maze)
        self.action_sequence = []
        self.cumulative_reward = 0
        return self.get_observation()

    def get_observation_size(self):
        return self.observation_size ** 2

    def get_grid_size(self):
        return self.maze.shape

    def play(self):
        print("Starting position:", self.current_position)
        print("Goal:", self.goal)
        print("Use 'w' for up, 's' for down, 'a' for left, 'd' for right, 'p' to activate portal, 'j' to jump.")
        print("Enter 'exit' to stop playing.")

        while True:
            self.print_maze_with_agent()
            action = input("Enter your action: ").strip().lower()

            if action == 'exit':
                print("Exiting the game.")
                break

            # Mapping input to actions
            action_map = {'w': 0, 's': 1, 'a': 2, 'd': 3, 'p': 4, 'j': 5}
            if action in action_map:
                observation, reward, done, _ = self.step(action_map[action])
                print("Reward:", reward)
                if done:
                    print("Congratulations! You've reached the goal!")
                    break
            else:
                print("Invalid action. Please try again.")

    def print_maze_with_agent(self):
        maze_copy = np.array(self.maze, copy=True)
        maze_copy[self.current_position] = 7
        print(maze_copy)
        print("obs:", self.get_observation().reshape((5, 5)))
        print("pos:", self.current_position)


# Example usage
if __name__ == "__main__":
    env = MazeEnv()
    env.play()
