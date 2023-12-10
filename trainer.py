# trainer.py
import os
import torch
import cpgpo
import utils
import random
import json
import numpy
from rl_env.maze import MazeEnv, ActorCriticMaze, ActorCriticMazeV2
from datetime import datetime

A2C_CONFIG = {
    'learning_rate': 0.0007,
    'epoch': 8000,
    'gamma': 0.99,
}

cpgpo_config_v1 = {
    'learning_rate': 0.0007,
    'epoch': 8000,
    'gamma': 0.99,
    'starting_n': 1,
    'n_growth': 0.1,  # Assuming this represents the growth rate per epoch or condition
    'max_n': 1,
    'epsilon': 0.8,
    'epsilon_decay': 0.004,  # Assuming this represents the decay rate per epoch or condition
    'min_epsilon': 0.0,
    'reference_models': "./trained_models/2dGridWorld/trained_a2c_models",  # Directory containing reference models
    'path_a2c': "./trained_models/maze/trained_a2c_models",
    'path_cpgpo': "./trained_models/maze/trained_n1_cpgpo_models_v1",
}

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)

def train_maz_a2c(num_agents, path_a2c, config=A2C_CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(path_a2c):
        os.makedirs(path_a2c)

    for i in range(num_agents):
        env = MazeEnv()  # Define your maze settings here
        model = ActorCriticMaze().to(device)  # Move model to device
        trained_model = utils.train_a2c(env, model, config['epoch'], config['learning_rate'], config['gamma'], device)
        model_save_path = os.path.join(path_a2c, f"a2c_maze_model_{i}.pth")
        torch.save(trained_model.state_dict(), model_save_path)


def train_maz_a2c_v2(num_agents, path_a2c, config=A2C_CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(path_a2c):
        os.makedirs(path_a2c)

    for i in range(num_agents):
        env = MazeEnv()  # Define your maze settings here
        model = ActorCriticMazeV2().to(device)  # Move model to device
        trained_model = utils.train_a2c(env, model, config['epoch'], config['learning_rate'], config['gamma'], device)
        model_save_path = os.path.join(path_a2c, f"a2c_maze_model_{i}.pth")
        torch.save(trained_model.state_dict(), model_save_path)

def train_maze_cpgpo(num_agents, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Check and create CPGPO path
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_cpgpo = os.path.join(config['path_cpgpo'], current_time)
    os.makedirs(path_cpgpo, exist_ok=True)

    # Save the configuration
    config_path = os.path.join(path_cpgpo, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Train CPGPO Agents
    for i in range(num_agents):
        # Refresh the list of all models from the reference directory before each training
        all_models = []
        for root, dirs, files in os.walk(config['reference_directory']):
            for filename in files:
                if filename.endswith(".pth"):
                    model_path = os.path.join(root, filename)
                    model = ActorCriticMaze().to(device)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.eval()
                    all_models.append(model)

        env = MazeEnv()
        model = ActorCriticMaze().to(device)
        reference_model = random.choice(all_models).to(device)
        trained_model = cpgpo.single_CPGPO(env, model, reference_model, config, device)
        model_save_path = os.path.join(path_cpgpo, f"cpgpo_maze_model_{i}.pth")
        torch.save(trained_model.state_dict(), model_save_path)

def train_maze_cpgpo_v2(num_agents, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Check and create CPGPO path
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_cpgpo = os.path.join(config['path_cpgpo'], current_time)
    os.makedirs(path_cpgpo, exist_ok=True)

    # Save the configuration
    config_path = os.path.join(path_cpgpo, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Train CPGPO Agents
    for i in range(num_agents):
        # Refresh the list of all models from the reference directory before each training
        all_models = []

        for root, dirs, files in os.walk(config['reference_directory']):
            for filename in files:
                if filename.endswith(".pth"):
                    model_path = os.path.join(root, filename)
                    model = ActorCriticMazeV2().to(device)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.eval()
                    all_models.append(model)
        print(len(all_models))
        env = MazeEnv()
        model = ActorCriticMazeV2().to(device)
        reference_model = random.choice(all_models).to(device)
        trained_model = cpgpo.single_CPGPO(env, model, reference_model, config, device)
        model_save_path = os.path.join(path_cpgpo, f"cpgpo_maze_model_{i}.pth")
        torch.save(trained_model.state_dict(), model_save_path)



if __name__ == "__main__":
    cpgpo_config_v2 = {
        'learning_rate': 0.0007,
        'epoch': 8000,
        'gamma': 0.99,
        'starting_n': 1,
        'n_growth': 0.1,  # Assuming this represents the growth rate per epoch or condition
        'max_n': 1,
        'epsilon': 0.8,
        'epsilon_decay': 0.004,  # Assuming this represents the decay rate per epoch or condition
        'min_epsilon': 0.0,
        'reference_directory': "./trained_models/maze/v2",  # Directory containing reference models
        'path_cpgpo': "./trained_models/maze/trained_n1_cpgpo_models_v1",
    }
    #train_maz_a2c(num_agents=50, path_a2c="trained_models/maze/v1/trained_a2c_models")
    train_maze_cpgpo_v2(1, cpgpo_config_v2)
