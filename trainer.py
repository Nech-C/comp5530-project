# trainer.py
import os
import torch
import cpgpo
import utils
import random
import json
import numpy
from maze import MazeEnv, ActorCriticMaze
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


def train_maze_cpgpo(num_agents, config):
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(config['path_a2c']):
        return

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_cpgpo = os.path.join(config['path_cpgpo'], current_time)
    if not os.path.exists(path_cpgpo):
        os.makedirs(path_cpgpo)

    config_path = os.path.join(path_cpgpo, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    a2c_models = []
    for filename in os.listdir(config['path_a2c']):
        if filename.endswith(".pth"):
            model_path = os.path.join(config['path_a2c'], filename)
            model = ActorCriticMaze().to(device)  # Initialize and move model to device
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()  # Set model to evaluation mode
            a2c_models.append(model)

    random.shuffle(a2c_models)
    # Train CPGPO Agents# Train CPGPO Agents
    all_models = a2c_models[0:4].copy()
    for i in range(num_agents):
        env = MazeEnv()  # Define your maze settings here
        model = ActorCriticMaze().to(device)  # Move model to device
        reference_model = random.choice(all_models).to(device)  # Move reference model to device
        trained_model = cpgpo.single_CPGPO(env, model, reference_model, config, device)
        model_save_path = os.path.join(path_cpgpo, f"cpgpo_maze_model_{i}.pth")
        torch.save(trained_model.state_dict(), model_save_path)
        all_models.append(trained_model)


if __name__ == "__main__":
    train_maz_a2c(num_agents=50, path_a2c="./trained_models/maze/trained_a2c_models")
    # train_maze_cpgpo(num_agents=50, config=cpgpo_config_v1)
