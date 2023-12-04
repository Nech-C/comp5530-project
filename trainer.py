import os
import utils
import grid_world
from torch import optim
import torch

import random



def train_n_a2c(num, episode, path):
    if not os.path.exists(path):
        os.makedirs(path)  # Create the directory if it does not exist

    for i in range(num):
        model_save_path = os.path.join(path, f"model_{i}.pth")  # Correct file path
        env = grid_world.SimpleGridEnv()
        utils.train_a2c(env, episode, 0.05, 0.99, model_save_path)  # Train and save each model

def train_single_CPGPO(num, path, config):
    if not os.path.exists(path):
        os.makedirs(path)  # Create the directory if it does not exist

    # Get the list of all model filenames in the directory
    model_filenames = os.listdir(config['reference_models'])

    for i in range(num):
        model_save_path = os.path.join(path, f"model_{i}.pth")  # Correct file path
        env = grid_world.SimpleGridEnv()
        model = utils.ActorCritic()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

        # Choose a random model from the directory as the reference model
        random_model_path = os.path.join(config['reference_models'], random.choice(model_filenames))
        reference_model = utils.ActorCritic()
        reference_model.load_state_dict(torch.load(random_model_path))

        # Train the model using the CPGPO algorithm
        trained_model = utils.single_CPGPO(
            env=env,
            model=model,
            reference_model=reference_model,
            optimizer=optimizer,
            epoch=config['epoch'],
            gamma=config['gamma'],
            starting_n=config['starting_n'],
            n_growth=config['n_growth'],
            max_n=config['max_n'],
            epsilon=config['epsilon'],
            epsilon_decay=config['epsilon_decay'],
            min_epsilon=config['min_epsilon']
        )

        # Save the trained model
        torch.save(trained_model.state_dict(), model_save_path)

def temp_func():
    #model_directory = "./trained_models/trained_a2c_models"
    model_directory = "./trained_models/CPGPO"
    models = utils.load_models_from_directory(model_directory)
    env = grid_world.SimpleGridEnv()
    num_runs = 100

    for i, model in enumerate(models):
        average_reward = utils.evaluate_model(model, env, num_runs)
        print(f"Model {i} Average Reward: {average_reward}")


if __name__ == "__main__":
    #train_n_a2c(50, 6000, "./trained_models/trained_a2c_models")
    temp_func()
    # config = {
    #     'learning_rate': 0.002,
    #     'epoch': 4000,
    #     'gamma': 0.99,
    #     'starting_n': 2,
    #     'n_growth': 0.1,  # Assuming this represents the growth rate per epoch or condition
    #     'max_n': 5,
    #     'epsilon': 0.2,
    #     'epsilon_decay': 0.01,  # Assuming this represents the decay rate per epoch or condition
    #     'min_epsilon': 0.1,
    #     'reference_models': "./trained_models/trained_a2c_models",  # Directory containing reference models
    # }
    #
    # train_single_CPGPO(num=10, path="./trained_models/CPGPO", config=config)

