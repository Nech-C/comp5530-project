# trainer.py
import os
import numpy as np
import torch
from maze import MazeEnv, ActorCriticMaze
import utils
import grid_world
from grid_world_2d import SimpleGridEnv2D, ActorCritic2D
from torch import optim

import random


def train_n_a2c(num, episode, path):
    if not os.path.exists(path):
        os.makedirs(path)  # Create the directory if it does not exist

    for i in range(num):
        model_save_path = os.path.join(path, f"model_{i}.pth")  # Correct file path
        env = grid_world.SimpleGridEnv()
        utils.train_a2c(env, episode, 0.05, 0.99, model_save_path)  # Train and save each model


def temp_func():
    # model_directory = "./trained_models/trained_a2c_models"
    model_directory = "./trained_models/CPGPO"
    models = utils.load_models_from_directory(model_directory)
    env = grid_world.SimpleGridEnv()
    num_runs = 100

    for i, model in enumerate(models):
        average_reward = utils.evaluate_model(model, env, num_runs)
        print(f"Model {i} Average Reward: {average_reward}")


def evaluate_group(models, num_episodes, env_class):
    total_reward = 0
    total_visitation = np.zeros(env_class().get_grid_size())
    prob_dists = []
    for model in models:
        env = env_class()
        average_reward, visitation_matrix = utils.evaluate_model(model, env, num_episodes)
        total_reward += average_reward
        total_visitation += visitation_matrix
        prob_dists.append(utils.all_action_prob_dists(env.all_states(), model))
    return total_reward, total_visitation, prob_dists


def train_and_eval_2d(num_agents, num_episodes, path_a2c, path_cpgpo, config):
    if not os.path.exists(path_a2c):
        os.makedirs(path_a2c)

    if not os.path.exists(path_cpgpo):
        os.makedirs(path_cpgpo)

    # Train A2C Agents
    a2c_models = []
    for i in range(num_agents):
        env = SimpleGridEnv2D()
        model = ActorCritic2D()
        trained_model = utils.train_a2c(env, model, num_episodes, config['learning_rate'], config['gamma'])
        model_save_path = os.path.join(path_a2c, f"a2c_model_{i}.pth")
        torch.save(trained_model.state_dict(), model_save_path)
        a2c_models.append(trained_model)

    # Train CPGPO Agents
    cpgpo_models = []
    all_models = a2c_models.copy()
    for i in range(num_agents):
        env = SimpleGridEnv2D()
        model = ActorCritic2D()
        reference_model = random.choice(all_models)
        trained_model = utils.single_CPGPO(env, model, reference_model, config)

        model_save_path = os.path.join(path_cpgpo, f"cpgpo_model_{i}.pth")
        torch.save(trained_model.state_dict(), model_save_path)
        cpgpo_models.append(trained_model)
        all_models.append(trained_model)  # Add to the pool for subsequent training

    # Evaluate and Compare
    results = {
        'a2c': {
            'average_reward': 0,
            'visitation': np.zeros(env.get_grid_size()),
            'similarity_score': 0
        },
        'cpgpo': {
            'average_reward': 0,
            'visitation': np.zeros(env.get_grid_size()),
            'similarity_score': 0
        }
    }

    # Evaluate A2C models
    total_reward_a2c, total_visitation_a2c, prob_dists_a2c = evaluate_group(a2c_models, num_episodes, SimpleGridEnv2D)

    total_reward_a2c, total_visitation_a2c, prob_dists_a2c = evaluate_group(a2c_models, num_episodes, SimpleGridEnv2D)
    results['a2c']['average_reward'] = total_reward_a2c / num_agents
    results['a2c']['visitation'] = total_visitation_a2c / num_agents
    # results['a2c']['similarity_score'] = utils.average_cosine_similarity( prob_dists_a2c)
    # Evaluate CPGPO models
    total_reward_cpgpo, total_visitation_cpgpo, prob_dists_cpgpo = evaluate_group(cpgpo_models, num_episodes,
                                                                                  SimpleGridEnv2D)
    results['cpgpo']['average_reward'] = total_reward_cpgpo / num_agents
    results['cpgpo']['visitation'] = total_visitation_cpgpo / num_agents
    # results['cpgpo']['similarity_score'] = utils.average_cosine_similarity(prob_dists_cpgpo)

    # Save results to a file
    with open('training_evaluation_results.txt', 'w') as file:
        file.write(str(results))

def train_and_eval_maze(num_agents, num_episodes, path_a2c, path_cpgpo, config):
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(path_a2c):
        os.makedirs(path_a2c)

    if not os.path.exists(path_cpgpo):
        os.makedirs(path_cpgpo)

    # Train A2C Agents
    # a2c_models = []
    # for i in range(num_agents):
    #     env = MazeEnv()  # Define your maze settings here
    #     model = ActorCriticMaze().to(device)  # Move model to device
    #     trained_model = utils.train_a2c(env, model, num_episodes, config['learning_rate'], config['gamma'], device)
    #     model_save_path = os.path.join(path_a2c, f"a2c_maze_model_{i}.pth")
    #     torch.save(trained_model.state_dict(), model_save_path)
    #     a2c_models.append(trained_model)


    # after training a2c
    a2c_models = []
    for filename in os.listdir(path_a2c):
        if filename.endswith(".pth"):
            model_path = os.path.join(path_a2c, filename)
            model = ActorCriticMaze().to(device)  # Initialize and move model to device
            model.load_state_dict(torch.load(model_path))
            model.eval()  # Set model to evaluation mode
            a2c_models.append(model)

    # Train CPGPO Agents# Train CPGPO Agents
    cpgpo_models = []
    all_models = a2c_models.copy()
    for i in range(num_agents):
        env = MazeEnv()  # Define your maze settings here
        model = ActorCriticMaze().to(device)  # Move model to device
        reference_model = random.choice(all_models).to(device)  # Move reference model to device
        trained_model = utils.single_CPGPO(env, model, reference_model, config, device)
        model_save_path = os.path.join(path_cpgpo, f"cpgpo_maze_model_{i}.pth")
        torch.save(trained_model.state_dict(), model_save_path)
        cpgpo_models.append(trained_model)
        all_models.append(trained_model)


    # Evaluate A2C and CPGPO models
    results = {'a2c': {'average_reward': 0, 'visitation': np.zeros((10, 10))},
               'cpgpo': {'average_reward': 0, 'visitation': np.zeros((10, 10))}}

    # Evaluate and Update Results for A2C and CPGPO
    for group_name, models in zip(['a2c', 'cpgpo'], [a2c_models, cpgpo_models]):
        group_total_reward = 0
        group_total_visitation = np.zeros(env.get_grid_size())

        for model in models:
            model = model.to(device)  # Move model to device for evaluation
            average_reward, visitation_matrix = utils.evaluate_model(model, env, num_episodes, device)
            group_total_reward += average_reward
            group_total_visitation += visitation_matrix

        # Calculate average results
        results[group_name]['average_reward'] = group_total_reward / num_agents
        results[group_name]['visitation'] = group_total_visitation / num_agents

    with open('maze_training_evaluation_results.txt', 'w') as file:
        file.write(str(results))

    return results



if __name__ == "__main__":
    # train_n_a2c(50, 6000, "./trained_models/trained_a2c_models")
    # temp_func()
    # 0.0009 for 2ac
    config = {
        'learning_rate': 0.0007,
        'epoch': 3000,
        'gamma': 0.99,
        'starting_n': 1,
        'n_growth': 0.1,  # Assuming this represents the growth rate per epoch or condition
        'max_n': 1,
        'epsilon': 0.25,
        'epsilon_decay': 0.01,  # Assuming this represents the decay rate per epoch or condition
        'min_epsilon': 0.001,
        'reference_models': "./trained_models/2dGridWorld/trained_a2c_models",  # Directory containing reference models
    }
    #
    # train_single_CPGPO(num=10, path="./trained_models/CPGPO", config=config)
    # train_and_eval_2d(num_agents=30, num_episodes=3000, path_a2c="./trained_models/2dGridWorld/trained_a2c_models",
                      #path_cpgpo="./trained_models/2dGridWorld/trained_cpgpo_models", config=config)

    results = train_and_eval_maze(num_agents=25, num_episodes=3000, path_a2c="./trained_models/maze/trained_a2c_models",
                      path_cpgpo="./trained_models/maze/trained_cpgpo_models", config=config)
    print(results)