import numpy as np
import torch
import statistics
import os
import numpy as np
import matplotlib.pyplot as plt
from rl_env.maze import MazeEnv, ActorCriticMaze
from collections import Counter

base_maze_dir = "./trained_models/maze"
a2c_dir_name_v1 = "trained_a2c_models"
cpgpo_dir_names_v1 = [
    "trained_n1_cpgpo_models_v1",
    "trained_n2_cpgpo_models_v1",
    "trained_n3_cpgpo_models_v1",
    "trained_n4_cpgpo_models_v1"
]
def evaluate_maze_agent(model, num_episodes):
    """
        model: the rl agent to be evaluated
        num_episodes: the number of episodes the agent needs to complete in env
    """
    env = MazeEnv()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visitation_matrices = []
    action_sequences = []
    num_steps = []
    rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_rewards = 0

        while not done:
            state_tensor = torch.FloatTensor(state).reshape(1, env.get_observation_size()).to(device)
            action_probs, _ = model(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            next_state, reward, done, info = env.step(action.item())

            episode_rewards += reward
            state = next_state

            if done:
                visitation_matrices.append(info['visitation_matrix'])
                action_sequences.append(info['action_sequence'])
                num_steps.append(info['total_steps'])
                rewards.append(episode_rewards)

    return visitation_matrices, action_sequences, num_steps, rewards


def evaluate_maze_agents(model_class, num_episodes, directories):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_steps = []
    all_rewards = []
    all_visitations = []
    all_action_sequences = []

    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith(".pth"):
                model_path = os.path.join(directory, filename)
                model = model_class().to(device)
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)

                visitation, act_seq, steps, rewards = evaluate_maze_agent(model, num_episodes)

                all_steps.extend(steps)
                all_rewards.extend(rewards)
                all_visitations.extend(visitation)
                all_action_sequences.extend(act_seq)

    return all_visitations, all_action_sequences, all_steps, all_rewards


import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def evaluate_maze_stats(visit_list, action_seq_list, step_num_list, reward_list, folder_path):
    # Creating the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Calculate mean, variance, max, min, and specific percentiles for steps and rewards
    stats = {
        'Step Mean': np.mean(step_num_list),
        'Step Variance': np.var(step_num_list),
        'Step Max': np.max(step_num_list),
        'Step Min': np.min(step_num_list),
        'Step Top 15 Min': sorted(step_num_list)[:15],
        'Step 0.01% Percentile': np.percentile(step_num_list, 0.01),
        'Step 0.05% Percentile': np.percentile(step_num_list, 0.05),
        'Step 0.1% Percentile': np.percentile(step_num_list, 0.1),
        'Step 1% Percentile': np.percentile(step_num_list, 1),
        'Reward Mean': np.mean(reward_list),
        'Reward Variance': np.var(reward_list),
        'Reward Max': np.max(reward_list),
        'Reward Min': np.min(reward_list),
        'Reward 0.01% Percentile': np.percentile(reward_list, 0.01),
        'Reward 0.05% Percentile': np.percentile(reward_list, 0.05),
        'Reward 0.1% Percentile': np.percentile(reward_list, 0.1),
        'Reward 1% Percentile': np.percentile(reward_list, 1),
    }

    # Function to plot and save histogram
    def plot_histogram(data, title, file_name):
        plt.figure()
        plt.hist(data, bins=20)
        plt.title(title)
        plt.savefig(os.path.join(folder_path, file_name))
        plt.show()
        plt.close()

    # Plotting and saving histograms for steps and rewards
    plot_histogram(step_num_list, "Step Distribution", "step_distribution.png")
    plot_histogram(reward_list, "Reward Distribution", "reward_distribution.png")

    # Most common actions
    action_counts = Counter(action for seq in action_seq_list for action in seq)
    most_common_actions = action_counts.most_common()

    # Action transition matrix
    num_actions = 6  # Assuming 6 possible actions
    action_transition_matrix = np.zeros((num_actions, num_actions))
    for seq in action_seq_list:
        for i in range(len(seq) - 1):
            action_transition_matrix[seq[i], seq[i + 1]] += 1

    # Normalize the transition matrix
    action_transition_matrix = np.nan_to_num(action_transition_matrix / action_transition_matrix.sum(axis=1, keepdims=True))

    # Visualize and save the action transition matrix
    plt.figure()
    plt.imshow(action_transition_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title('Action Transition Matrix')
    plt.xlabel('Next Action')
    plt.ylabel('Current Action')
    plt.savefig(os.path.join(folder_path, "action_transition_matrix.png"))
    plt.show()
    plt.close()

    # Save stats to a file
    stats_file_path = os.path.join(folder_path, "stats.txt")
    with open(stats_file_path, 'w') as file:
        for key, value in stats.items():
            file.write(f"{key}: {value}\n")

    return stats


if __name__ == '__main__':
    base_maze_dir = "./trained_models/maze"
    a2c_dir_name_v1 = "trained_a2c_models"
    cpgpo_dir_names_v1 = ["trained_n1_cpgpo_models_v1", "trained_n2_cpgpo_models_v1", "trained_n3_cpgpo_models_v1", "trained_n4_cpgpo_models_v1"]

    # Paths for the model directories
    a2c_v1_dir_paths = [os.path.join(base_maze_dir, a2c_dir_name_v1)]
    cpgpo_v1_dir_paths = [os.path.join(base_maze_dir, dir_name) for dir_name in cpgpo_dir_names_v1]

    # Evaluate A2C models
    visitation_matrices_a2c, action_sequences_a2c, steps_a2c, rewards_a2c = evaluate_maze_agents(ActorCriticMaze, 10, a2c_v1_dir_paths)
    result_folder = './evaluation_results'
    stats_a2c = evaluate_maze_stats(visitation_matrices_a2c, action_sequences_a2c, steps_a2c, rewards_a2c, os.path.join(result_folder, "a2c"))
    for key, value in stats_a2c.items():
        print(f"A2C {key}: {value}")

    # Evaluate CPGPO models
    visitation_matrices_cpgpo, action_sequences_cpgpo, steps_cpgpo, rewards_cpgpo = evaluate_maze_agents(ActorCriticMaze, 10, cpgpo_v1_dir_paths)
    stats_cpgpo = evaluate_maze_stats(visitation_matrices_cpgpo, action_sequences_cpgpo, steps_cpgpo, rewards_cpgpo, os.path.join(result_folder, "cpgpo"))
    for key, value in stats_cpgpo.items():
        print(f"CPGPO {key}: {value}")