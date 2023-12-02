# n_step_cp.py:
import torch
import torch.optim as optim
import os
import time
from utils import ActorCritic, log, get_discounted_rewards, report_old_model_distribution, save_model, \
    bprop_with_cumulative_prob, should_update, load_reference_models, nstep_cumulative_prob_from_logs, \
    nstep_cumulative_prob_from_states
from grid_world import SimpleGridEnv

# Hyperparameters (use default values as per your previous setup)
learning_rate = 0.002


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = f"{log_dir}/training_log_{timestamp}.txt"
    return open(log_filename, 'w')


def train(env, model, reference_model, optimizer, log_dir, model_dir, epoch=1000, gamma=0.99, n_step=4, epsilon=0.2):
    log_file = setup_logging(log_dir)
    try:
        report_old_model_distribution(reference_model, env, log_file)
        log("-" * 60, log_file)

        for episode in range(epoch):
            state = env.reset()
            done = False
            model.reset_trajectory()

            while not done:
                state_tensor = torch.FloatTensor(state).reshape(1, 3).float()
                action_prob, value = model(state_tensor)
                action_dist = torch.distributions.Categorical(action_prob)
                action = action_dist.sample()
                next_state, reward, done, _ = env.step(action.item())
                log_prob = action_dist.log_prob(action)

                model.log_probs.append(log_prob)
                model.values.append(value)
                model.rewards.append(reward)
                model.state_trajectory.append(state_tensor)
                model.action_trajectory.append(action)

                state = next_state

            returns = get_discounted_rewards(model.rewards, gamma)
            log_nstep_cp_new = nstep_cumulative_prob_from_logs(n_step, model.log_probs)
            log_nstep_cp_old = nstep_cumulative_prob_from_states(reference_model, n_step, model.state_trajectory,
                                                                 model.action_trajectory)

            # Log before update
            log(f"\nEpisode {episode} - Before Update:", log_file)
            log(f"{'State':>30} | {'Action':>6} | {'New CP':>10} | {'Old CP':>10} | {'Update':>6}", log_file)
            log("-" * 70, log_file)
            for state_tensor, action, log_cp_new, log_cp_old, ret, value in zip(model.state_trajectory,
                                                                                model.action_trajectory,
                                                                                log_nstep_cp_new, log_nstep_cp_old,
                                                                                returns, model.values):
                state_repr = ' '.join(f"{x:.2f}" for x in state_tensor.detach().numpy().flatten())
                cp_new = torch.exp(log_cp_new).item()
                cp_old = torch.exp(log_cp_old).item()
                update_decision = "Yes" if should_update(log_cp_new, log_cp_old, epsilon, ret - value) else "No"
                log(f"{state_repr:>30} | {action.item():>6} | {cp_new:>10.4f} | {cp_old:>10.4f} | {update_decision:>6}",
                    log_file)

            actor_loss, critic_loss = bprop_with_cumulative_prob(
                model.log_probs,
                model.values,
                returns,
                log_nstep_cp_new,
                log_nstep_cp_old,
                epsilon,
                optimizer
            )

            # Log after update
            log(f"\nEpisode {episode} - After Update:", log_file)
            log(f"{'State':>30} | {'Action Probs':>20}", log_file)
            log("-" * 70, log_file)
            for state in env.all_states():
                state_tensor = torch.FloatTensor(state).reshape(1, 3).float()
                action_prob, _ = model(state_tensor)
                action_prob = action_prob.detach().numpy().squeeze()
                state_repr = ' '.join(f"{x:.2f}" for x in state)
                log(f"{state_repr:>30} | [{action_prob[0]:.4f}, {action_prob[1]:.4f}]", log_file)

            last_reward = model.rewards[-1].item()
            log(f"Reward: {last_reward}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss}", log_file)
            log("-" * 70, log_file)

        save_model(model, model_dir, f"trained_model_{time.strftime('%Y%m%d-%H%M%S')}.pth")

    finally:
        log_file.close()
    return model


def main():
    # Initialize environment and models
    env = SimpleGridEnv()
    model = ActorCritic()
    reference_model = load_reference_models(["./trained_models/a2c_model1.pth"], 1)[0].eval()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Directories for logging and model saving
    log_dir = "./logs"
    model_dir = "./trained_models"

    # Train the model
    trained_model = train(env, model, reference_model, optimizer, 1000, 0.99, 4, 0.2, log_dir, model_dir)
    print("Training complete!")


if __name__ == "__main__":
    main()
