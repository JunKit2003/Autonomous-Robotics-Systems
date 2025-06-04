# lunar_lander_main.py
import gymnasium as gym  # We use gymnasium instead of gym
import numpy as np
import sys
import os
import torch
import random

# Add the parent directory to the Python path to allow importing custom modules from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from HRL_DQN_LunarLander import HRL_DQN  # Import the HRL_DQN class
from DQN_LunarLander import DQN           # Import the DQN class
from components.utils import plot_durations  # Utility function for plotting episode rewards over time
from NoiseInjector import NoiseInjector   # Class to inject environment noise

# Define hyperparameter configurations for Manager and Modules in HRL
hyper_parameter_config = {
    'MANAGER': {
        "BATCH_SIZE": 128,
        "LEARNING_RATE": 1e-4,
        "GAMMA": 0.99,
        "MEMORY_CAPACITY": 50000,
        "Q_NETWORK_ITERATION": 500,
        "INITIAL_SIGMA": 0.2,
        "SIGMA_DECAY": 0.995,
        "K_FINAL": 1.85,
        "A_CONSTANT": 5000,
        "MIN_FRAMES_BEFORE_LEARNING": 5000
    },
    'MODULE1': {
        "BATCH_SIZE": 128,
        "LEARNING_RATE": 1e-4,
        "GAMMA": 0.99,
        "MEMORY_CAPACITY": 50000,
        "Q_NETWORK_ITERATION": 500,
        "INITIAL_SIGMA": 0.2,
        "SIGMA_DECAY": 0.995,
        "K_FINAL": 2,
        "A_CONSTANT": 5000,
        "MIN_FRAMES_BEFORE_LEARNING": 5000
    },
    'MODULE2': {
        "BATCH_SIZE": 128,
        "LEARNING_RATE": 1e-4,
        "GAMMA": 0.99,
        "MEMORY_CAPACITY": 50000,
        "Q_NETWORK_ITERATION": 500,
        "INITIAL_SIGMA": 0.2,
        "SIGMA_DECAY": 0.995,
        "K_FINAL": 1.75,
        "A_CONSTANT": 5000,
        "MIN_FRAMES_BEFORE_LEARNING": 5000
    }
}

def update_episode_metrics(i, ep_reward, running_reward, steps_in_episode, episode_rewards, running_average_rewards, plot_path):
    # Update running_reward with exponential smoothing
    running_reward = running_reward * 0.99 + ep_reward * 0.01
    episode_rewards.append(ep_reward)
    running_average_rewards.append(running_reward)

    # Plot the episode durations (rewards) and running average
    plot_durations('LunarLander-v3 Reward', episode_rewards, running_average_rewards, plot_path)

    # Print episode stats
    print(f"Episode {i+1}: Steps = {steps_in_episode}, Total Reward = {ep_reward}, Running Average Reward = {running_reward:.2f}")

    return running_reward

def main():
    # Ask user for a seed for reproducibility
    seed = int(input("Enter a seed value (e.g., 42): "))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ask user for mode: Deterministic or Stochastic
    mode = input("Choose mode: Deterministic (D) or Stochastic (S): ").strip().upper()

    env_noise_params = {}
    if mode == 'D':
        print("Running in Deterministic mode")
        mode_str = 'Deterministic'
    elif mode == 'S':
        print("Running in Stochastic mode")
        # Collect noise parameters from user
        gravity_noise = float(input("Enter gravity noise (e.g., 0.1 for 10% increase): "))
        gravity = -10 * (1 + gravity_noise)  # Gravity is negative for LunarLander
        wind_power = float(input("Enter wind power (e.g., 15.0): "))
        turbulence_power = float(input("Enter turbulence power (e.g., 1.5): "))
        env_noise_params = {'gravity': gravity, 'wind_power': wind_power, 'turbulence_power': turbulence_power}
        mode_str = 'Stochastic'
    else:
        print("Invalid mode selected. Defaulting to Deterministic mode.")
        mode_str = 'Deterministic'

    # Ask user whether to use HRL agent or not
    use_hrl_input = input("Do you want to use the HRL agent? (Y/N): ").strip().upper()
    if use_hrl_input == 'Y':
        use_hrl = True
        agent_str = 'HRL_NoisyDueling_DQN_LunarLander-v3'
    else:
        use_hrl = False
        agent_str = 'NoisyDueling_DQN_LunarLander-v3'

    # Create output directory for plots
    base_plot_path = 'Output/LunarLander-v3/'
    plot_path = os.path.join(base_plot_path, f"{agent_str}_{mode_str}/")

    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Initialize the LunarLander environment
    env = gym.make("LunarLander-v3")
    assert env.spec is not None, "Environment 'LunarLander-v3' not found."

    # Set seed for environment
    env.reset(seed=seed)

    # If there are noise parameters, wrap the environment with NoiseInjector to apply noise
    if env_noise_params:
        env = NoiseInjector(env, **env_noise_params)

    # Extract environment parameters
    NUM_ACTIONS = env.action_space.n
    NUM_STATES = env.observation_space.shape[0]

    # Determine action shape for compatibility
    if isinstance(env.action_space, gym.spaces.Discrete):
        ENV_A_SHAPE = 0
    else:
        ENV_A_SHAPE = env.action_space.shape

    # Define environment configuration
    env_config = {
        "NUM_ACTIONS": NUM_ACTIONS,
        "NUM_STATES": NUM_STATES,
        "ENV_A_SHAPE": ENV_A_SHAPE
    }

    # Initialize the chosen agent (HRL or single DQN)
    if use_hrl:
        agent = HRL_DQN(hyper_parameter_config, env_config)
        plot_title = 'LunarLander-v3 with HRL_NoisyDueling_DQN'
    else:
        agent = DQN(hyper_parameter_config['MODULE1'], env_config)
        plot_title = 'LunarLander-v3 with NoisyDueling_DQN'

    episodes = 1000
    print("Collecting Experience....")

    running_reward = -200
    episode_rewards = []
    running_average_rewards = []

    baseline = 200
    n_episodes_solved = 100

    # Main training loop
    for i in range(episodes):
        state, info = env.reset()
        ep_reward = 0
        steps_in_episode = 0

        while True:
            # Choose action (and possibly which module to use if HRL is active)
            if use_hrl:
                action, module_choice = agent.choose_action(state)
            else:
                action = agent.choose_action(state)

            # Perform the action in the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store the transition
            if use_hrl:
                agent.store_transition(state, action, reward, next_state, module_choice)
            else:
                agent.store_transition(state, action, reward, next_state)

            ep_reward += reward
            steps_in_episode += 1

            # Agent learns from experience
            agent.learn()

            # Check if the episode ended
            if done:
                # Update metrics and print results
                running_reward = update_episode_metrics(
                    i, ep_reward, running_reward, steps_in_episode,
                    episode_rewards, running_average_rewards, plot_path
                )

                # Check if the environment is solved
                if len(episode_rewards) >= n_episodes_solved:
                    avg_reward = np.mean(episode_rewards[-n_episodes_solved:])
                    if avg_reward >= baseline:
                        print(f"Environment solved after {i+1} episodes with average reward {avg_reward:.2f} over the last {n_episodes_solved} episodes.")
                        break
                break

            # Move to next state
            state = next_state

    print("Training completed.")

if __name__ == '__main__':
    main()
