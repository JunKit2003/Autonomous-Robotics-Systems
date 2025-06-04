# Import necessary libraries
import gymnasium as gym
import numpy as np
import sys
import os
import torch
import random

# Add the parent directory to sys.path so we can import custom modules from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom classes and functions from local files
from HRL_DQN_CartPole import HRL_DQN
from DQN_CartPole import DQN
from components.utils import plot_durations  # Utility function for plotting training progress
from NoiseInjector import NoiseInjector

# Hyperparameter configuration dictionaries for Hierarchical Reinforcement Learning (HRL)
# and the modules it controls. Each dictionary holds parameters for batch size, learning rate,
# gamma (discount factor), replay memory capacity, target network update frequency, initial noise sigma,
# and other parameters related to training the DQN networks.
hyper_parameter_config = {
    'MANAGER': {
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 0.001,
        "GAMMA": 0.99,
        "MEMORY_CAPACITY": 5000,
        "TARGET_UPDATE_FREQUENCY": 1000,
        "INITIAL_SIGMA": 0.4,
        "K_FINAL": 4.0,
        "A_CONSTANT": 5000,
        "MIN_FRAMES_BEFORE_LEARNING": 32
    },
    'MODULE1': {
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 0.001,
        "GAMMA": 0.99,
        "MEMORY_CAPACITY": 5000,
        "TARGET_UPDATE_FREQUENCY": 1000,
        "INITIAL_SIGMA": 0.4,
        "K_FINAL": 4.0,
        "A_CONSTANT": 5000,
        "MIN_FRAMES_BEFORE_LEARNING": 32
    },
    'MODULE2': {
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 0.001,
        "GAMMA": 0.99,
        "MEMORY_CAPACITY": 5000,
        "TARGET_UPDATE_FREQUENCY": 1000,
        "INITIAL_SIGMA": 0.4,
        "K_FINAL": 4.0,
        "A_CONSTANT": 5000,
        "MIN_FRAMES_BEFORE_LEARNING": 32
    }
}

if __name__ == '__main__':
    # Prompt user for a random seed to ensure reproducibility of results
    seed = int(input("Enter random seed (e.g., 42): "))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # If GPU is available, set the seed for CUDA as well to ensure reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Check if CUDA (GPU) is available
    use_cuda = torch.cuda.is_available()

    # Prompt the user for the mode: Deterministic or Stochastic environment
    mode = input("Choose mode: Deterministic (D) or Stochastic (S): ").strip().upper()

    # Configure environment noise based on chosen mode
    if mode == 'D':
        # Deterministic: no environment noise changes
        env_noise_params = {}
        mode_str = 'Deterministic'
    elif mode == 'S':
        # Stochastic: user specifies gravity noise to alter environment dynamics
        gravity_noise = float(input("Enter gravity noise (e.g., 0.1 for 10% increase): "))
        gravity = 9.8 * (1 + gravity_noise)
        env_noise_params = {'gravity': gravity}
        mode_str = 'Stochastic'
    else:
        # If user inputs invalid mode, default to deterministic
        print("Invalid mode selected. Defaulting to Deterministic mode.")
        env_noise_params = {}
        mode_str = 'Deterministic'

    # Ask the user if they want to use the Hierarchical RL (HRL) agent or just the basic DQN
    use_hrl_input = input("Do you want to use the HRL agent? (Y/N): ").strip().upper()
    if use_hrl_input == 'Y':
        use_hrl = True
        agent_str = 'HRL_NoisyDueling_DQN_CartPole-v1'
    else:
        use_hrl = False
        agent_str = 'NoisyDueling_DQN_CartPole-v1'

    # Set up paths for saving output plots
    base_plot_path = 'Output/CartPole-v1/'
    plot_path = os.path.join(base_plot_path, f"{agent_str}_{mode_str}/")

    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Create and configure the Gym environment
    env = gym.make("CartPole-v1")
    env = env.unwrapped  # Remove the time limit wrapper for more control if needed

    # Inject noise into the environment if needed (based on mode)
    noise_injector = NoiseInjector(env_noise_params=env_noise_params)
    env = noise_injector.inject_env_noise(env)

    # Extract environment information
    NUM_ACTIONS = env.action_space.n
    NUM_STATES = env.observation_space.shape[0]

    # ENV_A_SHAPE is 0 for discrete action spaces, otherwise it would store shape for continuous spaces
    ENV_A_SHAPE = 0 if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape

    env_config = {
        "NUM_ACTIONS": NUM_ACTIONS,
        "NUM_STATES": NUM_STATES,
        "ENV_A_SHAPE": ENV_A_SHAPE
    }

    # Define a custom reward function for CartPole
    # This function rewards the agent for keeping the pole upright (minimizing pole angle) and the cart near the center
    def reward_func(env, x, x_dot, theta, theta_dot):
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.5
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        return reward

    # Function to update metrics at the end of each episode
    # This plots the durations (steps taken) and running average reward over time
    def update_episode_metrics(i, ep_reward, running_reward, steps_in_episode, episode_rewards, running_average_rewards, plot_path, plot_title):
        running_reward = running_reward * 0.99 + ep_reward * 0.01
        episode_rewards.append(ep_reward)
        running_average_rewards.append(running_reward)

        # Plot the reward curve for visualization and debugging
        plot_durations(plot_title, episode_rewards, running_average_rewards, path=plot_path)

        print(f"Episode {i+1}: Steps = {steps_in_episode}, Total Reward = {ep_reward}, Running Average Reward = {running_reward:.2f}")

        return running_reward

    # Main training loop
    def main():
        # Initialize the chosen agent (HRL or standard DQN)
        if use_hrl:
            agent = HRL_DQN(hyper_parameter_config, env_config, use_cuda=use_cuda, seed=seed)
            plot_title = 'CartPole-v1 with HRL_NoisyDueling_DQN'
        else:
            agent = DQN(hyper_parameter_config['MODULE1'], env_config, use_cuda=use_cuda, seed=seed)
            plot_title = 'CartPole-v1 with NoisyDueling_DQN'

        episodes = 1000
        print("Collecting Experience....")

        # Initialize running reward and metrics tracking
        running_reward = 10
        episode_rewards = []
        running_average_rewards = []

        # Criteria for considering the environment "solved"
        baseline = 475
        n_episodes_solved = 100

        # Training loop for the specified number of episodes
        for i in range(episodes):
            state, _ = env.reset()
            ep_reward = 0
            steps_in_episode = 0

            while True:
                # Choose an action depending on whether we use HRL or not
                if use_hrl:
                    action, module_choice = agent.choose_action(state)
                else:
                    action = agent.choose_action(state)

                # Perform the action in the environment
                next_state, _, truncated, terminated, info = env.step(action)
                done = truncated or terminated

                # Extract state variables for reward calculation
                x, x_dot, theta, theta_dot = next_state
                reward = reward_func(env, x, x_dot, theta, theta_dot)

                # Store the transition in replay memory
                if use_hrl:
                    agent.store_transition(state, action, reward, next_state, module_choice)
                else:
                    agent.store_transition(state, action, reward, next_state)

                ep_reward += reward
                steps_in_episode += 1

                # Trigger the agent to learn from its memory
                agent.learn()

                # Safety check to avoid infinite loops
                if steps_in_episode >= 1000:
                    done = True

                if done:
                    # Update metrics and plot progress
                    running_reward = update_episode_metrics(
                        i, ep_reward, running_reward, steps_in_episode,
                        episode_rewards, running_average_rewards, plot_path, plot_title
                    )

                    # Check if the average reward over the last n_episodes_solved episodes meets the baseline
                    if len(episode_rewards) >= n_episodes_solved:
                        avg_reward = np.mean(episode_rewards[-n_episodes_solved:])
                        if avg_reward >= baseline:
                            print(f"Environment solved after {i+1} episodes with average reward {avg_reward:.2f} over the last {n_episodes_solved} episodes.")
                            break
                    break

                # Move to the next state
                state = next_state

        print("Training completed.")

    main()
