# DQN_LunarLander.py
import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
import random

from lunar_lander_model import NoisyDuelingNet  # Import the neural network model (Noisy Dueling DQN)

# Define a named tuple 'Transition' to store a single transition in the replay memory
# Each transition includes (state, action, reward, next_state)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

# Replay Memory class for storing past experiences
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity              # Maximum number of transitions the memory can hold
        self.memory = []                      # List to store transitions
        self.position = 0                     # Current position to insert the next transition (circular buffer)

    def push(self, *args):
        """Save a transition into the replay memory."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)          # Expand memory if it's not full yet
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity  # Move position forward and wrap around (circular)

    def sample(self, batch_size):
        # Randomly sample a batch of transitions from the memory
        return random.sample(self.memory, batch_size)

    def __len__(self):
        # Return the current size of the memory (number of stored transitions)
        return len(self.memory)


# DQN class for training a Dueling Noisy DQN on the Lunar Lander environment
class DQN():
    def __init__(self, hyper_parameter_config, env_config):
        super(DQN, self).__init__()

        # Extract hyperparameters from the configuration dictionary
        self.batch_size = hyper_parameter_config["BATCH_SIZE"]                # Batch size for training updates
        self.learning_rate = hyper_parameter_config["LEARNING_RATE"]          # Learning rate for the optimizer
        self.gamma = hyper_parameter_config["GAMMA"]                          # Discount factor for future rewards
        self.memory_capacity = hyper_parameter_config["MEMORY_CAPACITY"]      # Replay buffer capacity
        self.q_network_iteration = hyper_parameter_config["Q_NETWORK_ITERATION"]  # Frequency of target network update
        self.initial_sigma = hyper_parameter_config["INITIAL_SIGMA"]          # Initial sigma for noisy layers
        self.sigma_decay = hyper_parameter_config["SIGMA_DECAY"]              # Decay factor for noise sigma
        self.k_final = hyper_parameter_config["K_FINAL"]                      # Final scaling factor for noise
        self.a_constant = hyper_parameter_config["A_CONSTANT"]                # Constant for noise schedule
        self.min_frames_before_learning = hyper_parameter_config["MIN_FRAMES_BEFORE_LEARNING"]  # Minimum frames before training starts

        # Extract environment configuration
        self.num_states = env_config["NUM_STATES"]   # Number of state features
        self.num_actions = env_config["NUM_ACTIONS"] # Number of possible actions
        self.env_a_shape = env_config["ENV_A_SHAPE"] # Shape of actions (0 if discrete)

        # Create the evaluation (online) and target networks using the Noisy Dueling architecture
        self.eval_net = NoisyDuelingNet(env_config, initial_sigma=self.initial_sigma)
        self.target_net = NoisyDuelingNet(env_config, initial_sigma=self.initial_sigma)

        # Initialize the target network with the same weights as the evaluation network
        self.target_net.load_state_dict(self.eval_net.state_dict())

        # Initialize counters and replay memory
        self.learn_step_counter = 0
        self.memory = ReplayMemory(self.memory_capacity)

        # Set up the optimizer (Adam) and loss function (Huber loss for stability)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.SmoothL1Loss()  # Huber loss is less sensitive to outliers than MSE

        # Initialize sigma scale factor for adjusting noise over time
        self.sigma_scale = 1.0

    def choose_action(self, state):
        # Convert state to a PyTorch tensor and add batch dimension
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        state = torch.FloatTensor(state).unsqueeze(0)

        # No gradient needed, just forward pass to choose best action
        with torch.no_grad():
            # Reset noise before selecting action so that we always apply fresh noise
            self.eval_net.reset_noise()

            # Evaluate Q-values for each action
            action_value = self.eval_net(state)

            # Choose the action with the highest Q-value
            action = torch.argmax(action_value, dim=1).item()
        return action

    def store_transition(self, state, action, reward, next_state):
        # Ensure states are numpy arrays
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state)

        # Store the transition in replay memory
        self.memory.push(state, action, reward, next_state)

    def learn(self):
        # If we haven't collected enough samples yet, skip learning
        if len(self.memory) < self.min_frames_before_learning:
            return

        # Every q_network_iteration steps, copy the eval_net weights into the target_net
        if self.learn_step_counter % self.q_network_iteration == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter += 1

        # Adjust sigma dynamically, decreasing it over training steps for less noise as training progresses
        self.sigma_scale = max(
            self.initial_sigma * (1 - (self.learn_step_counter / (self.a_constant + self.learn_step_counter))),
            self.initial_sigma / self.k_final
        )
        self.eval_net.adjust_sigma(self.sigma_scale)
        self.target_net.adjust_sigma(self.sigma_scale)

        # Sample a batch of transitions from replay memory
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert batch data into tensors
        batch_state = torch.FloatTensor(np.array(batch.state))
        batch_action = torch.LongTensor(np.array(batch.action)).unsqueeze(1)
        batch_reward = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1)
        batch_next_state = torch.FloatTensor(np.array(batch.next_state))

        # Compute the predicted Q-values for the chosen actions Q(s,a)
        self.eval_net.reset_noise()
        q_eval = self.eval_net(batch_state).gather(1, batch_action)

        # Compute target Q-values with the target network Q'(s',a')
        with torch.no_grad():
            self.target_net.reset_noise()
            q_next = self.target_net(batch_next_state).max(1)[0].unsqueeze(1)

        # Compute the TD target: r + gamma * max_a Q'(s', a)
        q_target = batch_reward + self.gamma * q_next

        # Compute the Huber loss between predicted Q-values and target Q-values
        loss = self.loss_func(q_eval, q_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), 1)

        self.optimizer.step()

        # Reset noise after learning step to ensure fresh noise for next actions
        self.eval_net.reset_noise()
        self.target_net.reset_noise()
