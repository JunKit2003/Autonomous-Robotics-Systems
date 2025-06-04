# DQN_CartPole.py

import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
import random

from cart_pole_model import NoisyDuelingNet  # Import the Dueling DQN network

# Define a transition tuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

# Replay Memory class
class ReplayMemory(object):

    def __init__(self, capacity, seed=42):
        self.capacity = capacity
        self.memory = []
        self.position = 0  # Circular buffer position
        self.seed = seed
        random.seed(seed)

    def push(self, *args):
        """Save a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # Expand memory
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity  # Circular buffer

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN agent class
class DQN():
    def __init__(self, hyper_parameter_config, env_config, use_cuda=False, seed=42):
        super(DQN, self).__init__()

        # Set hyperparameters
        self.batch_size = hyper_parameter_config["BATCH_SIZE"]
        self.learning_rate = hyper_parameter_config["LEARNING_RATE"]
        self.gamma = hyper_parameter_config["GAMMA"]
        self.memory_capacity = hyper_parameter_config["MEMORY_CAPACITY"]
        self.target_update_frequency = hyper_parameter_config["TARGET_UPDATE_FREQUENCY"]

        # NROWAN-DQN specific parameters
        self.initial_sigma = hyper_parameter_config["INITIAL_SIGMA"]
        self.k_final = hyper_parameter_config["K_FINAL"]
        self.a = hyper_parameter_config["A_CONSTANT"]
        self.min_frames_before_learning = hyper_parameter_config["MIN_FRAMES_BEFORE_LEARNING"]
        self.update_count = 0

        # Environment configuration
        self.num_states = env_config["NUM_STATES"]
        self.num_actions = env_config["NUM_ACTIONS"]
        self.env_a_shape = env_config["ENV_A_SHAPE"]

        # Networks
        self.eval_net = NoisyDuelingNet(env_config, initial_sigma=self.initial_sigma, use_cuda=use_cuda)
        self.target_net = NoisyDuelingNet(env_config, initial_sigma=self.initial_sigma, use_cuda=use_cuda)
        self.target_net.load_state_dict(self.eval_net.state_dict())

        if use_cuda:
            self.eval_net.cuda()
            self.target_net.cuda()

        self.memory = ReplayMemory(self.memory_capacity, seed=seed)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()

        # Initialize k and N_f
        self.k = 0.0
        self.N_f = 0

        self.use_cuda = use_cuda

        # Set seed
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        if self.use_cuda:
            state = state.cuda()
        with torch.no_grad():
            self.eval_net.reset_noise()
            action_value = self.eval_net(state)
            action = torch.argmax(action_value, dim=1).item()
        return action

    def store_transition(self, state, action, reward, next_state):
        self.memory.push(state, action, reward, next_state)

    def learn(self):
        if len(self.memory) < self.min_frames_before_learning:
            return

        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        batch_state = torch.FloatTensor(np.array(batch.state))
        batch_action = torch.LongTensor(np.array(batch.action)).unsqueeze(1)
        batch_reward = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1)
        batch_next_state = torch.FloatTensor(np.array(batch.next_state))

        if self.use_cuda:
            batch_state = batch_state.cuda()
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()
            batch_next_state = batch_next_state.cuda()

        self.eval_net.reset_noise()
        q_eval = self.eval_net(batch_state).gather(1, batch_action)

        with torch.no_grad():
            self.target_net.reset_noise()
            next_actions = self.eval_net(batch_next_state).argmax(dim=1).unsqueeze(1)
            q_next = self.target_net(batch_next_state).gather(1, next_actions)

        q_target = batch_reward + self.gamma * q_next

        td_loss = self.loss_func(q_eval, q_target)

        D = self.eval_net.get_noise_norm()

        total_loss = td_loss + self.k * D

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Adjust k
        self.N_f += 1
        self.k = self.k_final - self.k_final * np.exp(-self.N_f / self.a)

        # Adjust sigma in the output layer
        sigma = self.initial_sigma * np.exp(-self.N_f / self.a)
        self.eval_net.adjust_output_noise_sigma(sigma)

        # Update target network
        if self.N_f % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

