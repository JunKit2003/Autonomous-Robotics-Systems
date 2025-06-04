# HRL_DQN_CartPole.py

import torch
import torch.nn as nn
import numpy as np
from DQN_CartPole import DQN  # Import the basic DQN class

# HRL_DQN integrates a hierarchical structure:
# A "manager" policy chooses which module to use, and each module is a separate DQN.
# This allows learning different skills or tasks in separate modules and switching between them.
class HRL_DQN():
    def __init__(self, hyper_parameter_config, env_config, use_cuda=False, seed=42):
        super(HRL_DQN, self).__init__()

        # Module 1: a DQN specialized for one aspect of the task
        self.module1 = DQN(hyper_parameter_config['MODULE1'], env_config, use_cuda=use_cuda, seed=seed)
        # Module 2: a DQN specialized for handling noise or another aspect of the task
        self.module2 = DQN(hyper_parameter_config['MODULE2'], env_config, use_cuda=use_cuda, seed=seed)

        # Manager: a DQN that decides which module to use at each step.
        # It has an action space of size 2 (either module 1 or module 2).
        manager_env_config = {
            "NUM_ACTIONS": 2,
            "NUM_STATES": env_config["NUM_STATES"],
            "ENV_A_SHAPE": env_config["ENV_A_SHAPE"]
        }

        self.manager = DQN(hyper_parameter_config['MANAGER'], manager_env_config, use_cuda=use_cuda, seed=seed)

    def choose_action(self, state):
        # Manager picks which module to use
        module_choice = self.manager.choose_action(state)

        # Depending on the manager's choice, select action from the chosen module
        if module_choice == 0:
            action = self.module1.choose_action(state)
        else:
            action = self.module2.choose_action(state)

        return action, module_choice

    def store_transition(self, state, action, reward, next_state, module_choice):
        # Store transitions in manager and the chosen module's memory
        self.manager.store_transition(state, module_choice, reward, next_state)
        if module_choice == 0:
            self.module1.store_transition(state, action, reward, next_state)
        else:
            self.module2.store_transition(state, action, reward, next_state)

    def learn(self):
        # Each component learns separately from its own replay buffer
        self.manager.learn()
        self.module1.learn()
        self.module2.learn()
