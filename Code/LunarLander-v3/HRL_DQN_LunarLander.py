# HRL_DQN_LunarLander.py
import torch
import torch.nn as nn
import numpy as np
from DQN_LunarLander import DQN  # Import the basic DQN class

# Hierarchical RL (HRL) DQN class that uses a Manager and Modules (Module1 and Module2)
# The manager decides which module to use at each timestep.
class HRL_DQN():
    def __init__(self, hyper_parameter_config, env_config):
        super(HRL_DQN, self).__init__()

        # Create two DQN agents representing two different modules/skills
        self.module1 = DQN(hyper_parameter_config['MODULE1'], env_config)
        self.module2 = DQN(hyper_parameter_config['MODULE2'], env_config)

        # Create a high-level manager that chooses which module to use
        manager_env_config = {
            "NUM_ACTIONS": 2,                    # Manager chooses between 2 modules
            "NUM_STATES": env_config["NUM_STATES"],
            "ENV_A_SHAPE": env_config["ENV_A_SHAPE"]
        }

        self.manager = DQN(hyper_parameter_config['MANAGER'], manager_env_config)

    def choose_action(self, state):
        # Manager picks which module to use based on current state
        module_choice = self.manager.choose_action(state)

        # Depending on the chosen module, the corresponding module picks the action
        if module_choice == 0:
            # Use Module 1
            action = self.module1.choose_action(state)
        else:
            # Use Module 2
            action = self.module2.choose_action(state)

        return action, module_choice

    def store_transition(self, state, action, reward, next_state, module_choice):
        # Store transitions in both the manager and the chosen module's replay buffer
        self.manager.store_transition(state, module_choice, reward, next_state)

        if module_choice == 0:
            self.module1.store_transition(state, action, reward, next_state)
        else:
            self.module2.store_transition(state, action, reward, next_state)

    def learn(self):
        # Each agent (manager, module1, module2) learns from its own replay buffer
        self.manager.learn()
        self.module1.learn()
        self.module2.learn()
