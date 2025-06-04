# lunar_lander_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from NoisyLinear import NoisyLinear

# NoisyDuelingNet defines a Dueling DQN architecture with NoisyLinear layers.
# Dueling: Splits the network into separate value and advantage streams.
class NoisyDuelingNet(nn.Module):
    def __init__(self, env_config, initial_sigma=0.5):
        super(NoisyDuelingNet, self).__init__()
        self.num_states = env_config["NUM_STATES"]
        self.num_actions = env_config["NUM_ACTIONS"]

        # Common fully-connected layers before splitting into value and advantage
        self.fc1 = nn.Linear(self.num_states, 128)
        self.fc2 = nn.Linear(128, 128)

        # Value stream: predicts the value of being in a given state
        self.value_fc = nn.Sequential(
            NoisyLinear(128, 64, sigma_init=initial_sigma),
            nn.ReLU(),
            NoisyLinear(64, 1, sigma_init=initial_sigma)
        )

        # Advantage stream: predicts how much better taking a certain action is compared to average
        self.advantage_fc = nn.Sequential(
            NoisyLinear(128, 64, sigma_init=initial_sigma),
            nn.ReLU(),
            NoisyLinear(64, self.num_actions, sigma_init=initial_sigma)
        )

    def forward(self, x):
        # Compute common features
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Compute value and advantage
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)

        # Combine value and advantage into Q-values
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values

    def reset_noise(self):
        # Reset noise parameters in all noisy layers
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def adjust_sigma(self, sigma_scale):
        # Adjust sigma (noise scale) in all noisy layers
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.adjust_sigma(sigma_scale)
