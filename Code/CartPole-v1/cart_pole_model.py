import torch.nn as nn
import torch.nn.functional as F

from NoisyLinear import NoisyLinear

# NoisyDuelingNet defines a neural network architecture for a Dueling DQN with Noisy Layers.
# The network splits into two streams after a shared feature layer:
# 1) A value stream, estimating the state value.
# 2) An advantage stream, estimating the relative advantage of each action.
class NoisyDuelingNet(nn.Module):
    def __init__(self, env_config, initial_sigma=0.4, use_cuda=False):
        super(NoisyDuelingNet, self).__init__()

        self.num_states = env_config["NUM_STATES"]
        self.num_actions = env_config["NUM_ACTIONS"]

        self.use_cuda = use_cuda

        # Common feature extraction layer: from input state to a hidden representation
        self.feature = nn.Sequential(
            NoisyLinear(self.num_states, 128, sigma_init=initial_sigma, use_cuda=use_cuda),
            nn.ReLU()
        )

        # Value stream: outputs a single scalar value representing the value of the current state
        self.value_stream = nn.Sequential(
            NoisyLinear(128, 64, sigma_init=initial_sigma, use_cuda=use_cuda),
            nn.ReLU(),
            NoisyLinear(64, 1, sigma_init=initial_sigma, use_cuda=use_cuda)
        )

        # Advantage stream: outputs a vector representing the advantage of each action
        self.advantage_stream = nn.Sequential(
            NoisyLinear(128, 64, sigma_init=initial_sigma, use_cuda=use_cuda),
            nn.ReLU(),
            NoisyLinear(64, self.num_actions, sigma_init=initial_sigma, use_cuda=use_cuda)
        )

    def forward(self, x):
        # Compute the common feature representation
        x = self.feature(x)
        # Compute value and advantage
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # Combine value and advantage into Q-values
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def reset_noise(self):
        # Reset the noise parameters in all NoisyLinear layers
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def adjust_output_noise_sigma(self, sigma):
        # Adjust sigma (noise level) in the output layer's NoisyLinear layers
        self.value_stream[-1].set_sigma(sigma)
        self.advantage_stream[-1].set_sigma(sigma)

    def get_noise_norm(self):
        # Compute the norm of the noise parameters for regularization
        D = 0.0
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                D += module.get_noise_norm()
        return D
