# NoisyLinear.py
import torch
import torch.nn as nn
import math

# NoisyLinear defines a linear layer with factorized Gaussian noise added to weights and biases.
# This is used to implement NoisyNets for exploration in DQN.
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5, bias=True):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters: mu and sigma for weights
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))

        # Non-learnable noise buffers
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
            self.register_buffer('bias_epsilon', torch.empty(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)
            self.register_buffer('bias_epsilon', None)

        # Initialize parameters and reset noise
        self.reset_parameters(sigma_init)
        self.reset_noise()

    def reset_parameters(self, sigma_init):
        # Initialize parameters using a uniform distribution for mu
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_init / math.sqrt(self.in_features))

        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        # Sample fresh noise for each forward pass during training
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))

        if self.bias_epsilon is not None:
            self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        # During training, apply noisy weights and biases
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon if self.bias_mu is not None else None
        else:
            # During evaluation, use the deterministic mean weights and biases
            weight = self.weight_mu
            bias = self.bias_mu if self.bias_mu is not None else None

        return nn.functional.linear(input, weight, bias)

    @staticmethod
    def _scale_noise(size):
        # Factorized Gaussian noise: noise = sign(x)*sqrt(|x|)
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def adjust_sigma(self, sigma_scale):
        # Adjust the sigma parameters (noise magnitude) to scale exploration over time
        self.weight_sigma.data.mul_(sigma_scale)
        if self.bias_sigma is not None:
            self.bias_sigma.data.mul_(sigma_scale)
