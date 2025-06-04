# NoisyLinear.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# NoisyLinear implements a linear layer with parametric noise added to the weights and biases.
# This technique is used in NoisyNet DQN to encourage exploration without an epsilon parameter.
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017, use_cuda=False):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_cuda = use_cuda

        # Mean and sigma parameters for weights and biases
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        # Epsilon terms to be sampled each forward pass
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.sigma_init = sigma_init

        # Initialize parameters and reset noise
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # Initialize weight_mu and bias_mu uniformly within a range based on in_features
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        # Initialize sigma parameters (standard deviation of noise)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        # Sample new epsilon values for weights and biases
        if self.use_cuda:
            self.weight_epsilon.copy_(torch.randn(self.out_features, self.in_features).cuda())
            self.bias_epsilon.copy_(torch.randn(self.out_features).cuda())
        else:
            self.weight_epsilon.copy_(torch.randn(self.out_features, self.in_features))
            self.bias_epsilon.copy_(torch.randn(self.out_features))

    def forward(self, input):
        # During training, add noise to weights and biases
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # During evaluation, use only the mean parameters
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(input, weight, bias)

    def set_sigma(self, sigma):
        # Adjust the sigma (noise scale) used for this layer
        self.weight_sigma.data.fill_(sigma / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(sigma / np.sqrt(self.out_features))

    def get_noise_norm(self):
        # Compute the squared norm of the noise parameters (weight_sigma and bias_sigma)
        return self.weight_sigma.norm() ** 2 + self.bias_sigma.norm() ** 2
