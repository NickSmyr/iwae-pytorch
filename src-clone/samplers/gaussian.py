import math
import sys
from itertools import chain
from typing import Optional

import numpy as np
import torch
from torch import nn

from ifaces import DistributionSampler


class UnitGaussianSampler(nn.Module, DistributionSampler):
    def __init__(self, device: str = 'cpu'):
        nn.Module.__init__(self)
        self.device = device

    def forward(self, x):
        return self.sample(x)

    def sample(self, shape_or_x: tuple or torch.Tensor) -> torch.Tensor:
        if type(shape_or_x) == tuple:
            return torch.randn(shape_or_x).to(self.device)
        return torch.randn_like(shape_or_x)

    def log_likelihood(self, samples: torch.Tensor, x: torch.Tensor = None) -> torch.Tensor:
        """
        Given samples as rows of a matrix, returns their log-likelihood under the zero mean unit covariance
        Gaussian as a COLUMN vector.
        """
        return -0.5 * samples.shape[1] * math.log(2 * math.pi) - 0.5 * torch.sum(samples.square(), dim=1)


# noinspection PyMethodMayBeStatic
class Exponential(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


class GaussianSampler(nn.Module, DistributionSampler):
    def __init__(self, h_network: nn.Module, mean_network: nn.Module, sigma_network: nn.Module):
        nn.Module.__init__(self)
        self.h_network = h_network
        self.mean_network = mean_network
        self.sigma_network = sigma_network
        self.prior = UnitGaussianSampler()

    def forward(self, x):
        return self.sample(x)

    def get_mean_sigma(self, x: torch.Tensor):
        """
        Returns the mean and the square root of the covariance of the Gaussian.
        :param torch.Tensor x: input to the input layer
        """
        h = self.h_network(x)
        mean = self.mean_network(h)
        sigma = self.sigma_network(h)
        return mean, sigma

    def get_mean(self, x: torch.Tensor):
        """
        :param torch.Tensor x: input to the input layer
        """
        h = self.h_network(x)
        return self.mean_network(h)

    def sample(self, shape_or_x: tuple or torch.Tensor):
        assert type(shape_or_x) == torch.Tensor
        mean, sigma = self.get_mean_sigma(shape_or_x)
        unit_gaussian_samples = torch.randn_like(mean)
        return sigma * unit_gaussian_samples + mean

    @staticmethod
    def log_likelihood_for_mean_sigma(samples: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor):
        return -0.5 * samples.shape[1] * math.log(2 * np.pi) \
               - 0.5 * torch.sum(((samples - mean) / (sigma + sys.float_info.epsilon)).square() + 2.0 *
                                 torch.log(sigma + sys.float_info.epsilon), dim=1)

    def log_likelihood(self, samples: torch.Tensor, x: torch.Tensor):
        mean, sigma = self.get_mean_sigma(x)
        return GaussianSampler.log_likelihood_for_mean_sigma(samples, mean, sigma)

    def first_linear_layer_weights_np(self) -> np.ndarray:
        assert type(self.h_network[0]) == nn.Linear
        return self.h_network[0].weight.data.clone().detach().cpu().numpy().T

    @staticmethod
    @torch.no_grad()
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)

    @staticmethod
    def random(n_units: list, mean: Optional[torch.Tensor or float] = None):
        h_network = nn.Sequential(*list(chain.from_iterable(
            [nn.Linear(in_features=h_in, out_features=h_out), nn.Tanh()]
            for h_in, h_out in zip(n_units[:-2], n_units[1:-1])
        )))
        h_network.apply(GaussianSampler.init_weights)
        mean_network = nn.Linear(n_units[-2], n_units[-1])
        mean_network.apply(GaussianSampler.init_weights)
        if mean is not None:
            mean_network.bias.data.copy_(mean)
        sigma_network = nn.Sequential(
            nn.Linear(n_units[-2], n_units[-1]),
            Exponential(),
        )
        sigma_network.apply(GaussianSampler.init_weights)
        return GaussianSampler(h_network, mean_network, sigma_network)
