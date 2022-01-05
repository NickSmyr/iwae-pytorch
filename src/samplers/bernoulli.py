from itertools import chain
from typing import Optional

import torch
from torch import nn

from ifaces import DistributionSampler
from samplers.gaussian import GaussianSampler


class BernoulliSampler(nn.Module, DistributionSampler):
    def __init__(self, mean_network: nn.Module):
        nn.Module.__init__(self)
        self.mean_network = mean_network

    def forward(self, x):
        return self.sample(x)

    def get_mean(self, x: torch.Tensor):
        return self.mean_network(x)

    def sample(self, shape_or_x: tuple or torch.Tensor) -> torch.Tensor:
        assert type(shape_or_x) == torch.Tensor
        mean = self.get_mean(x=shape_or_x)
        return torch.bernoulli(mean).type(mean.type())

    def log_likelihood(self, samples: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        mean = self.get_mean(x)
        return torch.sum(samples * torch.log(mean) + (1.0 - samples) * torch.log(1.0 - mean), dim=1)

    def last_linear_layer_weights_np(self):
        return self.mean_network[-2].weight.data.clone().detach().cpu().numpy()

    def first_linear_layer_weights_np(self):
        return self.mean_network[0].weight.data.clone().detach().cpu().numpy().T

    @staticmethod
    def random(n_units: list, bias: Optional[torch.Tensor or float] = None):
        mean_network_layers = list(chain.from_iterable(
            [nn.Linear(in_features=h_in, out_features=h_out), nn.Tanh()]
            for h_in, h_out in zip(n_units[:-2], n_units[1:-1])
        ))
        mean_network_layers.append(nn.Linear(in_features=n_units[-2], out_features=n_units[-1]))
        mean_network_layers.append(nn.Sigmoid())
        mean_network = nn.Sequential(*mean_network_layers)
        mean_network.apply(GaussianSampler.init_weights)
        if bias is not None:
            with torch.no_grad():
                mean_network[-2].bias.copy_(bias)
        return BernoulliSampler(mean_network)
