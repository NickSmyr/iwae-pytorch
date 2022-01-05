from itertools import chain
from typing import Optional, List
import math

import torch
from torch import nn

from utils.logging import CommandLineLogger


half_log2pi = 1.0/2.0 * math.log(2.0 * math.pi)


class Exponential(nn.Module):
    """
    An exponential output unit.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


def calc_log_likelihood_of_samples_gaussian(samples, mean, sigma):
    """
    Calculate log p(samples|mean, sigma) assuming a gaussian distribution
    """
    result = -torch.log(sigma) - half_log2pi - 1.0 / 2.0 * torch.square((samples - mean) / sigma)
    result = torch.sum(result, dim=1)
    return result


def calc_log_likelihood_of_samples_bernoulli(samples, mean):
    """
    Calculate log p(samples|mean) assuming a bernoulli distribution
    """
    result = samples * torch.log(mean) + (1.0 - samples) * torch.log(1.0 - mean)
    result = torch.sum(result, dim=1)
    return result


class GaussianStochasticLayer(nn.Module):
    """
    A gaussian stochastic layer.
    """
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        use_bias: bool = True,
        output_bias = None):
        """
        Constructor for a stochastic layer.
        :param (int) input_dim: input dimension for layer
        :param hidden_dims: number of neurons in encoder's hidden layers (and decoder's respective ones)
        :param (int) output_dim: output dimension for layer
        :param (bool) use_bias: set to True to add bias to the fully-connected (aka Linear) layers of the networks
        """
        nn.Module.__init__(self)

        # Hidden network
        self.hidden_network = nn.Sequential(*chain.from_iterable(
            [nn.Linear(h_in, h_out, bias=use_bias), nn.Tanh()]
            for h_in, h_out in zip([input_dim] + hidden_dims[:-1], hidden_dims[1:])
        ))

        # Sampler
        self.mean_network = nn.Linear(hidden_dims[-1], output_dim, bias=use_bias)
        if not output_bias is None:
            with torch.no_grad():
                self.mean_network.bias.copy_(output_bias)
        self.sigma_network = nn.Sequential(nn.Linear(hidden_dims[-1], output_dim), Exponential())


    def calc_mean_sigma(self, x):
        hidden = self.hidden_network(x)

        mean = self.mean_network(hidden)
        sigma = self.sigma_network(hidden)

        return mean, sigma


    def forward(self, x):
        """
        Calculate output samples from layer given input
        """
        mean, sigma = self.calc_mean_sigma(x)
        epsilon_samples = torch.randn_like(mean)
        samples = mean + epsilon_samples * sigma

        return samples


    def calc_log_likelihood(self, x, samples):
        """
        Calculate p(x|samples,theta)
        """
        mean, sigma = self.calc_mean_sigma(samples)
        p = calc_log_likelihood_of_samples_gaussian(x, mean, sigma)

        return p



class BernoulliStochasticLayer(nn.Module):
    """
    A bernoulli stochastic layer.
    """
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        use_bias: bool = True,
        output_bias = None):
        """
        Constructor for a stochastic layer.
        :param (int) input_dim: input dimension for layer
        :param hidden_dims: number of neurons in encoder's hidden layers (and decoder's respective ones)
        :param (int) output_dim: output dimension for layer
        :param (bool) use_bias: set to True to add bias to the fully-connected (aka Linear) layers of the networks
        """
        nn.Module.__init__(self)

        # Hidden network
        self.hidden_network = nn.Sequential(*chain.from_iterable(
            [nn.Linear(h_in, h_out, bias=use_bias), nn.Tanh()]
            for h_in, h_out in zip([input_dim] + hidden_dims[:-1], hidden_dims[1:])
        ))

        # Sampler
        mean_linear = nn.Linear(hidden_dims[-1], output_dim, bias=use_bias)
        if not output_bias is None:
            with torch.no_grad():
                mean_linear.bias.copy_(output_bias)
        self.mean_network = nn.Sequential(mean_linear, nn.Sigmoid())


    def calc_mean(self, x):
        hidden = self.hidden_network(x)
        mean = self.mean_network(hidden)

        return mean


    def forward(self, x):
        """
        Calculate output samples from layer given input
        """
        mean = self.calc_mean(x)
        samples = torch.bernoulli(mean)

        return samples


    def calc_log_likelihood(self, x, samples):
        """
        Calculate p(x|samples,theta)
        """
        mean = self.calc_mean(samples)
        p = calc_log_likelihood_of_samples_bernoulli(x, mean)

        return p



class VAE(nn.Module):
    """
    VAE Class:
    This class implements the entire VAE model architecture, while also inheriting from Pytorch's nn.Module.
    """

    def __init__(
        self,
        k: int = 1,
        c_in: int = 1,
        w_in_width: int = 28,
        w_in_height: int = 28,
        q_dim: int = 64,
        p_dim: Optional[int] = None,
        use_bias: bool = True,
        output_bias = None,
        hidden_dims: Optional[List[int]] = None,
        logger: Optional[CommandLineLogger] = None):
        """
        VAE class constructor.
        :param (int) k: number of samples per data point
        :param (int) c_in: number of input channels
        :param (int) w_in_width: width of input
        :param (int) w_in_height: height of input
        :param (int) q_dim: encoder's output dim
        :param (optional) p_dim: decoder's input dim (i.e. dimensionality of samples), or None to be the same as q_dim
        :param (bool) use_bias: set to True to add bias to the fully-connected (aka Linear) layers of the networks
        :param (optional) hidden_dims: number of neurons in encoder's hidden layers (and decoder's respective ones)
        :param (optional) logger: CommandLineLogger instance to be used when verbose is enabled
        """
        nn.Module.__init__(self)
        self.logger = CommandLineLogger(name=self.__class__.__name__) if logger is None else logger

        self.k = k

        # Set defaults
        if hidden_dims is None:
            hidden_dims = [1024, 512, 32]
        if p_dim is None:
            p_dim = q_dim

        input_dim = c_in * w_in_width * w_in_height

        # Encoder Network
        self.encoder = GaussianStochasticLayer(input_dim, hidden_dims, q_dim, use_bias, None)

        # Decoder Network
        self.decoder = BernoulliStochasticLayer(p_dim, list(reversed(hidden_dims)), input_dim, use_bias, output_bias)


    def forward(self, x):
        h_samples = self.encoder(x)
        x_samples = self.decoder(h_samples)

        return x_samples


    def calc_log_p(self, x, h_samples):
        """
        Calculate log p(x,h|theta)
        """
        # Calculate prior: log p(h|theta)
        prior = calc_log_likelihood_of_samples_gaussian(h_samples, torch.zeros_like(h_samples), torch.ones_like(h_samples))

        # Calculate log p(x|h,theta)
        p = self.decoder.calc_log_likelihood(x, h_samples)

        return prior + p


    def calc_log_q(self, h_samples, x):
        """
        Calculate log q(h|x)
        """
        q = self.encoder.calc_log_likelihood(h_samples, x)

        return q


    def calc_log_w(self, x):
        """
        Calculate log w(x,h,theta) = log p(x,h|theta) - log q(h|x)
        """
        h_samples = self.encoder.forward(x)

        log_p = self.calc_log_p(x, h_samples)
        log_q = self.calc_log_q(h_samples, x)

        return log_p - log_q


    def objective(self, x):
        """
        Estimate the negative lower bound on the log-likelihood
        (the negative lower bound is used to have a function that should be minimized)
        """
        x = x.repeat_interleave(self.k, dim=0)
        log_w = self.calc_log_w(x)

        return -torch.sum(log_w) / self.k


    def estimate_loss(self, x, k=1):
        """
        Estimate the log-likelihood, averaging over k samples for each data point
        """
        x = x.repeat_interleave(k, dim=0)

        log_w = self.calc_log_w(x)

        log_w = log_w.reshape(-1, k)

        max_values = log_w.max(dim=1, keepdim=True).values
        log_mean_exp = max_values + torch.log(torch.mean(torch.exp(log_w - max_values), dim=1, keepdim=True))

        return log_mean_exp
