import sys
from itertools import chain
from typing import Optional
import math

import numpy as np
import torch
from torch import nn

from samplers.gaussian import Exponential
from utils.logging import CommandLineLogger
from utils_clone.pytorch import reshape_and_tile_images

half_log2pi = 1.0 / 2.0 * math.log(2.0 * math.pi)


def calc_log_likelihood_of_samples_gaussian(samples, mean, sigma):
    """
    Calculate log p(samples|mean, sigma) assuming a gaussian distribution
    """
    eps = sys.float_info.epsilon
    result = -torch.log(sigma + eps) - half_log2pi - 1.0 / 2.0 * torch.square((samples - mean) / sigma)
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
            output_bias=None):
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
            for h_in, h_out in zip([input_dim] + hidden_dims[:-1], hidden_dims)
        ))

        # Sampler
        self.mean_network = nn.Linear(hidden_dims[-1], output_dim, bias=use_bias)
        if output_bias is not None:
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

    def first_linear_layer_weights_np(self) -> np.ndarray:
        assert type(self.hidden_network[0]) == nn.Linear
        return self.hidden_network[0].weight.data.clone().detach().cpu().numpy().T


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
            output_bias=None):
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
            [nn.Linear(h_in, h_out, bias=use_bias), nn.Tanh(), nn.Linear(h_out, h_out, bias=use_bias), nn.Tanh()]
            for h_in, h_out in zip([input_dim] + hidden_dims[:-1], hidden_dims[1:])
        ))

        # Sampler
        mean_linear = nn.Linear(hidden_dims[-1], output_dim, bias=use_bias)
        if output_bias is not None:
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
        samples = torch.bernoulli(mean).type(mean.type())
        return samples

    def calc_log_likelihood(self, x, samples):
        """
        Calculate p(x|samples,theta)
        """
        mean = self.calc_mean(samples)
        eps = sys.float_info.epsilon
        p = calc_log_likelihood_of_samples_bernoulli(x, mean + eps)

        return p

    def first_linear_layer_weights_np(self):
        return self.mean_network[0].weight.data.clone().detach().cpu().numpy().T


class VAE(nn.Module):
    """
    VAE Class:
    This class implements the entire VAE model architecture, while also inheriting from Pytorch's nn.Module.
    """

    def __init__(
            self,
            k: int = 1,
            latent_units=None,
            hidden_units_q=None,
            hidden_units_p=None,
            output_bias=None,
            logger: Optional[CommandLineLogger] = None):
        """
        VAE class constructor.
        :param (int) k: number of samples per data point
        :param (list) latent_units: number of neurons in the output of every stochastic layer
        :param (list) hidden_units_q: number of neurons in encoder's stochastic layers
        :param (list) hidden_units_p: number of neurons in decoder's stochastic layers
        :param (bool) output_bias: hardcoded bias of output layer
        :param (optional) logger: CommandLineLogger instance to be used when verbose is enabled
        """
        nn.Module.__init__(self)
        self.logger = CommandLineLogger(name=self.__class__.__name__) if logger is None else logger

        self.k = k

        # Set default arguments
        if latent_units is None:
            latent_units = [50]
        if hidden_units_p is None:
            hidden_units_p = [[200, 200]]
        if hidden_units_q is None:
            hidden_units_q = [[200, 200]]

        self.embedding_dim = latent_units[-1]
        # Encoder Network
        layers_q = []
        for units_prev, units_next, hidden_units in zip(latent_units, latent_units[1:], hidden_units_q):
            layers_q.append(
                GaussianStochasticLayer(units_prev, hidden_units, units_next, use_bias=True, output_bias=None)
            )
        self.encoder_layers = nn.ModuleList(layers_q)
        self.encoder = nn.Sequential(*self.encoder_layers)

        # Decoder Network
        layers_p = []
        for units_prev, units_next, hidden_units in \
                zip(list(reversed(latent_units))[:-1], list(reversed(latent_units))[1:-1], hidden_units_p[:-1]):
            layers_p.append(
                GaussianStochasticLayer(units_prev, hidden_units, units_next, use_bias=True, output_bias=None)
            )
        # Last layer is a Bernoulli
        layers_p.append(
            BernoulliStochasticLayer(latent_units[1], hidden_units_p[-1], latent_units[0], use_bias=True,
                                     output_bias=output_bias)
        )
        self.decoder_layers = nn.ModuleList(layers_p)
        self.decoder = nn.Sequential(*self.decoder_layers)
        # We reverse the whole inputs values
        # hidden_dims_reversed = list(reversed([list(reversed(hidden_dims[i])) for i in range(len(hidden_dims))]))
        # q_dim_reversed = list(reversed(q_dim))
        #
        # self.decoder_layers = torch.nn.ModuleList()
        #
        # self.decoder_layers.append(
        #     GaussianStochasticLayer(p_dim, hidden_dims_reversed[0], q_dim_reversed[0], use_bias, None))
        #
        # for l in range(1, L - 1):
        #     self.decoder_layers.append(
        #         GaussianStochasticLayer(q_dim_reversed[l - 1], hidden_dims_reversed[l], q_dim_reversed[l], use_bias,
        #                                 None)
        #     )
        # # Last layer is a Bernoulli
        # self.decoder_layers.append(
        #     BernoulliStochasticLayer(q_dim_reversed[L - 1], hidden_dims_reversed[L - 1], input_dim, use_bias, None))
        #
        # self.decoder = nn.Sequential(*self.decoder_layers)
        # # self.decoder = BernoulliStochasticLayer(p_dim, list(reversed(hidden_dims)), input_dim, use_bias, output_bias)

    def forward(self, x):
        h_samples = self.encoder(x)
        x_samples = self.decoder(h_samples)

        return x_samples

    def calc_log_p(self, x, h_samples):
        """
        Calculate log p(x,h|theta)
        """
        # Calculate prior: log p(h|theta)
        prior = calc_log_likelihood_of_samples_gaussian(h_samples, torch.zeros_like(h_samples),
                                                        torch.ones_like(h_samples))

        # Calculate log p(x|h,theta)
        p = 0
        p += self.decoder_layers[-1].calc_log_likelihood(x, h_samples)

        return prior + p

    def calc_log_q(self, h_samples, x):
        """
        Calculate log q(h|x)
        """
        q_list = []
        samples = [h_samples]
        for i in range(len(self.encoder_layers)):
            # q_list.append(self.encoder_layers[i].calc_log_likelihood(samples[i], x))
            samples.append(self.encoder_layers[i](samples[i]))

        return sum(q_list)

    def calc_log_w(self, x):
        """
        Calculate log w(x,h,theta) = log p(x,h|theta) - log q(h|x)
        """
        # h_samples = self.encoder(x)
        # log_p = self.calc_log_p(x, h_samples)
        # log_q = self.calc_log_q(h_samples, x)
        # return log_p - log_q
        h_samples = [x]
        for layer in self.encoder_layers:
            h_samples.append(layer(h_samples[-1]))
        return self.log_weights_from_q_samples(h_samples)

    def log_weights_from_q_samples(self, q_samples):
        log_weights = torch.zeros(q_samples[-1].shape[0], device=q_samples[-1].device)
        for layer_q, layer_p, prev_sample, next_sample in zip(self.encoder_layers, reversed(self.decoder_layers),
                                                              q_samples, q_samples[1:]):
            log_weights += layer_p.calc_log_likelihood(prev_sample.clone(), next_sample.clone()) - \
                           layer_q.calc_log_likelihood(next_sample, prev_sample)
        # log_weights += self.prior.log_likelihood(q_samples[-1])
        log_weights += calc_log_likelihood_of_samples_gaussian(q_samples[-1], torch.zeros_like(q_samples[-1]),
                                                               torch.ones_like(q_samples[-1]))
        return log_weights

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

    def first_p_layer_weights_np(self):
        return self.decoder_layers[0].first_linear_layer_weights_np()

    def get_samples(self, num_samples, device='cpu'):
        prior_samples = torch.randn((num_samples, self.embedding_dim)).to(device)
        samples = [prior_samples]
        for layer in self.decoder_layers[:-1]:
            samples.append(layer(samples[-1]))
        samples_function = self.decoder_layers[-1].calc_mean(samples[-1])
        return reshape_and_tile_images(samples_function.detach().cpu().numpy())
