from itertools import chain
from typing import Optional, List
import math

import torch
from torch import nn

from utils.logging import CommandLineLogger


half_log2pi = 1.0/2.0 * math.log(2.0 * math.pi)


class Exponential(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


def calc_log_likelihood_of_samples(samples, mean, sigma):
    result = -torch.log(sigma) - half_log2pi - 1.0 / 2.0 * torch.square((samples - mean) / sigma)
    result = torch.sum(result, dim=1)
    return result


class VAE(nn.Module):
    """
    VAE Class:
    This class implements the entire VAE model architecture, while also inheriting from Pytorch's nn.Module.
    """

    def __init__(
        self,
        c_in: int = 1,
        w_in_width: int = 28,
        w_in_height: int = 28,
        q_dim: int = 64,
        p_dim: Optional[int] = None,
        device = "cpu",
        use_bias: bool = True,
        output_bias = None,
        hidden_dims: Optional[List[int]] = None,
        logger: Optional[CommandLineLogger] = None):
        """
        VAE class constructor.
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

        self.device = device

        # Set defaults
        if hidden_dims is None:
            hidden_dims = [1024, 512, 32]
        if p_dim is None:
            p_dim = q_dim

        input_dim = c_in * w_in_width * w_in_height

        # Encoder Network
        enc_hidden_dims = [input_dim] + hidden_dims
        self.encoder = nn.Sequential(*chain.from_iterable(
            [nn.Linear(h_in, h_out, bias=use_bias), nn.Tanh()]
            for h_in, h_out in zip(enc_hidden_dims[:-1], enc_hidden_dims[1:])
        ))

        # Encoder Sampler
        self.encoder_mean = nn.Linear(hidden_dims[-1], q_dim, bias=use_bias)
        self.encoder_sigma = nn.Sequential(nn.Linear(hidden_dims[-1], q_dim), Exponential())

        # Decoder Network
        dec_hidden_dims = [p_dim] + list(reversed(hidden_dims))
        self.decoder = nn.Sequential(*chain.from_iterable(
            [nn.Linear(h_in, h_out, bias=use_bias), nn.Tanh()]
            for h_in, h_out in zip(dec_hidden_dims[:-1], dec_hidden_dims[1:])
        ))

        # Decoder Sampler
        self.decoder_mean = nn.Linear(hidden_dims[0], input_dim, bias=use_bias)
        if not output_bias is None:
            with torch.no_grad():
                self.decoder_mean.bias.copy_(output_bias)
        self.decoder_sigma = nn.Sequential(nn.Linear(hidden_dims[0], input_dim), Exponential())


    def forward(self, x):
        determinstic_h = self.encoder(x)
        enc_mean = self.encoder_mean(determinstic_h)
        enc_sigma = self.encoder_sigma(determinstic_h)

        epsilon_samples = torch.randn(enc_mean.size()).to(device=self.device)

        h_samples = enc_mean + epsilon_samples * enc_sigma

        determinstic_h_dec = self.decoder(h_samples)
        dec_mean = self.decoder_mean(determinstic_h_dec)
        dec_sigma = self.decoder_sigma(determinstic_h_dec)

        epsilon_samples_dec = torch.randn(dec_mean.size()).to(device=self.device)

        x_samples = dec_mean + epsilon_samples_dec * dec_sigma

        return x_samples


    def calc_log_p(self, x, h_samples):
        """
        Calculate p(x,h|theta)
        """

        # Calculate prior: log p(h|theta)

        prior = calc_log_likelihood_of_samples(h_samples, torch.zeros_like(h_samples), torch.ones_like(h_samples))

        # Calculate p(x|h,theta)
        deterministic_h = self.decoder(h_samples)
        mean = self.decoder_mean(deterministic_h)
        sigma = self.decoder_sigma(deterministic_h)

        p = calc_log_likelihood_of_samples(x, mean, sigma)

        return prior + p


    def calc_log_q(self, h_samples, x):
        """
        Calculate q(h|x)
        """
        deterministic_h = self.encoder(x)
        mean = self.encoder_mean(deterministic_h)
        sigma = self.encoder_sigma(deterministic_h)

        q = calc_log_likelihood_of_samples(h_samples, mean, sigma)

        return q


    def objective(self, x):
        """
        Estimate the negative lower bound on the log-likelihood
        (the negative lower bound is used to have a function that should be minimized)
        """
        deterministic_h = self.encoder(x)
        enc_mean = self.encoder_mean(deterministic_h)
        enc_sigma = self.encoder_sigma(deterministic_h)

        epsilon_samples = torch.randn(enc_mean.size()).to(device=self.device)

        h_samples = enc_mean + epsilon_samples * enc_sigma

        log_p = self.calc_log_p(x, h_samples)
        log_q = self.calc_log_q(h_samples, x)

        return -torch.mean(log_p - log_q)
