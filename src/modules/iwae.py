from typing import Optional, List

import torch

from utils.logging import CommandLineLogger

import modules.vae as vae


class IWAE(vae.VAE):
    """
    IWAE Class:
    This class implements the IWAE model architecture, inheriting from VAE.
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
        IWAE class constructor.
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
        vae.VAE.__init__(self, k, c_in, w_in_width, w_in_height, q_dim, p_dim, use_bias, output_bias, hidden_dims, logger)


    def objective(self, x):
        """
        Estimate the negative lower bound on the log-likelihood
        (the negative lower bound is used to have a function that should be minimized)
        """
        x = x.repeat_interleave(self.k, dim=0)
        log_w = self.calc_log_w(x)

        detached_log_w = log_w.clone().detach()
        log_w_matrix = detached_log_w.reshape(-1, self.k)
        log_w_max_values = log_w_matrix.max(dim=1, keepdim=True).values
        w = torch.exp(log_w_matrix - log_w_max_values)
        w_normalized_matrix = w / torch.sum(w, dim=1, keepdim=True)
        w_normalized = w_normalized_matrix.reshape(log_w.shape)

        return -torch.dot(w_normalized.detach(), log_w)
