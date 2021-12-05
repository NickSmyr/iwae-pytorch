from itertools import chain
from typing import Optional, List

from torch import nn

from utils.logging import CommandLineLogger


class IWAE(nn.Module):
    """
    IWAE Class:
    This class implements the entire IWAE model architecture, while also inheriting from Pytorch's nn.Module.
    """

    def __init__(self, c_in: int = 1, w_in: int = 28, q_dim: int = 64, p_dim: Optional[int] = None,
                 use_bias: bool = True, hidden_dims: Optional[List[int]] = None,
                 logger: Optional[CommandLineLogger] = None):
        """
        IWAE class constructor.
        :param (int) c_in: number of input channels
        :param (int) w_in: width (and height) of input
        :param (int) q_dim: encoder's output dim
        :param (optional) p_dim: decoder's input dim (i.e. dimensionality of samples), or None to be the same as q_dim
        :param (bool) use_bias: set to True to add bias to the fully-connected (aka Linear) layers of the networks
        :param (optional) hidden_dims: number of neurons in encoder's hidden layers (and decoder's respective ones)
        :param (optional) logger: CommandLineLogger instance to be used when verbose is enabled
        """
        nn.Module.__init__(self)
        self.logger = CommandLineLogger(name=self.__class__.__name__) if logger is None else logger

        # Set defaults
        if hidden_dims is None:
            hidden_dims = [1024, 512, 32]
        if p_dim is None:
            p_dim = q_dim

        # Encoder Network
        enc_hidden_dims = [c_in * w_in ** 2] + hidden_dims
        self.encoder = nn.Sequential(*list(chain.from_iterable(
            [nn.Linear(in_features=h_in, out_features=h_out, bias=use_bias), nn.Tanh()]
            for h_in, h_out in zip(enc_hidden_dims[:-1], enc_hidden_dims[1:])
        )))

        # Encoder Sampler
        # TODO

        # Decoder Network
        dec_hidden_dims = [p_dim] + list(reversed(hidden_dims))
        self.decoder = nn.Sequential(*list(chain.from_iterable(
            [nn.Linear(in_features=h_in, out_features=h_out, bias=use_bias), nn.Tanh()]
            for h_in, h_out in zip(dec_hidden_dims[:-1], dec_hidden_dims[1:])
        )))

        # Decoder Sampler
        # TODO
