from itertools import chain
from typing import Optional, List

import torch
from torch import nn


class AE(nn.Module):
    """
    AE Class:
    Simple deterministic autoencoder
    """

    def __init__(self, c_in: int = 1, w_in: int = 28, latent_dim: int = 2,
                 use_bias: bool = True, hidden_dims: Optional[List[int]] = None):
        """
        AE class constructor.
        :param (int) c_in: number of input channels
        :param (int) w_in: width (and height) of input
        :param (int) latent_dim: latent space dim
        :param (bool) use_bias: set to True to add bias to the fully-connected (aka Linear) layers of the networks
        :param (optional) hidden_dims: number of neurons in encoder's hidden layers (and decoder's respective ones)
        """
        nn.Module.__init__(self)

        # Set defaults
        if hidden_dims is None:
            hidden_dims = [256]

        # Encoder Network
        enc_hidden_dims = [c_in * w_in ** 2] + hidden_dims + [latent_dim]
        self.encoder = nn.Sequential(
            *list(
                chain.from_iterable(
                    [
                        nn.Linear(
                            in_features=h_in,
                            out_features=h_out,
                            bias=use_bias),
                        nn.ReLU()
                    ] for h_in, h_out in zip(enc_hidden_dims[:-1], enc_hidden_dims[1:])
        )))

        # Decoder Network
        dec_hidden_dims = list(reversed(enc_hidden_dims))
        self.decoder = nn.Sequential(*list(chain.from_iterable(
            [nn.Linear(in_features=h_in, out_features=h_out, bias=use_bias), nn.ReLU()]
            for h_in, h_out in zip(dec_hidden_dims[:-1], dec_hidden_dims[1:])
        )))

        self.whole = nn.Sequential(self.encoder, self.decoder)


    def forward(self, x):
        return self.whole(x)
