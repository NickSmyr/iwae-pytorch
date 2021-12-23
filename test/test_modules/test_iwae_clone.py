import unittest

import torch

from iwae import IWAEClone


class TestIWAECloneModule(unittest.TestCase):

    def setUp(self) -> None:
        self.c_in, self.w_in = 1, 28
        data_dimension = self.c_in * self.w_in ** 2
        latent_units = [50]
        hidden_units_q = [[200, 200]]
        hidden_units_p = [[200, 200]]
        self.device = 'cuda'
        self.iwae = IWAEClone.random(latent_units=[data_dimension] + latent_units,
                                     hidden_units_q=hidden_units_q,
                                     hidden_units_p=hidden_units_p,
                                     data_type='binary', device=self.device)

    def test_network_structure(self) -> None:
        x = torch.randn(1, self.w_in ** 2, device=self.device)
        print(self.iwae.prior)
        print(self.iwae.q_layers)
        print(self.iwae.p_layers)
        L_q = self.iwae(x, k=10, model_type='iwae')
        print(L_q)
        l_p_min = self.iwae.log_marginal_likelihood_estimate(x, k=10)
        print(l_p_min)
