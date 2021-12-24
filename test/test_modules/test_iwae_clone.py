import unittest

import numpy as np
import torch
from matplotlib import pyplot as plt

from iwae_clone import IWAEClone


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
                                     data_type='continuous', device=self.device)

    def test_network_structure(self) -> None:
        x = torch.randn(1, self.w_in ** 2, device=self.device)
        print(self.iwae.prior)
        print(self.iwae.q_layers)
        print(self.iwae.p_layers)
        L_q = self.iwae(x, k=10, model_type='iwae')
        print(L_q)
        l_p_min = self.iwae.log_marginal_likelihood_estimate(x, k=10)
        print(l_p_min)

    def test_get_samples(self) -> None:
        samples = self.iwae.get_samples(num_samples=99)
        self.assertTupleEqual((10 * 28, 10 * 28), tuple(samples.shape))
        self.assertTrue(np.allclose(samples[-28:, -28:], 0.0))
        self.assertFalse(np.allclose(samples[:28, :28], 0.0))
        plt.imshow(samples, cmap='Greys')
        plt.show()

    def tearDown(self) -> None:
        del self.iwae
        torch.cuda.synchronize('cuda:0')
