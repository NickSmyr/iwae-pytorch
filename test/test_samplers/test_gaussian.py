import math
import unittest

import numpy as np
import torch
from torch import nn

from samplers.gaussian import UnitGaussianSampler, GaussianSampler, Exponential


class TestUnitGaussianSampler(unittest.TestCase):

    def setUp(self) -> None:
        self.sampler = UnitGaussianSampler()
        self.samples_1 = torch.from_numpy(np.array([
            [0, 0]
        ]))
        self.samples_2 = torch.from_numpy(np.array([
            [0, 0],
            [0, 0],
        ]))
        self.samples_3 = torch.from_numpy(np.array([
            [10, 0],
            [0, 10],
            [10, 10],
        ]))

    def test_samples(self):
        samples = self.sampler.sample(shape_or_x=(1, 1))
        self.assertTupleEqual((1, 1), tuple(samples.shape))
        samples = self.sampler.sample(shape_or_x=(1, 30))
        self.assertTupleEqual((1, 30), tuple(samples.shape))
        samples = self.sampler.sample(shape_or_x=(1000, 2))
        self.assertTupleEqual((1000, 2), tuple(samples.shape))
        samples = self.sampler.sample(shape_or_x=(100000, 2))
        sample_mean = torch.mean(samples, dim=0).numpy()
        self.assertTrue(np.allclose(sample_mean, 0.0, rtol=1e-2, atol=1e-2))

    def test_log_likelihood(self):
        lls = self.sampler.log_likelihood(self.samples_1)
        self.assertTupleEqual((1,), tuple(lls.shape))
        self.assertTrue(np.isclose(-math.log(2 * math.pi), lls.item()))
        lls = self.sampler.log_likelihood(self.samples_2)
        self.assertTupleEqual((2,), tuple(lls.shape))
        self.assertTrue(np.allclose(lls, -math.log(2 * math.pi)))
        lls = self.sampler.log_likelihood(self.samples_3)
        self.assertTrue(np.allclose(torch.exp(lls), 0.0))
        self.assertEqual(-0.5 * (2 * math.log(2 * math.pi) + self.samples_3[0].pow(2).sum()), lls[0])
        self.assertEqual(-0.5 * (2 * math.log(2 * math.pi) + self.samples_3[1].pow(2).sum()), lls[1])
        self.assertEqual(-0.5 * (2 * math.log(2 * math.pi) + self.samples_3[2].pow(2).sum()), lls[2])


class TestGaussianSampler(unittest.TestCase):

    def setUp(self) -> None:
        # q network with 1 stochastic layer
        self.n_units_1 = [28 * 28, 200, 200, 50]
        self.sampler_1 = GaussianSampler.random(self.n_units_1)
        # q network with 2 stochastic layers
        self.n_units_2 = [28 * 28, 200, 200, 100, 100, 2]
        self.sampler_2 = GaussianSampler.random(self.n_units_2)

    def test_networks(self) -> None:
        # Sampler 1
        self.assertEqual(type(self.sampler_1.h_network), nn.Sequential)
        self.assertEqual(4, len(self.sampler_1.h_network))
        self.assertEqual(type(self.sampler_1.h_network[0]), nn.Linear)
        self.assertTupleEqual((28 * 28, 200), (self.sampler_1.h_network[0].in_features,
                                               self.sampler_1.h_network[0].out_features))
        self.assertEqual(type(self.sampler_1.h_network[1]), nn.Tanh)
        self.assertEqual(type(self.sampler_1.h_network[2]), nn.Linear)
        self.assertTupleEqual((200, 200), (self.sampler_1.h_network[2].in_features,
                                           self.sampler_1.h_network[2].out_features))
        self.assertEqual(type(self.sampler_1.h_network[3]), nn.Tanh)
        self.assertEqual(type(self.sampler_1.mean_network), nn.Linear)
        self.assertTupleEqual((200, 50), (self.sampler_1.mean_network.in_features,
                                          self.sampler_1.mean_network.out_features))
        self.assertEqual(type(self.sampler_1.sigma_network), nn.Sequential)
        self.assertEqual(type(self.sampler_1.sigma_network[0]), nn.Linear)
        self.assertTupleEqual((200, 50), (self.sampler_1.sigma_network[0].in_features,
                                          self.sampler_1.sigma_network[0].out_features))
        self.assertEqual(type(self.sampler_1.sigma_network[1]), Exponential)
        x_in = torch.randn((1, self.n_units_1[0]))
        self.assertTupleEqual((1, self.n_units_1[-1]), tuple(self.sampler_1.mean_network(
            self.sampler_1.h_network(x_in)
        ).shape))
        # Sampler 2
        self.assertEqual(type(self.sampler_2.h_network), nn.Sequential)
        self.assertEqual(8, len(self.sampler_2.h_network))
        self.assertEqual(type(self.sampler_2.h_network[0]), nn.Linear)
        self.assertTupleEqual((28 * 28, 200), (self.sampler_2.h_network[0].in_features,
                                               self.sampler_2.h_network[0].out_features))
        self.assertEqual(type(self.sampler_2.h_network[1]), nn.Tanh)
        self.assertEqual(type(self.sampler_2.h_network[2]), nn.Linear)
        self.assertTupleEqual((200, 200), (self.sampler_2.h_network[2].in_features,
                                           self.sampler_2.h_network[2].out_features))
        self.assertEqual(type(self.sampler_2.h_network[3]), nn.Tanh)
        self.assertEqual(type(self.sampler_2.h_network[4]), nn.Linear)
        self.assertTupleEqual((200, 100), (self.sampler_2.h_network[4].in_features,
                                           self.sampler_2.h_network[4].out_features))
        self.assertEqual(type(self.sampler_2.h_network[5]), nn.Tanh)
        self.assertEqual(type(self.sampler_2.h_network[6]), nn.Linear)
        self.assertTupleEqual((100, 100), (self.sampler_2.h_network[6].in_features,
                                           self.sampler_2.h_network[6].out_features))
        self.assertEqual(type(self.sampler_2.h_network[7]), nn.Tanh)
        self.assertEqual(type(self.sampler_2.mean_network), nn.Linear)
        self.assertTupleEqual((100, 2), (self.sampler_2.mean_network.in_features,
                                         self.sampler_2.mean_network.out_features))
        self.assertEqual(type(self.sampler_2.sigma_network), nn.Sequential)
        self.assertEqual(type(self.sampler_2.sigma_network[0]), nn.Linear)
        self.assertTupleEqual((100, 2), (self.sampler_2.sigma_network[0].in_features,
                                         self.sampler_2.sigma_network[0].out_features))
        self.assertEqual(type(self.sampler_2.sigma_network[1]), Exponential)
        x_in = torch.randn((1, self.n_units_2[0]))
        self.assertTupleEqual((1, self.n_units_2[-1]), tuple(self.sampler_2.mean_network(
            self.sampler_2.h_network(x_in)
        ).shape))

    def test_samples(self):
        # Sampler 1
        samples = self.sampler_1.sample(shape_or_x=torch.randn((1, 28 * 28)))
        self.assertTupleEqual((1, self.n_units_1[-1]), tuple(samples.shape))
        samples = self.sampler_1.sample(shape_or_x=torch.randn((10, 28 * 28)))
        self.assertTupleEqual((10, self.n_units_1[-1]), tuple(samples.shape))
        # Sampler 2
        samples = self.sampler_2.sample(shape_or_x=torch.randn((1, 28 * 28)))
        self.assertTupleEqual((1, self.n_units_2[-1]), tuple(samples.shape))
        samples = self.sampler_2.sample(shape_or_x=torch.randn((10, 28 * 28)))
        self.assertTupleEqual((10, self.n_units_2[-1]), tuple(samples.shape))

    def test_log_likelihood(self):
        lls = self.sampler_1.log_likelihood(samples=torch.randn((1, self.n_units_1[-1])), x=torch.randn((1, 28 * 28)))
        self.assertTupleEqual((1,), tuple(lls.shape))
        lls = self.sampler_1.log_likelihood(samples=torch.randn((10, self.n_units_1[-1])), x=torch.randn((1, 28 * 28)))
        self.assertTupleEqual((10,), tuple(lls.shape))
        lls = self.sampler_1.log_likelihood(samples=torch.randn((1, self.n_units_1[-1])), x=torch.randn((10, 28 * 28)))
        self.assertTupleEqual((10,), tuple(lls.shape))
        lls = self.sampler_1.log_likelihood(samples=torch.randn((10, self.n_units_1[-1])), x=torch.randn((10, 28 * 28)))
        self.assertTupleEqual((10,), tuple(lls.shape))
