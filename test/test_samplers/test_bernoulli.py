import unittest

import torch
from torch import nn

from samplers.bernoulli import BernoulliSampler


class TestBernoulliSampler(unittest.TestCase):

    def setUp(self) -> None:
        # q network with 1 stochastic layer
        self.n_units_1 = [100, 200, 200, 28 * 28]
        self.sampler_1 = BernoulliSampler.random(self.n_units_1)
        # q network with 2 stochastic layers
        self.n_units_2 = [100, 200, 200, 100, 100, 28 * 28]
        self.sampler_2 = BernoulliSampler.random(self.n_units_2)

    def test_networks(self) -> None:
        # Sampler 1
        self.assertEqual(type(self.sampler_1.mean_network), nn.Sequential)
        self.assertEqual(6, len(self.sampler_1.mean_network))
        self.assertEqual(type(self.sampler_1.mean_network[0]), nn.Linear)
        self.assertTupleEqual((100, 200), (self.sampler_1.mean_network[0].in_features,
                                           self.sampler_1.mean_network[0].out_features))
        self.assertEqual(type(self.sampler_1.mean_network[1]), nn.Tanh)
        self.assertEqual(type(self.sampler_1.mean_network[2]), nn.Linear)
        self.assertTupleEqual((200, 200), (self.sampler_1.mean_network[2].in_features,
                                           self.sampler_1.mean_network[2].out_features))
        self.assertEqual(type(self.sampler_1.mean_network[3]), nn.Tanh)
        self.assertEqual(type(self.sampler_1.mean_network[4]), nn.Linear)
        self.assertTupleEqual((200, 28 * 28), (self.sampler_1.mean_network[4].in_features,
                                               self.sampler_1.mean_network[4].out_features))
        self.assertEqual(type(self.sampler_1.mean_network[5]), nn.Sigmoid)
        x_in = torch.randn((1, self.n_units_1[0]))
        self.assertTupleEqual((1, self.n_units_1[-1]), tuple(self.sampler_1.mean_network(x_in).shape))
        # Sampler 2
        self.assertEqual(type(self.sampler_2.mean_network), nn.Sequential)
        self.assertEqual(10, len(self.sampler_2.mean_network))
        self.assertEqual(type(self.sampler_2.mean_network[0]), nn.Linear)
        self.assertTupleEqual((100, 200), (self.sampler_2.mean_network[0].in_features,
                                           self.sampler_2.mean_network[0].out_features))
        self.assertEqual(type(self.sampler_2.mean_network[1]), nn.Tanh)
        self.assertEqual(type(self.sampler_2.mean_network[2]), nn.Linear)
        self.assertTupleEqual((200, 200), (self.sampler_2.mean_network[2].in_features,
                                           self.sampler_2.mean_network[2].out_features))
        self.assertEqual(type(self.sampler_2.mean_network[3]), nn.Tanh)
        self.assertEqual(type(self.sampler_2.mean_network[4]), nn.Linear)
        self.assertTupleEqual((200, 100), (self.sampler_2.mean_network[4].in_features,
                                           self.sampler_2.mean_network[4].out_features))
        self.assertEqual(type(self.sampler_2.mean_network[5]), nn.Tanh)
        self.assertEqual(type(self.sampler_2.mean_network[6]), nn.Linear)
        self.assertTupleEqual((100, 100), (self.sampler_2.mean_network[6].in_features,
                                           self.sampler_2.mean_network[6].out_features))
        self.assertEqual(type(self.sampler_2.mean_network[7]), nn.Tanh)
        self.assertEqual(type(self.sampler_2.mean_network[8]), nn.Linear)
        self.assertTupleEqual((100, 28 * 28), (self.sampler_2.mean_network[8].in_features,
                                               self.sampler_2.mean_network[8].out_features))
        self.assertEqual(type(self.sampler_2.mean_network[9]), nn.Sigmoid)
        x_in = torch.randn((1, self.n_units_2[0]))
        self.assertTupleEqual((1, self.n_units_2[-1]), tuple(self.sampler_2.mean_network(x_in).shape))

    def test_samples(self):
        # Sampler 1
        samples = self.sampler_1.sample(shape_or_x=torch.randn((1, self.n_units_1[0])))
        self.assertTupleEqual((1, self.n_units_1[-1]), tuple(samples.shape))
        samples = self.sampler_1.sample(shape_or_x=torch.randn((10, self.n_units_1[0])))
        self.assertTupleEqual((10, self.n_units_1[-1]), tuple(samples.shape))
        # Sampler 2
        samples = self.sampler_2.sample(shape_or_x=torch.randn((1, self.n_units_2[0])))
        self.assertTupleEqual((1, self.n_units_2[-1]), tuple(samples.shape))
        samples = self.sampler_2.sample(shape_or_x=torch.randn((10, self.n_units_2[0])))
        self.assertTupleEqual((10, self.n_units_2[-1]), tuple(samples.shape))

    def test_log_likelihood(self):
        lls = self.sampler_1.log_likelihood(samples=torch.randn((1, self.n_units_1[-1])),
                                            x=torch.randn((1, self.n_units_1[0])))
        self.assertTupleEqual((1,), tuple(lls.shape))
        lls = self.sampler_1.log_likelihood(samples=torch.randn((10, self.n_units_1[-1])),
                                            x=torch.randn((1, self.n_units_1[0])))
        self.assertTupleEqual((10,), tuple(lls.shape))
        lls = self.sampler_1.log_likelihood(samples=torch.randn((1, self.n_units_1[-1])),
                                            x=torch.randn((10, self.n_units_1[0])))
        self.assertTupleEqual((10,), tuple(lls.shape))
        lls = self.sampler_1.log_likelihood(samples=torch.randn((10, self.n_units_1[-1])),
                                            x=torch.randn((10, self.n_units_1[0])))
        self.assertTupleEqual((10,), tuple(lls.shape))
