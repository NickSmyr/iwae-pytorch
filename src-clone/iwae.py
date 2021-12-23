import torch
from torch import nn

from samplers.bernoulli import BernoulliSampler
from samplers.gaussian import GaussianSampler, UnitGaussianSampler


class IWAEClone(nn.Module):
    def __init__(self, q_layers, p_layers, prior):
        nn.Module.__init__(self)
        self.q_layers = q_layers
        self.p_layers = p_layers
        self.prior = prior

    def q_samples(self, x):
        samples = [x]
        for layer in self.q_layers:
            samples.append(layer(samples[-1]))
        return samples

    def log_weights_from_q_samples(self, q_samples):
        log_weights = torch.zeros(q_samples[-1].shape[0], device=q_samples[-1].device)
        for layer_q, layer_p, prev_sample, next_sample in zip(self.q_layers, reversed(self.p_layers), q_samples,
                                                              q_samples[1:]):
            log_weights += layer_p.log_likelihood(prev_sample, next_sample) - \
                           layer_q.log_likelihood(next_sample, prev_sample)
        log_weights += self.prior.log_likelihood(q_samples[-1])
        return log_weights

    def forward(self, x: torch.Tensor, k: int, model_type: str = 'iwae'):
        rep_x = x.repeat_interleave(k, dim=0)
        q_samples = self.q_samples(rep_x)

        log_ws = self.log_weights_from_q_samples(q_samples)
        log_ws_matrix = log_ws.reshape(x.shape[0], k)
        log_ws_minus_max = log_ws_matrix - torch.max(log_ws_matrix, dim=1, keepdim=True)[0]

        ws = torch.exp(log_ws_minus_max)
        ws_normalized = ws / torch.sum(ws, dim=1, keepdim=True)
        ws_normalized_vector = ws_normalized.reshape(log_ws.shape)

        # Compute ELBO
        print(f'Training a {model_type.upper()} (k={k})')
        if model_type in ['vae', 'VAE']:
            L_q = (1.0 / k) * torch.sum(log_ws, dim=0)
        else:
            L_q = (1.0 / k) * torch.dot(ws_normalized_vector.detach(), log_ws)
        return L_q

    def log_marginal_likelihood_estimate(self, x, k):
        rep_x = x.repeat_interleave(k, dim=0)
        q_samples = self.q_samples(rep_x)

        log_ws = self.log_weights_from_q_samples(q_samples)
        log_ws_matrix = log_ws.reshape(x.shape[0], k)
        ws_matrix_max = torch.max(log_ws_matrix, dim=1, keepdim=True)[0]
        log_ws_minus_max = log_ws_matrix - ws_matrix_max

        ws = torch.exp(log_ws_minus_max)

        log_marginal_estimate = ws_matrix_max + \
                                torch.log(torch.mean(ws, dim=1, keepdim=True))
        return log_marginal_estimate

    def first_q_layer_weights_np(self):
        return self.q_layers[0].first_linear_layer_weights_np()

    def last_p_layer_weights_np(self):
        return self.p_layers[-1].last_linear_layer_weights_np()

    def first_p_layer_weights_np(self):
        return self.p_layers[0].first_linear_layer_weights_np()

    @staticmethod
    def random(latent_units, hidden_units_q, hidden_units_p, bias=None, data_type='binary', device='cpu'):
        layers_q = []
        for units_prev, units_next, hidden_units in zip(latent_units, latent_units[1:], hidden_units_q):
            layers_q.append(
                GaussianSampler.random([units_prev] + hidden_units + [units_next])
                    .to(device)
            )

        layers_p = []
        for units_prev, units_next, hidden_units in \
                zip(list(reversed(latent_units))[:-1], list(reversed(latent_units))[1:-1], hidden_units_p[:-1]):
            layers_p.append(
                GaussianSampler.random([units_prev] + hidden_units + [units_next])
                    .to(device)
            )
        if data_type == 'binary':
            layers_p.append(
                BernoulliSampler.random([latent_units[1]] + hidden_units_p[-1] + [latent_units[0]], bias=bias)
                    .to(device)
            )
        elif data_type == 'continuous':
            layers_p.append(
                GaussianSampler.random([latent_units[1]] + hidden_units_p[-1] + [latent_units[0]], bias)
                    .to(device)
            )

        prior = UnitGaussianSampler()
        return IWAEClone(layers_q, layers_p, prior)
