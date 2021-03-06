import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from samplers.bernoulli import BernoulliSampler
from samplers.gaussian import GaussianSampler, UnitGaussianSampler
from utils_clone.pytorch import reshape_and_tile_images


class IWAEClone(nn.Module):
    def __init__(self, q_layers, p_layers, prior):
        nn.Module.__init__(self)
        self.q_layers = nn.ModuleList(q_layers)
        self.p_layers = nn.ModuleList(p_layers)
        self.prior = prior

    @property
    def params(self):
        return self.parameters()

    # @property
    # def params(self) -> list:
    #     params = []
    #     for ql in self.q_layers:
    #         params += list(ql.parameters())
    #     for pl in self.p_layers:
    #         params += list(pl.parameters())
    #     return params + list(self.prior.parameters())

    def q_samples(self, x):
        samples = [x]
        for layer in self.q_layers:
            samples.append(layer(samples[-1]))
        return samples

    def log_weights_from_q_samples(self, q_samples):
        log_weights = torch.zeros(q_samples[-1].shape[0], device=q_samples[-1].device)
        for layer_q, layer_p, prev_sample, next_sample in zip(self.q_layers, reversed(self.p_layers),
                                                              q_samples, q_samples[1:]):
            log_weights += layer_p.log_likelihood(prev_sample.clone(), next_sample.clone()) - \
                           layer_q.log_likelihood(next_sample, prev_sample)
        log_weights += self.prior.log_likelihood(q_samples[-1])
        return log_weights

    def forward(self, x: torch.Tensor, k: int, model_type: str = 'iwae'):
        rep_x = x.repeat_interleave(k, dim=0)
        q_samples = self.q_samples(rep_x)

        log_ws = self.log_weights_from_q_samples(q_samples)
        log_ws_matrix = log_ws.reshape(x.shape[0], k)
        log_ws_minus_max = log_ws_matrix - torch.max(log_ws_matrix, dim=1, keepdim=True)[0]

        ws = torch.exp(log_ws_minus_max.clone().detach())
        ws_normalized = ws / torch.sum(ws, dim=1, keepdim=True)
        ws_normalized_vector = ws_normalized.reshape(log_ws.shape)

        # Compute ELBO
        # print(f'Training a {model_type.upper()} (k={k})')
        if model_type in ['vae', 'VAE']:
            L_q = - (1.0 / k) * torch.sum(log_ws, dim=0)
        else:
            L_q = - torch.dot(ws_normalized_vector.detach(), log_ws)
        return L_q

    def estimate_loss(self, x, k):
        return self.log_marginal_likelihood_estimate(x, k)

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
    def random(latent_units, hidden_units_q, hidden_units_p, bias=None, data_type='binary'):
        layers_q = []
        for units_prev, units_next, hidden_units in zip(latent_units, latent_units[1:], hidden_units_q):
            layers_q.append(
                GaussianSampler.random([units_prev] + hidden_units + [units_next])
            )

        layers_p = []
        for units_prev, units_next, hidden_units in \
                zip(list(reversed(latent_units))[:-1], list(reversed(latent_units))[1:-1], hidden_units_p[:-1]):
            layers_p.append(
                GaussianSampler.random([units_prev] + hidden_units + [units_next])
            )
        if data_type == 'binary':
            layers_p.append(
                BernoulliSampler.random([latent_units[1]] + hidden_units_p[-1] + [latent_units[0]], bias=bias)
            )
        elif data_type == 'continuous':
            layers_p.append(
                GaussianSampler.random([latent_units[1]] + hidden_units_p[-1] + [latent_units[0]], mean=bias)
            )

        prior = UnitGaussianSampler()
        return IWAEClone(layers_q, layers_p, prior)

    # ----------------------------------------------------------------
    #  Functions for reproducing paper results - visualizations
    # ----------------------------------------------------------------

    def get_samples(self, num_samples, device='cpu'):
        prior_samples = torch.randn(num_samples, self.first_p_layer_weights_np().shape[0], device=device)
        samples = [prior_samples]
        for layer in self.p_layers[:-1]:
            samples.append(layer.sample(samples[-1]))
        samples_function = self.p_layers[-1].get_mean(samples[-1])
        return reshape_and_tile_images(samples_function.detach().cpu().numpy())

    def get_first_q_layer_weights(self):
        return reshape_and_tile_images(self.first_q_layer_weights_np())

    def get_last_p_layer_weights(self):
        return reshape_and_tile_images(self.last_p_layer_weights_np().T)

    def measure_marginal_log_likelihood(self, dataloader: DataLoader, k: int = 50, device='cpu'):
        s = []
        N = 0
        pbar = tqdm(dataloader)
        with torch.no_grad():
            for x in pbar:
                if type(x) == list:
                    x = x[0].squeeze()
                N += x.shape[0]
                s.append(
                    self.log_marginal_likelihood_estimate(x.type(torch.get_default_dtype()).to(device), k=k)
                        .detach().cpu().numpy()
                )
                pbar.set_description(f'[mean(s)|{np.mean(s):.03f}] ')
        return s / N

    def plot_true_posterior(self, x: torch.Tensor, num_samples: int = 6):
        def get_ll_for_x_h(_x, h):
            q_samples = [_x, h]
            ll = torch.zeros(q_samples[-1].shape[0], device=q_samples[-1].device)
            for layer_q, layer_p, prev_sample, next_sample in zip(self.q_layers, reversed(self.p_layers),
                                                                  q_samples, q_samples[1:]):
                ll += layer_p.log_likelihood(prev_sample.clone(), next_sample.clone())
            return ll + self.prior.log_likelihood(h)

        h1, h2 = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))
        ll_h = np.zeros_like(h1)
        for i in range(len(h1)):
            for j in range(len(h2)):
                ll_h[i, j] = get_ll_for_x_h(x, torch.tensor([[h1[i, j], h2[i, j]]]).type(x.type()).to(x.device))

        plt.figure()
        plt.pcolormesh(h1, h2, ll_h, cmap='Greys')
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    _m = IWAEClone.random(latent_units=[28 * 28] + [2], hidden_units_q=[[50]], hidden_units_p=[[50]])
    _m.plot_true_posterior(x=torch.randn(1, 784))
