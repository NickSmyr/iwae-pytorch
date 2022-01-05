import torch

import modules.vae as vae


class IWAE(vae.VAE):
    """
    IWAE Class:
    This class implements the IWAE model architecture, inheriting from VAE.
    """

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
