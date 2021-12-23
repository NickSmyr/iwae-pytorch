import abc
import os
import random

import numpy as np
import torch.random


class DistributionSampler(metaclass=abc.ABCMeta):

    # @property
    # @abc.abstractmethod
    # def params(self):
    #     raise NotImplementedError

    @abc.abstractmethod
    def sample(self, shape_or_x: tuple or torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def log_likelihood(self, samples: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def set_seed(seed: int) -> int:
        assert os.environ.get('RNG_SEED') is None, 'seed had already been set'
        os.environ['RNG_SEED'] = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
