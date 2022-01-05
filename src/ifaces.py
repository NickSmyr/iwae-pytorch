import abc
import json
import os
import random
import sys

import numpy as np
import torch.random
#import wget

# noinspection PyPep8Naming
import dataloaders.ConvertData as converters
from utils.data import get_checksum


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
        os.environ['RNG_SEED'] = str(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


class DownloadableDataset(metaclass=abc.ABCMeta):
    DATA_DIR = None

    def __init__(self, which: str):
        self.which = which
        self.converter = getattr(converters, f'convert_{self.which.upper()}')
        self.data = None
        self.current_checksums = None

        # Download Dataset
        if not self.exists():
            print(f'Dataset not existent in path ({self.path}). Downloading now...', file=sys.stderr)
            self.download()
            self.convert()

    @property
    def path(self) -> bool:
        return os.path.join(DownloadableDataset.DATA_DIR, self.which.lower())

    def exists(self) -> bool:
        return os.path.exists(self.path) and self.validate()

    def clean_download(self, url: str, clean_first: bool = False):
        basename = os.path.basename(url)
        if clean_first and os.path.exists(os.path.join(self.path, basename)):
            os.remove(os.path.join(self.path, basename))
        if clean_first and basename.endswith('gz') \
                and os.path.exists(os.path.join(self.path, basename.replace('.gz', ''))):
            os.remove(os.path.join(self.path, basename.replace('.gz', '')))
        if clean_first and os.path.exists(os.path.join(self.path, basename + '.npy')):
            os.remove(os.path.join(self.path, basename + '.npy'))
        if not os.path.exists(os.path.join(self.path, basename)):
            wget.download(url, out=self.path)

    @abc.abstractmethod
    def download(self) -> bool:
        raise NotImplementedError

    def validate(self) -> bool:
        current_checksums = {os.path.basename(f): get_checksum(os.path.join(self.path, f))
                             for f in os.listdir(self.path) if not f.endswith('.json') and not f.endswith('.npy')}
        self.current_checksums = current_checksums
        checksum_json_path = os.path.join(self.path, 'checksums.json')

        def validate_checksums(_c1, _c2) -> bool:
            for _f, _c in _c1.items():
                if _f not in _c2.keys() or _c != _c2[_f]:
                    return False
            return True

        if not os.path.exists(checksum_json_path) or not os.path.isfile(checksum_json_path):
            return False
        with open(checksum_json_path) as json_fp:
            if not validate_checksums(current_checksums, json.load(json_fp)):
                return False
        return True

    def convert(self) -> bool:
        checksum_json_path = os.path.join(self.path, 'checksums.json')
        if not self.validate():
            self.converter(self.path)
            with open(checksum_json_path, 'w') as json_fp:
                json.dump(self.current_checksums, json_fp, indent=4)

    def get_train_bias_np(self) -> np.ndarray:
        data_mean = np.mean(self.data, axis=0)[None, :]
        return -np.log(1. / np.clip(data_mean, 0.001, 0.999) - 1.)

    def get_train_bias(self) -> torch.Tensor:
        return torch.from_numpy(self.get_train_bias_np().flatten())

    @staticmethod
    def set_data_directory(abs_path: str) -> None:
        if not os.path.exists(abs_path):
            os.mkdir(abs_path)
        if not os.path.exists(os.path.join(abs_path, 'mnist')):
            os.mkdir(os.path.join(abs_path, 'mnist'))
        if not os.path.exists(os.path.join(abs_path, 'binary_mnist')):
            os.mkdir(os.path.join(abs_path, 'binary_mnist'))
        if not os.path.exists(os.path.join(abs_path, 'omniglot')):
            os.mkdir(os.path.join(abs_path, 'omniglot'))
        DownloadableDataset.DATA_DIR = abs_path
