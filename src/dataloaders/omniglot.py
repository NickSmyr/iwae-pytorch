import os

import numpy as np
import torch
# noinspection PyProtectedMember
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from dataloaders.mnist import MnistDataset
from ifaces import DownloadableDataset


class OmniglotDataset(Dataset, DownloadableDataset):
    def __init__(self, train_not_test: bool = True):
        DownloadableDataset.__init__(self, which='omniglot')
        if train_not_test:
            self.data = np.load(os.path.join(self.path, 'train.npy'))
        else:
            self.data = np.load(os.path.join(self.path, 'test.npy'))
        # Instantiate Pytorch Dataset
        Dataset.__init__(self)

    def __getitem__(self, index) -> torch.Tensor:
        return MnistDataset.DTransforms(self.data[index])

    def __len__(self) -> int:
        return len(self.data)

    def download(self) -> bool:
        # Download omniglot data
        self.clean_download(url='https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/' +
                                'chardata.mat', clean_first=True)
        return True

    @property
    def title(self) -> str:
        return 'omniglot'


class OmniglotDataloader(DataLoader):
    def __init__(self, train_not_test: bool = True, **kwargs):
        self.dataset = None  # type: OmniglotDataset
        DataLoader.__init__(self, dataset=OmniglotDataset(train_not_test=train_not_test), **kwargs)


if __name__ == '__main__':
    DownloadableDataset.set_data_directory('../../data')
    _dl = OmniglotDataloader(train_not_test=True, batch_size=100, pin_memory=False)
    _first_batch = next(iter(_dl))
    print(_first_batch.shape)
    plt.imshow(1 - _first_batch[5].reshape(28, 28), cmap='gray')
    plt.show()
