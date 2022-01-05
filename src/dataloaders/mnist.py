import os

import matplotlib.pyplot as plt
import numpy as np
import torch
# noinspection PyProtectedMember
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import FashionMNIST as FashionMnistDataset
from torchvision.transforms import transforms

from ifaces import DownloadableDataset
from utils.data import unzip_gz


class MnistDataset(Dataset, DownloadableDataset):
    DTransforms = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: torch.bernoulli(x)),
        transforms.Lambda(lambda x: torch.flatten(x)),
    ])

    def __init__(self, train_not_test: bool = True):
        DownloadableDataset.__init__(self, which='mnist')
        if train_not_test:
            self.data = np.load(os.path.join(self.path, 'train-images-idx3-ubyte.npy'))
        else:
            self.data = np.load(os.path.join(self.path, 't10k-images-idx3-ubyte.npy'))
        # Instantiate Pytorch Dataset
        Dataset.__init__(self)

    def __getitem__(self, index) -> torch.Tensor:
        return MnistDataset.DTransforms(self.data[index])

    def __len__(self) -> int:
        return len(self.data)

    def download(self) -> bool:
        # Download mnist data
        #   - train
        self.clean_download(url='http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', clean_first=True)
        self.clean_download(url='http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', clean_first=True)
        #   - test
        self.clean_download(url='http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', clean_first=True)
        self.clean_download(url='http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', clean_first=True)
        # Unzip mnist data
        for filename in os.listdir(self.path):
            f = os.path.join(self.path, filename)
            unzip_gz(f)
            os.remove(f)
        return True

    @property
    def title(self) -> str:
        return 'mnist'


class BinaryMnistDataset(Dataset, DownloadableDataset):
    DTransforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x)),
    ])

    def __init__(self, train_not_test: bool = True):
        DownloadableDataset.__init__(self, which='binary_mnist')
        if train_not_test:
            self.data = np.load(os.path.join(self.path, 'binarized_mnist_train.amat.npy'))
        else:
            self.data = np.load(os.path.join(self.path, 'binarized_mnist_test.amat.npy'))
        # Instantiate Pytorch Dataset
        Dataset.__init__(self)

    def __getitem__(self, index) -> torch.Tensor:
        return BinaryMnistDataset.DTransforms(self.data[index])

    def __len__(self) -> int:
        return len(self.data)

    def download(self) -> bool:
        # Download binary mnist data
        #   - train
        self.clean_download(url='http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/' +
                                'binarized_mnist_train.amat', clean_first=False)
        self.clean_download(url='http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/' +
                                'binarized_mnist_valid.amat', clean_first=False)
        #   - test
        self.clean_download(url='http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/' +
                                'binarized_mnist_test.amat', clean_first=False)
        return True

    @property
    def title(self) -> str:
        return 'binary_mnist'


class MnistDataloader(DataLoader):
    def __init__(self, train_not_test: bool = True, **kwargs):
        self.dataset = None  # type: MnistDataset
        DataLoader.__init__(self, dataset=MnistDataset(train_not_test=train_not_test), **kwargs)


class BinaryMnistDataloader(DataLoader):
    def __init__(self, train_not_test: bool = True, **kwargs):
        self.dataset = None  # type: BinaryMnistDataset
        DataLoader.__init__(self, dataset=BinaryMnistDataset(train_not_test=train_not_test), **kwargs)


class FashionMnistDataloader(DataLoader):
    def __init__(self, train_not_test: bool = True, **kwargs):
        # Add method to compute the mean image
        def get_train_bias_np(_self) -> np.ndarray:
            data_mean = np.mean(_self.data.numpy(), axis=0)[None, :]
            return -np.log(1. / np.clip(data_mean, 0.001, 0.999) - 1.)

        def get_train_bias(_self) -> torch.Tensor:
            return torch.from_numpy(_self.get_train_bias_np().flatten())

        setattr(FashionMnistDataset, 'get_train_bias_np', get_train_bias_np)
        setattr(FashionMnistDataset, 'get_train_bias', get_train_bias)
        # Instantiate dataset
        self.dataset = FashionMnistDataset(root=DownloadableDataset.DATA_DIR, train=train_not_test, download=True,
                                           transform=MnistDataset.DTransforms)
        self.dataset.title = 'fashion_mnist'
        # Instantiate dataloader
        DataLoader.__init__(self, dataset=self.dataset, **kwargs)


if __name__ == '__main__':
    DownloadableDataset.set_data_directory('../../data')

    # MNIST
    _dl = FashionMnistDataloader(train_not_test=True, batch_size=10, pin_memory=False, shuffle=False)
    _first_batch = next(iter(_dl))[0]
    print(_first_batch.shape)
    plt.imshow(1 - _first_batch[5].reshape(28, 28), cmap='gray')
    plt.show()

    # Binary MNIST
    _dl = BinaryMnistDataloader(train_not_test=True, batch_size=10, pin_memory=False)
    _first_batch = next(iter(_dl))
    print(_first_batch.shape)
    plt.imshow(1 - _first_batch[0].reshape(28, 28), cmap='gray')
    plt.show()
