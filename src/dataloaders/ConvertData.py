import os
import struct
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch


def binarize_batch(x : List):
    xb = torch.vstack(x)
    xb = torch.bernoulli(xb)
    return xb

def convert_MNIST(dir, plot_samples=False):
    def load(full_path):
        with open(full_path, 'rb') as f:
            f.seek(4)
            nimages, rows, cols = struct.unpack('>iii', f.read(12))

            images = np.fromfile(f, dtype=np.dtype(np.ubyte))
            images = (images / 255.0).astype('float32').reshape((nimages, rows, cols))

        # print(images.shape)
        return images

    def store(full_path, data):
        np.save(full_path, data)

    for filename in ['train-images-idx3-ubyte', 't10k-images-idx3-ubyte']:
        full_path = os.path.join(dir, filename)
        data = load(full_path)
        store(full_path, data)

        if plot_samples:
            plt.subplots(1, 10)
            for i in range(10):
                plt.subplot(1, 10, i + 1)
                img = data[i].squeeze()
                plt.imshow(img, cmap="gray")
            plt.show()


def convert_BinaryMNIST(dir, plot_samples=False):
    def load(full_path):
        with open(full_path) as f:
            lines = f.readlines()

        data = np.array([[int(i) for i in line.split()] for line in lines])
        images = data.astype('float32')
        images = images.reshape((-1, 28, 28))

        # print(images.shape)
        return images

    def store(full_path, data):
        np.save(full_path, data)

    for filename in ['binarized_mnist_train.amat', 'binarized_mnist_valid.amat', 'binarized_mnist_test.amat']:
        full_path = os.path.join(dir, filename)
        data = load(full_path)
        store(full_path, data)

        if plot_samples:
            plt.subplots(1, 10)
            for i in range(10):
                plt.subplot(1, 10, i + 1)
                img = data[i].squeeze()
                plt.imshow(img, cmap="gray")
            plt.show()


def convert_BINARY_MNIST(*args, **kwargs):
    return convert_BinaryMNIST(*args, **kwargs)


def convert_OMNIGLOT(dir, plot_samples=False):
    def load(full_path):
        omni_raw = scipy.io.loadmat(full_path)

        def reshape_data(data):
            return data.reshape((-1, 28, 28), order='F')

        train_data = reshape_data(omni_raw['data'].T.astype('float32'))
        test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))

        # print(train_data.shape)
        # print(test_data.shape)
        return train_data, test_data

    def store(imgs_filename, data):
        np.save(imgs_filename, data)

    filename = "chardata.mat"
    full_path = os.path.join(dir, filename)
    train_data, test_data = load(full_path)

    for filename, data in zip(["train", "test"], [train_data, test_data]):
        full_path = os.path.join(dir, filename)
        store(full_path, data)

    if plot_samples:
        plt.subplots(1, 10)
        for i in range(10):
            plt.subplot(1, 10, i + 1)
            img = train_data[i].squeeze()
            plt.imshow(img, cmap="gray")
        plt.show()


if __name__ == '__main__':
    convert_MNIST(dir="data\MNIST")
    convert_BinaryMNIST(dir="data\BinaryMNIST")
    convert_OMNIGLOT(dir="data\OMNIGLOT")
