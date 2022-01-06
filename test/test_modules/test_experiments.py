import unittest

import matplotlib.pyplot as plt

from ifaces import DownloadableDataset
from modules.experiments import load_checkpoint
from modules.vae import VAE


class TestExperiments(unittest.TestCase):

    def test_loading(self):
        DownloadableDataset.set_data_directory('../data')
        model : VAE = load_checkpoint("../../checkpoints/mnist_iwae_k01_L2_bs100_final.pkl")
        res = model.get_samples(10)


if __name__ == '__main__':
    unittest.main()
