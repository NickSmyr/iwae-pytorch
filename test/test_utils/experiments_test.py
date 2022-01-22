import unittest
import pickle

import pytest
import torch

from dataloaders.omniglot import OmniglotDataloader
from ifaces import DistributionSampler, DownloadableDataset
from iwae_clone import IWAEClone
from utils.data_download import download_data
class MyTestCase(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        # Set up and tear down for pytest
        # Initialize training
        DistributionSampler.set_seed(seed=42)
        DownloadableDataset.set_data_directory('../data')
        yield

    def test_experiments(self):
        checkpoint_path = "../../checkpoints/omniglot_k50_L1_efinal.pkl"
        # TODO
        return
        #with open(checkpoint_path, 'rb') as f:
            #obj = pickle.load(f)
        # TODO couldn't this configuration be stored inside the model?

        _batch_size = 100
        # _dataloader = BinaryMnistDataloader(train_not_test=True, batch_size=_batch_size, pin_memory=True, shuffle=True)
        _dataloader = OmniglotDataloader(train_not_test=True, batch_size=_batch_size, pin_memory=True, shuffle=True)

        _device = 'cuda'
        _latent_units = [50]
        _hidden_units_q = [[200, 200]]
        _hidden_units_p = [[200, 200]]
        _k = 50
        with open(checkpoint_path, 'rb') as f:
            obj = torch.load(f)
        print(obj)
        _model = IWAEClone.random(latent_units=[28 * 28] + _latent_units, hidden_units_q=_hidden_units_q,
                         hidden_units_p=_hidden_units_p, data_type='binary', device=_device,
                         bias=_dataloader.dataset.get_train_bias())

        _model.load_state_dict(obj['model'])
        print(_model)

if __name__ == '__main__':
    unittest.main()
