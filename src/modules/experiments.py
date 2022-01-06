import importlib
import os
import matplotlib.pyplot as plt

import torch

from ifaces import DownloadableDataset
from modules.iwae import IWAE
from modules.vae import VAE


def load_checkpoint(checkpoint_fname):
    state_dict = torch.load(checkpoint_fname, map_location='cpu')
    checkpoint_fname = os.path.basename(checkpoint_fname)
    splitted_checkpint_fname = checkpoint_fname.split("_")

    dataset_name = splitted_checkpint_fname[0]
    model_type = splitted_checkpint_fname[1]
    k = int(splitted_checkpint_fname[2][1:])
    num_layers = int(splitted_checkpint_fname[3][1:])
    train_bs = int(splitted_checkpint_fname[4][2:])
    assert splitted_checkpint_fname[5].split(".")[0] == "final", "Checkpoint is not a final checkpoint"


    try:
        dataloader_class_name = ''.join(x.title() for x in dataset_name.split('_')) + 'Dataloader'

        _module = importlib.import_module('dataloaders.' + dataset_name.split('_')[-1])
        _dataloader_class = getattr(_module, dataloader_class_name)
        _train_dataloader = _dataloader_class(train_not_test=True, batch_size=train_bs, pin_memory=True,
                                              shuffle=True)

        # Smaller batch size for final performance testing as k could be large
        final_validation_batch_size = 1
        _test_dataloader = _dataloader_class(train_not_test=False, batch_size=final_validation_batch_size,
                                             pin_memory=True, shuffle=True)
    except AttributeError:
        raise Exception(f'Unknown dataset name {dataset_name}')

    if num_layers == 1:
        _latent_units = [50]
        _hidden_units_q = [[200, 200]]
        _hidden_units_p = [[200, 200]]
    elif num_layers == 2:
        _latent_units = [100, 50]
        _hidden_units_q = [[200, 200], [100, 100]]
        _hidden_units_p = [[100, 100], [200, 200]]
    else:
        raise Exception("Invalid number of layers")

    if model_type == 'vae':
        _model = VAE(k=k, latent_units=[28 * 28] + _latent_units, hidden_units_q=_hidden_units_q,
                     hidden_units_p=_hidden_units_p, output_bias=_train_dataloader.dataset.get_train_bias())
    else:  # model_type == 'iwae':
        _model = IWAE(k=k, latent_units=[28 * 28] + _latent_units, hidden_units_q=_hidden_units_q,
                      hidden_units_p=_hidden_units_p, output_bias=_train_dataloader.dataset.get_train_bias())

    _model.load_state_dict(state_dict['model'])

    return _model

def load_and_plot_samples(checkpoint_path, num_samples):
    DownloadableDataset.set_data_directory('../data')
    model: VAE = load_checkpoint(checkpoint_path)
    res = model.get_samples(num_samples)
    plt.imshow(res)
    plt.show()

if __name__ == '__main__':
    load_and_plot_samples("../../checkpoints/mnist_iwae_k01_L2_bs100_final.pkl", 100)