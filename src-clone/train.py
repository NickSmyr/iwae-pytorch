import math
import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
# noinspection PyProtectedMember
from torch.utils.data import DataLoader
from tqdm import tqdm

# noinspection PyUnresolvedReferences
from dataloaders.mnist import MnistDataloader, BinaryMnistDataloader
# noinspection PyUnresolvedReferences
from dataloaders.omniglot import OmniglotDataloader
from ifaces import DistributionSampler, DownloadableDataset
from iwae_clone import IWAEClone


def train(model: IWAEClone, dataloader: DataLoader, optimizer: Optimizer, k: int, scheduler: LambdaLR, n_epochs: int,
          model_type: str = 'iwae', debug: bool = False):
    # Load checkpoint
    chkpts_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
    if not os.path.exists(chkpts_dir_path) or not os.path.isdir(chkpts_dir_path):
        os.mkdir(chkpts_dir_path)
    state_fname_s = f'{dataloader.dataset.title}_k{k:02d}_L{len(model.p_layers)}_e__EPOCH__.pkl'
    start_epoch = 0
    for i in range(0, 9):
        e = 3 ** i
        state_fpath_e = os.path.join(chkpts_dir_path, state_fname_s.replace('__EPOCH__', f'{e:03d}'))
        state_fpath = os.path.join(chkpts_dir_path, state_fname_s.replace('__EPOCH__', f'{3 ** (i + 1):03d}'))
        if not os.path.exists(state_fpath) and os.path.exists(state_fpath_e):
            state_dict = torch.load(state_fpath_e, map_location=model.device)
            model.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optimizer'])
            scheduler.load_state_dict(state_dict['scheduler'])
            print(f'[CHECKPOINT] Loaded checkpoint after epoch {e}.')
            start_epoch = e + 1
            break

    # Print welcome message
    print("Training for {} epochs with {} learning rate".format(n_epochs - start_epoch,
                                                                optimizer.param_groups[0]['lr']))
    time.sleep(0.1)

    # Main training loop
    for e in range(start_epoch, n_epochs):
        ls = []
        pbar = tqdm(dataloader)
        for x in pbar:
            # Zero out discriminator gradient (before backprop)
            optimizer.zero_grad()
            # Perform forward pass
            L_k_q = model(x.to(model.device), k=k, model_type=model_type)
            ls.append(-L_k_q.item())
            pbar.set_description(f'[e|{e:03d}/{n_epochs:03d}][l|{np.mean(ls):.03f}] ')
            assert not np.isnan(np.mean(ls))
            # Perform backward pass (i.e. backprop)
            L_k_q.backward()
            # Update weights
            optimizer.step()
        # Update LR
        scheduler.step()
        # Save model checkpoint
        if e > 0 and int(math.log(e, 3)) == math.log(e, 3):
            state_fpath = os.path.join(chkpts_dir_path, state_fname_s.replace('__EPOCH__', f'{e:03d}'))
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, state_fpath)
            print('')
            print(f'[CHECKPOINT] Saved checkpoint after epoch {e}.')
            print('')
            time.sleep(0.1)
        # After each epoch preview some samples
        if e % 30 == 0:
            samples = model.get_samples(100)
            plt.imshow(samples, cmap='Greys')
            plt.axis('off')
            plt.title(f'[{model_type.upper()}] 100 samples after epoch={e:03d}')
            plt.show()

        if debug:
            break
    # Return trained model
    return model


def get_epoch_lr(epoch: int) -> float:
    if epoch <= 0:
        return 1e-3
    epoch_pow = 8
    for _ep in range(8):
        if epoch < sum(3 ** i for i in range(_ep)):
            epoch_pow = _ep
            break
    return 1e-4 * round(10. ** (1 - (epoch_pow - 1) / 7.), 1)


def update_lr(epoch: int, debug: bool = True) -> float:
    """
    :param int epoch: current epoch
    :param bool debug: set to True to have a message printed whenever the LR is updated
    :return: gamma (i.e. the multiplicative factor with which to multiply current lr)
    """
    old_lr = get_epoch_lr(epoch - 1)
    new_lr = get_epoch_lr(epoch)
    if debug and new_lr < old_lr:
        print('')
        print(f'Adjusting learning rate to {new_lr}')
        print('')
        time.sleep(0.1)
    return new_lr / old_lr


def plot_lr():
    plt.plot([_e for _e in range(3 ** 8)], [get_epoch_lr(_e) for _e in range(3 ** 8)], '-.o')
    plt.suptitle('Learning Rate Scheduling (epoch range: [0, 3^8])')
    plt.title(f'initial={get_epoch_lr(0)} | final={get_epoch_lr(3 ** 8)}')
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.show()

# TODO add second layer option
def train_and_save_checkpoints(cuda : bool,
                               k : int,
                               num_layers : int,
                               dataset : str,
                               model_type : str,
                               batch_size : int,
                               debug : bool,
                               ):
    """
    Method to train one configuration, and output a checkpoint file
    :param cuda: True for GPU acceleration
    :param k: The number of samples
    :param num_layers: The number of stochastic layers
    :param dataset: The dataset to train on. One of ["mnist", "binarymnist", "omniglot"]
    :param model_type: The model to use. One of ["vae", "iwae"]
    :param batch_size: The batch size to use
    :param debug: If True, will execute only one epoch of the training process
    """
    # Initialize training
    DistributionSampler.set_seed(seed=42)
    DownloadableDataset.set_data_directory('../data')

    _device = torch.device('cuda') if cuda else torch.device('cpu')
    _k = k
    _batch_size = batch_size
    # Stub the training process soo that we can test that the result format is usable
    _debug = debug

    if num_layers == 1:
        _latent_units = [50]
        _hidden_units_q = [[200, 200]]
        _hidden_units_p = [[200, 200]]
    elif num_layers == 2:
        _latent_units = [100, 50]
        _hidden_units_q = [[200, 200], [100, 100]]
        _hidden_units_p = [[100, 100], [200, 200]]
    else:
        raise Exception("The number of layers must be either 1 or 2")

    available_datasets = ['mnist', 'binarymnist', 'omniglot']
    if dataset == available_datasets[0]:
        _dataloader = MnistDataloader(train_not_test=True, batch_size=_batch_size, pin_memory=True, shuffle=True)
    elif dataset == available_datasets[1]:
        _dataloader = BinaryMnistDataloader(train_not_test=True, batch_size=_batch_size, pin_memory=True, shuffle=True)
    elif dataset == available_datasets[2]:
        _dataloader = OmniglotDataloader(train_not_test=True, batch_size=_batch_size, pin_memory=True, shuffle=True)

    else:
        raise Exception("Unknown dataset name ", dataset)

    _model = IWAEClone.random(latent_units=[28 * 28] + _latent_units, hidden_units_q=_hidden_units_q,
                              hidden_units_p=_hidden_units_p, data_type='binary', device=_device,
                              bias=_dataloader.dataset.get_train_bias())
    print(_dataloader.dataset.get_train_bias_np().shape)
    _optimizer = optim.Adam(params=_model.params, lr=1e-3, betas=(0.99, 0.999), eps=1e-4)
    # _optimizer = optim.SGD(params=_model.params, lr=1e-3)
    _scheduler = LambdaLR(_optimizer, lr_lambda=update_lr)
    train(model=_model, dataloader=_dataloader, optimizer=_optimizer, scheduler=_scheduler, k=_k, n_epochs=3 ** 8,
          model_type=model_type, debug=_debug)
    print('[DONE]')

    # Save the final checkpoint
    _chkpts_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
    _state_fname_s = f'{_dataloader.dataset.title}_k{_k:02d}_L{len(_model.p_layers)}' \
                     f'_bs{batch_size}_{model_type}_final.pkl'
    _state_fpath = os.path.join(_chkpts_dir_path, _state_fname_s)
    torch.save({
        'model': _model.state_dict(),
        'optimizer': _optimizer.state_dict(),
        'scheduler': _scheduler.state_dict()
    }, _state_fpath)
    print('')
    print(f'[CHECKPOINT] Saved checkpoint after training')
    print('')

if __name__ == '__main__':
   train_and_save_checkpoints(cuda=True,
                              k=50,
                              num_layers=2,
                              dataset='mnist',
                              model_type='iwae',
                              batch_size=100,
                              debug=True)
