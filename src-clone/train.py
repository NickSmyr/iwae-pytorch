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
          model_type: str = 'iwae', debug : bool = False):
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


if __name__ == '__main__':
    # Initialize training
    DistributionSampler.set_seed(seed=42)
    DownloadableDataset.set_data_directory('../data')

    # Plot Learning Rate Scheduling
    plot_lr()

    _device = 'cuda'
    _latent_units = [50]
    _hidden_units_q = [[200, 200]]
    _hidden_units_p = [[200, 200]]
    _k = 50
    _batch_size = 100
    # Stub the training process soo that we can test that the result format is usable
    debug=True
    # _dataloader = MnistDataloader(train_not_test=True, batch_size=_batch_size, pin_memory=True, shuffle=True)
    #_dataloader = BinaryMnistDataloader(train_not_test=True, batch_size=_batch_size, pin_memory=True, shuffle=True)
    _dataloader = OmniglotDataloader(train_not_test=True, batch_size=_batch_size, pin_memory=True, shuffle=True)
    _model = IWAEClone.random(latent_units=[28 * 28] + _latent_units, hidden_units_q=_hidden_units_q,
                              hidden_units_p=_hidden_units_p, data_type='binary', device=_device,
                              bias=_dataloader.dataset.get_train_bias())
    print(_dataloader.dataset.get_train_bias_np().shape)
    _optimizer = optim.Adam(params=_model.params, lr=1e-3, betas=(0.99, 0.999), eps=1e-4)
    # _optimizer = optim.SGD(params=_model.params, lr=1e-3)
    _scheduler = LambdaLR(_optimizer, lr_lambda=update_lr)
    train(model=_model, dataloader=_dataloader, optimizer=_optimizer, scheduler=_scheduler, k=_k, n_epochs=3 ** 8,
          model_type='iwae', debug=debug)
    print('[DONE]')

    # Save the final checkpoint
    chkpts_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
    state_fname_s = f'{_dataloader.dataset.title}_k{_k:02d}_L{len(_model.p_layers)}_e__EPOCH__.pkl'

    #state_fpath_e = os.path.join(chkpts_dir_path, state_fname_s.replace('__EPOCH__', f'{e:03d}'))
    state_fpath = os.path.join(chkpts_dir_path, state_fname_s.replace('__EPOCH__', f'{"final"}'))
    torch.save({
        'model': _model.state_dict(),
        'optimizer': _optimizer.state_dict(),
        'scheduler': _scheduler.state_dict()
    }, state_fpath)
    print('')
    print(f'[CHECKPOINT] Saved checkpoint after training')
    print('')