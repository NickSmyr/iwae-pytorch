import importlib
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
from tqdm.autonotebook import tqdm

# noinspection PyUnresolvedReferences
import dataloaders
# noinspection PyUnresolvedReferences
from dataloaders.omniglot import OmniglotDataloader
from ifaces import DistributionSampler, DownloadableDataset
from iwae_clone import IWAEClone
from modules.vae import VAE
from modules.iwae import IWAE
import modules.validate


def get_state_name(dataset_title, model_type, use_clone, k, num_layers, batch_size):
    return f'{dataset_title}_{model_type}{"-clone" if use_clone else ""}_k{k:02d}_L{num_layers}_bs{batch_size}'


def calculate_and_display_L5000(test_dataloader, model, device):
    validation_k = 5000
    loss = modules.validate.validate(test_dataloader, model, device, {"k": validation_k})
    print(f"Validation average loss L_{validation_k}: {loss:>8f} \n")


def train(model, dataloader: DataLoader, optimizer: Optimizer, k: int, scheduler: LambdaLR, n_epochs: int,
          model_type: str = 'iwae', state_name: str = '', debug: bool = False, device='cpu', chkpts_dir_path=None):
    # Load checkpoint
    if chkpts_dir_path is None:
        chkpts_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
    if not os.path.exists(chkpts_dir_path) or not os.path.isdir(chkpts_dir_path):
        os.mkdir(chkpts_dir_path)
    state_fname_s = state_name + '_e__EPOCH__.pkl'
    start_epoch = 0
    for i in range(0, 33):
        e = 100 * i
        state_fpath_e = os.path.join(chkpts_dir_path, state_fname_s.replace('__EPOCH__', f'{e:03d}'))
        state_fpath = os.path.join(chkpts_dir_path, state_fname_s.replace('__EPOCH__', f'{100 * (i + 1):03d}'))
        if not os.path.exists(state_fpath) and os.path.exists(state_fpath_e):
            state_dict = torch.load(state_fpath_e, map_location=device)
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
    L1 = 0.0
    for e in range(start_epoch, n_epochs):
        ls = []
        x = None
        pbar = tqdm(dataloader)
        for x in pbar:
            # Zero out discriminator gradient (before backprop)
            optimizer.zero_grad()
            # Perform forward pass
            if type(x) == list:
                x = x[0].squeeze()
            if isinstance(model, IWAEClone):
                L_k_q = model(torch.bernoulli(x.type(torch.get_default_dtype()).to(device)), k=k, model_type=model_type)
            else:
                L_k_q = model.objective(torch.bernoulli(x.type(torch.get_default_dtype()).to(device)))
            ls.append(-L_k_q.item())
            pbar.set_description(f'[e|{e:03d}/{n_epochs:03d}][l|{np.mean(ls):.03f}][L1|{L1:.03f}] ')
            assert not np.isnan(np.mean(ls))
            # Perform backward pass (i.e. backprop)
            L_k_q.backward()
            # Update weights
            optimizer.step()

        with torch.no_grad():
            L1 = model.estimate_loss(x=x.clone().type(torch.get_default_dtype()).to(device), k=1).mean().item()

        # Update LR
        scheduler.step()
        # Save model checkpoint
        if e % 100 == 0:
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
            with torch.no_grad():
                samples = model.get_samples(100, device=device)
            plt.imshow(samples, cmap='Greys')
            plt.axis('off')
            plt.title(f'[{model_type.upper()}] 100 samples after epoch={e:03d}')
            plt.draw()
            plt.show(block=False)
            plt.pause(0.001)

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


def train_and_save_checkpoints(seed: int,
                               cuda: bool,
                               k: int,
                               num_layers: int,
                               dataset: str,
                               model_type: str,
                               use_clone: bool,
                               batch_size: int,
                               debug: bool,
                               dtype=torch.float64,
                               chkpts_dir_path=None
                               ):
    """
    Method to train one configuration, and output a checkpoint file
    :param seed: Seed value
    :param cuda: True for GPU acceleration
    :param k: The number of samples
    :param num_layers: The number of stochastic layers
    :param dataset: The dataset to train on. One of ["mnist", "binarymnist", "omniglot"]
    :param model_type: The model to use. One of ["vae", "iwae"]
    :param use_clone: set to True to use the clone of the origianl model (aka IWAEClone); otherwise our own
                      implementation will be used
    :param batch_size: The batch size to use
    :param debug: If True, will execute only one epoch of the training process
    :param dtype: one of `torch.float64` (or `torch.double`), `torch.float32` (or `torch.float`)
    :param chkpts_dir_path: absolute path to checkpoints directory
    """
    if cuda:
        assert torch.cuda.is_available(), "No CUDA CPU available."
    # Initialize training
    DistributionSampler.set_seed(seed=seed)
    if dtype == torch.float64:
        torch.set_default_dtype(torch.float64)
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

    # Instantiate Dataloader
    dataloader_class_name = ''.join(x.title() for x in dataset.split('_')) + 'Dataloader'
    try:
        _module = importlib.import_module('dataloaders.' + dataset.split('_')[-1])
        _dataloader_class = getattr(_module, dataloader_class_name)
        _train_dataloader = _dataloader_class(train_not_test=True, batch_size=_batch_size, pin_memory=True, shuffle=True)

        # Smaller batch size for final performance testing as k could be large
        final_validation_batch_size = 20

        _test_dataloader = _dataloader_class(train_not_test=False, batch_size=final_validation_batch_size, pin_memory=True, shuffle=True)
    except AttributeError:
        raise Exception(f'Unknown dataset name {dataset}')

    # Instantiate model's Module
    if use_clone:
        _model: IWAEClone = IWAEClone.random(latent_units=[28 * 28] + _latent_units, hidden_units_q=_hidden_units_q,
                                             hidden_units_p=_hidden_units_p, bias=_train_dataloader.dataset.get_train_bias())
        _model.prior.device = _device
    else:
        if model_type == 'vae':
            _model = VAE(k=k, latent_units=[28 * 28] + _latent_units, hidden_units_q=_hidden_units_q,
                         hidden_units_p=_hidden_units_p, output_bias=_train_dataloader.dataset.get_train_bias())
        else:  # model_type == 'iwae':
            _model = IWAE(k=k, latent_units=[28 * 28] + _latent_units, hidden_units_q=_hidden_units_q,
                          hidden_units_p=_hidden_units_p, output_bias=_train_dataloader.dataset.get_train_bias())
    _model = _model.to(_device)

    # Instantiate Optimizer & LR-Scheduler
    _optimizer = optim.Adam(params=_model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-4)
    _scheduler = LambdaLR(_optimizer, lr_lambda=update_lr)

    # Construct general name for checkpoint files
    state_name = get_state_name(_train_dataloader.dataset.title, model_type, use_clone, k, num_layers, batch_size)

    # Start the training loop
    train(model=_model, dataloader=_train_dataloader, optimizer=_optimizer, scheduler=_scheduler, k=_k, n_epochs=3280,
          model_type=model_type, state_name=state_name, debug=_debug, device=_device, chkpts_dir_path=chkpts_dir_path)
    print('[DONE]')

    calculate_and_display_L5000(_test_dataloader, _model, _device)

    # Save the final checkpoint
    _state_fpath = os.path.join(chkpts_dir_path, state_name + '_final.pkl')
    torch.save({
        'model': _model.state_dict(),
        'optimizer': _optimizer.state_dict(),
        'scheduler': _scheduler.state_dict()
    }, _state_fpath)
    print('')
    print(f'[CHECKPOINT] Saved checkpoint after training')
    print('')


if __name__ == '__main__':
    DownloadableDataset.set_data_directory('../data')
    train_and_save_checkpoints(seed=42,
                               cuda=True,
                               k=1,
                               num_layers=1,
                               dataset='mnist',
                               model_type='vae',
                               use_clone=False,
                               batch_size=400,
                               debug=False,
                               dtype=torch.float32,
                               chkpts_dir_path='../checkpoints')
