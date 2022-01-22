import os
import re

import torch

def load_hparams_from_checkpoint_name(name):
    """
    Loads hparams from a checkpoint file name in the format (inside parenthesis are variables)
    (dsname)_k(k)_L(nlayers)_bs(batch_size)_(model_type)_(epoch_number).pkl
    """
    parts = name.split("_")
    return {
        "dataset_name" : parts[0],
        "k" : int(parts[1][1:]),
        "num_layers" : int(parts[2][1:]),
        "batch_size" : int(parts[3][2:]),
        "model_type" : parts[4],
        "epoch" : int(parts[5].split(".")[0])
    }

# All parameters apart from epoch
def get_latest_checkpoint(dataset_name, k, num_layers, batch_size, model_type, checkpoint_dir):
    pass

def load_model_optimizer_scheduler(checkpoint_filename):
    model,optimizer,scheduler = torch.load(checkpoint_filename)
    return model, optimizer, scheduler

def save_model_optimizer_scheduler_hparams(model, optimizer, scheduler,
                                       dataset_name, batch_size, model_type,
                                       num_layers, k, checkpoint_path):
    """
    Method to save a model and possibly an optimizer and a scheduler. This method can also only save
    the model if an optimizer and scheduler is not needed
    :param model: The model (iwae/vae) to save (must not be None)
    """
    # Save the final checkpoint
    _chkpts_dir_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(__file__))), 'checkpoints')
    _state_fname_s = f'{dataset_name}_k{k}_L{num_layers}' \
                     f'_bs{batch_size}_{model_type}_final.pkl'
    #_state_fpath = os.path.join(_chkpts_dir_path, _state_fname_s)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None
    }, os.path.join(checkpoint_path,_state_fname_s))
