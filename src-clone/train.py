from torch.optim import Optimizer
# noinspection PyProtectedMember
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ifaces import DistributionSampler
from iwae import IWAEClone


def train(model: IWAEClone, dataset: Dataset, optimizer: Optimizer, k: int, batch_size: int, n_epochs: int,
          model_type: str = 'iwae', seed: int = 42):
    # Set seed
    DistributionSampler.set_seed(seed=seed)
    # Start/Continue training loop
    print("training for {} epochs with {} learning rate".format(n_epochs, optimizer.param_groups[0]['lr']))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for x in tqdm(dataloader):
        # Zero out discriminator gradient (before backprop)
        optimizer.zero_grad()
        # Perform forward pass
        L_k_q = model(x, k=k, model_type=model_type)
        # Perform backward pass (i.e. backprop)
        L_k_q.backward(retain_graph=True)
        # Update weights
        optimizer.step()
    # Return trained model
    return model
