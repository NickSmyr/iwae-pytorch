import torch
from tqdm.autonotebook import tqdm

from dataloaders.mnist import BinaryMnistDataloader


def validate(dataloader, model, device, loss_parameters={}):
    total_loss = 0
    num_data_points = 0

    with torch.no_grad():
        for X in tqdm(dataloader):
            num_data_points += len(X)
            X = X.type(torch.get_default_dtype()).to(device)
            if type(dataloader) != BinaryMnistDataloader:
                X = torch.bernoulli(X)
            loss = torch.sum(model.estimate_loss(X, **loss_parameters))
            total_loss += loss.item()

    total_loss /= num_data_points

    return total_loss
