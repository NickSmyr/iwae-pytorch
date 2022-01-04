import torch


def validate(dataloader, model, device):
    loss = 0
    num_data_points = 0

    with torch.no_grad():
        for X, y in dataloader:
            num_data_points += len(X)
            X = X.to(device)
            objective = model.objective(X)
            loss += objective.item()

    loss /= num_data_points

    return loss
