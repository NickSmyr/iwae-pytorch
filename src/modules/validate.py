import torch


def validate(dataloader, model, device):
    loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            objective = model.objective(X)
            loss += objective.item()

    num_batches = len(dataloader)
    loss /= num_batches
    print(f"Validation average loss: {loss:>8f} \n")
