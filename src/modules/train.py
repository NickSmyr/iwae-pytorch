def train(dataloader, model, optimizer, device):

    size = len(dataloader.dataset)

    for batch, X in enumerate(dataloader):
        # Move data to GPU
        X = X.to(device)

        # Compute objective function
        objective = model.objective(X)

        # Backpropagation
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = objective.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
