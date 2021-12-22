import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Lambda

import matplotlib.pyplot as plt

from modules.ae import AE
#from src.modules.vae import VAE
#from src.modules.iwae import IWAE

from modules.train import train
from modules.validate import validate


# Parameters as in the IWAE paper
batch_size = 20
# In the IWAE paper, this parameter is 8
#epochs_exp = 8
epochs_exp = 2
beta1 = 0.9
beta2 = 0.999
adam_epsilon = 1e-4

# Set to 'AE', 'VAE' or 'IWAE' (the latter not implemented yet)
model_type = 'AE'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

training_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=Compose([ToTensor(), Lambda(lambda x: torch.flatten(x))])
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=Compose([ToTensor(), Lambda(lambda x: torch.flatten(x))])
)

train_dataloader = DataLoader(training_data, batch_size=batch_size, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)

if model_type == 'AE':
    model = AE()
# else if model_type == 'VAE':
#     model = VAE()
# else if model_type == 'IWAE':
#     model = IWAE()
else:
    print(f"Unknown model type: {model_type}")
    exit(1)

model = model.to(device)

loss_fn = torch.nn.MSELoss()

for i in range(epochs_exp):
    # Learning rate schedule as in the IWAE paper
    learning_rate = 0.001 * 10 ** (-i / 7.0)

    print(f"Epoch exponent {i}/{epochs_exp - 1}, learning rate: {learning_rate:.2e}\n----------------")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=adam_epsilon
    )

    num_epochs = 3 ** i

    for t in range(num_epochs):
        print(f"Epoch exponent {i}/{epochs_exp - 1}, epoch {t + 1}/{num_epochs}\n----------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        validate(test_dataloader, model, loss_fn, device)


# Transfer model back to cpu and visualize some results
model = model.to('cpu')

plt.subplots(2, 10)
for i in range(10):
    X = test_data[i][0]
    pred = model(X)

    plt.subplot(2, 10, i + 1)
    img = X.detach().squeeze().numpy().reshape((28, 28))
    plt.imshow(img, cmap="gray")

    plt.subplot(2, 10, i + 10 + 1)
    img = pred.detach().squeeze().numpy().reshape((28, 28))
    plt.imshow(img, cmap="gray")

plt.show()
