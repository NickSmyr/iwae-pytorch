import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Lambda

import matplotlib.pyplot as plt

from modules.ae import AE
from modules.vae import VAE
from modules.iwae import IWAE

from modules.train import train
from modules.validate import validate


# Parameters as in the IWAE paper
batch_size = 200
# In the IWAE paper, this parameter is 8
#epochs_exp = 8
epochs_exp = 2
beta1 = 0.9
beta2 = 0.999
adam_epsilon = 1e-4

# Set to 'AE', 'VAE' or 'IWAE'
model_type = 'VAE'

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

# Initialize bias of output unit based on means of training data
mean_of_training_data = torch.stack([d for d, l in training_data]).mean(0)
output_bias = -torch.log(1 / torch.clamp(mean_of_training_data, 0.0001, 0.9999) - 1)


if model_type == 'AE':
    model = AE()
elif model_type == 'VAE':
    # Single stochastic layer architecture from IWAE paper
    model = VAE(q_dim=50, hidden_dims=[200, 200], device=device, output_bias=output_bias)
elif model_type == 'IWAE':
    model = IWAE()
else:
    print(f"Unknown model type: {model_type}")
    exit(1)

model = model.to(device)

for i in range(epochs_exp):
    # Learning rate schedule as in the IWAE paper
    learning_rate = 0.001 * 10.0 ** (-i / 7.0)

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
        train(train_dataloader, model, optimizer, device)
        validate(test_dataloader, model, device)


# Visualize some results

plt.subplots(2, 10)
for i in range(10):
    X = test_data[i][0].to(device)
    pred = model(X)

    plt.subplot(2, 10, i + 1)
    img = X.detach().to('cpu').squeeze().numpy().reshape((28, 28))
    plt.imshow(img, cmap="gray")

    plt.subplot(2, 10, i + 10 + 1)
    img = pred.detach().to('cpu').squeeze().numpy().reshape((28, 28))
    plt.imshow(img, cmap="gray")

plt.show()
