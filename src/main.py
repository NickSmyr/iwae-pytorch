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
train_batch_size = 200

# Larger batch size for validation
validation_batch_size = 20000

# Smaller batch size for final performance testing as k could be large
final_validation_batch_size = 20

# In the IWAE paper, this parameter is 8
#epochs_exp = 8
epochs_exp = 2
beta1 = 0.9
beta2 = 0.999
adam_epsilon = 1e-4

# Set to 'AE', 'VAE' or 'IWAE'
model_type = 'VAE'

# k hyper parameter
k = 1


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

training_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=Compose([ToTensor(), Lambda(lambda x: torch.bernoulli(torch.flatten(x)))])
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=Compose([ToTensor(), Lambda(lambda x: torch.bernoulli(torch.flatten(x)))])
)

train_dataloader = DataLoader(training_data, batch_size=train_batch_size, pin_memory=True)
validation_dataloader = DataLoader(test_data, batch_size=validation_batch_size, pin_memory=True)
final_validation_dataloader = DataLoader(test_data, batch_size=final_validation_batch_size, pin_memory=True)

# Initialize bias of output unit based on means of training data
mean_of_training_data = torch.stack([d for d, l in training_data]).mean(0)
output_bias = -torch.log(1 / torch.clamp(mean_of_training_data, 0.0001, 0.9999) - 1)


if model_type == 'AE':
    model = AE()
elif model_type == 'VAE':
    # Single stochastic layer architecture from IWAE paper
    # Disable initialization of output bias for now
    model = VAE(k=k, q_dim=50, hidden_dims=[200, 200], device=device)#, output_bias=output_bias)
elif model_type == 'IWAE':
    model = IWAE()
else:
    print(f"Unknown model type: {model_type}")
    exit(1)

model = model.to(device)

losses = []

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
        loss = validate(validation_dataloader, model, device)
        losses.append(loss)
        print(f"Validation average loss: {loss:>8f} \n")


# Calculate performance
validation_k = 5000
loss = validate(final_validation_dataloader, model, device, {"k": validation_k})
print(f"Validation average loss L_{validation_k}: {loss:>8f} \n")

# Plot validation losses per epoch
plt.figure()
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Estimate of the lower bound of the log-likelihood')

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
