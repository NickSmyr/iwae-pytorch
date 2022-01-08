import importlib
import os
import matplotlib.pyplot as plt

import torch
import numpy as np

import dataloaders.mnist
from ifaces import DownloadableDataset
from modules.iwae import IWAE
from modules.vae import VAE


def load_checkpoint(checkpoint_fname):
    state_dict = torch.load(checkpoint_fname, map_location='cpu')
    checkpoint_fname = os.path.basename(checkpoint_fname)
    splitted_checkpint_fname = checkpoint_fname.split("_")

    dataset_name = splitted_checkpint_fname[0]
    model_type = splitted_checkpint_fname[1]

    two_dim_latent_space = model_type[-2:] == '2d'
    if two_dim_latent_space: model_type = model_type[:-2]

    k = int(splitted_checkpint_fname[2][1:])
    num_layers = int(splitted_checkpint_fname[3][1:])
    train_bs = int(splitted_checkpint_fname[4][2:])
    assert splitted_checkpint_fname[5].split(".")[0] == "final", "Checkpoint is not a final checkpoint"

    try:
        dataloader_class_name = ''.join(x.title() for x in dataset_name.split('_')) + 'Dataloader'

        _module = importlib.import_module('dataloaders.' + dataset_name.split('_')[-1])
        _dataloader_class = getattr(_module, dataloader_class_name)
        _train_dataloader = _dataloader_class(train_not_test=True, batch_size=train_bs, pin_memory=True,
                                              shuffle=True)

        # Smaller batch size for final performance testing as k could be large
        final_validation_batch_size = 1
        _test_dataloader = _dataloader_class(train_not_test=False, batch_size=final_validation_batch_size,
                                             pin_memory=True, shuffle=True)
    except AttributeError:
        raise Exception(f'Unknown dataset name {dataset_name}')

    if num_layers == 1:
        _latent_units = [2 if two_dim_latent_space else 50]
        _hidden_units_q = [[200, 200]]
        _hidden_units_p = [[200, 200]]
    elif num_layers == 2:
        _latent_units = [100, 2 if two_dim_latent_space else 50]
        _hidden_units_q = [[200, 200], [100, 100]]
        _hidden_units_p = [[100, 100], [200, 200]]
    else:
        raise Exception("Invalid number of layers")

    if model_type == 'vae':
        _model = VAE(k=k, latent_units=[28 * 28] + _latent_units, hidden_units_q=_hidden_units_q,
                     hidden_units_p=_hidden_units_p, output_bias=_train_dataloader.dataset.get_train_bias())
    else:  # model_type == 'iwae':
        _model = IWAE(k=k, latent_units=[28 * 28] + _latent_units, hidden_units_q=_hidden_units_q,
                      hidden_units_p=_hidden_units_p, output_bias=_train_dataloader.dataset.get_train_bias())

    _model.load_state_dict(state_dict['model'])

    return _model


def load_and_plot_samples(checkpoint_path, num_samples):
    DownloadableDataset.set_data_directory('../data')
    model: VAE = load_checkpoint(checkpoint_path)
    res = model.get_samples(num_samples)
    plt.imshow(res)
    plt.show()


def calc_log_p(model, x_sample, minx, maxx, miny, maxy, num_points):
    x = np.linspace(minx, maxx, num_points)
    y = np.linspace(miny, maxy, num_points)

    xg, yg = np.meshgrid(x, y)

    xg1d = xg.reshape((-1, 1))
    yg1d = yg.reshape((-1, 1))

    log_p = np.empty((num_points, num_points))
    for i in range(num_points):
        h_samples_np = np.vstack((xg[i], yg[i])).T

        with torch.no_grad():
            h_samples = torch.from_numpy(h_samples_np).type(torch.get_default_dtype())
            log_p_partial = model.calc_log_p(x_sample, h_samples).detach().numpy()
            log_p[i, :] = log_p_partial

    return x, y, log_p


def true_posterior_find_box(model, x_sample, minx, maxx, miny, maxy, num_points, nll_range):
    x, y, log_p = calc_log_p(model, x_sample, minx, maxx, miny, maxy, num_points)

    max_nll = np.max(log_p)
    max_index = np.unravel_index(np.argmax(log_p), log_p.shape)

    # Disable code for finding the size of the interesting area in the latent space

    # min_nll = max_nll - nll_range

    # indices = np.where(log_p > min_nll)

    # minxi = np.min(indices[1])
    # maxxi = np.max(indices[1])
    # minyi = np.min(indices[0])
    # maxyi = np.max(indices[0])

    # minx = x[minxi]
    # maxx = x[maxxi]
    # miny = y[minyi]
    # maxy = y[maxyi]

    # width = maxx - minx
    # height = maxy - miny

    # centerx = minx + 0.5 * width
    # centery = miny + 0.5 * height

    # size = 1.5 * max(width, height)

    # size = min(2.0, size)

    # Instead, always plot a square box of the same size in latent space
    size = 1.0

    centerx = x[max_index[1]]
    centery = y[max_index[0]]

    minx = centerx - 0.5 * size
    maxx = centerx + 0.5 * size
    miny = centery - 0.5 * size
    maxy = centery + 0.5 * size

    print(centerx, centery, size, minx, maxx, miny, maxy)

    return minx, maxx, miny, maxy


def generate_true_posterior_image(model, x_sample):
    num_points = 300

    search_minx = -5.0
    search_maxx = 5.0
    search_miny = -5.0
    search_maxy = 5.0

    nll_range = 10

    minx, maxx, miny, maxy = true_posterior_find_box(model, x_sample, search_minx, search_maxx, search_miny,
                                                     search_maxy, num_points, nll_range)

    x, y, log_p = calc_log_p(model, x_sample, minx, maxx, miny, maxy, num_points)

    max_nll = np.max(log_p)

    adjusted_log_p = log_p - max_nll

    # The structure of the posterior is easier to see with a smaller base

    # image = np.exp(adjusted_log_p)
    image = np.power(1.02, adjusted_log_p)

    return image, minx, maxx, miny, maxy


def generate_and_plot_posterior(model, x_sample):
    image, minx, maxx, miny, maxy = generate_true_posterior_image(model, x_sample)
    plt.imshow(-image, extent=[minx, maxx, miny, maxy], origin='lower', cmap="gray")


def generate_and_plot_reconstructed_samples(model, x_sample, return_mean: bool = False):
    img = np.zeros((0, 28))

    for j in range(5):
        pred = model(x_sample, return_mean=return_mean)
        pred_img = pred.detach().squeeze().numpy().reshape((28, 28))
        img = np.vstack((img, pred_img))

    plt.xticks([])
    plt.yticks([])
    plt.imshow(-img, cmap="gray")


def plot_true_posteriors_and_reconstructed_samples(checkpoint1_path, checkpoint2_path, figure_path,
                                                   return_mean: bool = False):
    test_dataloader = dataloaders.mnist.MnistDataloader(
        train_not_test=False,
        batch_size=400,
        pin_memory=True,
        shuffle=False)

    model1: VAE = load_checkpoint(checkpoint1_path)
    model2: IWAE = load_checkpoint(checkpoint2_path)

    x_samples_indices = [3, 2, 1, 32, 4, 8]
    x_samples_ = [test_dataloader.dataset[i] for i in x_samples_indices]
    x_samples = [torch.bernoulli(x_sample_) for x_sample_ in x_samples_]

    num_samples = len(x_samples)

    plt.subplots(num_samples, 5, figsize=(10, 15))

    plt.tight_layout()

    for i, x_sample in enumerate(x_samples):
        # Plot x sample
        plt.subplot(num_samples, 5, i * 5 + 1)
        if return_mean:
            img = x_samples_[i]
        else:
            img = x_sample
        img = img.detach().squeeze().numpy().reshape((28, 28))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(-img, cmap="gray")

        # Model 1
        plt.subplot(num_samples, 5, i * 5 + 2)
        generate_and_plot_posterior(model1, x_sample)

        plt.subplot(num_samples, 5, i * 5 + 3)
        generate_and_plot_reconstructed_samples(model1, x_sample, return_mean=return_mean)

        # Model 2
        plt.subplot(num_samples, 5, i * 5 + 4)
        generate_and_plot_posterior(model2, x_sample)

        plt.subplot(num_samples, 5, i * 5 + 5)
        generate_and_plot_reconstructed_samples(model2, x_sample, return_mean=return_mean)

    plt.savefig(figure_path)


def plot_lerp_samples(checkpoint1_path, checkpoint2_path, figure_path, n_img: int = 20):
    test_dataloader = dataloaders.mnist.MnistDataloader(
        train_not_test=False,
        batch_size=400,
        pin_memory=True,
        shuffle=False)

    model1: VAE = load_checkpoint(checkpoint1_path)
    model2: IWAE = load_checkpoint(checkpoint2_path)

    # Find boundaries
    # for i, batch in enumerate(test_dataloader):
    #     h = model1.encoder(batch)
    #     print(h.min(dim=0).values, h.max(dim=0).values)
    #     if i > 5:
    #         break
    with torch.no_grad():
        # For VAE
        h1_bounds = -3.5, 3.5
        h2_bounds = -3.5, 3.0
        h1, h2 = np.meshgrid(np.linspace(h1_bounds[0], h1_bounds[1], n_img),
                             np.linspace(h2_bounds[0], h2_bounds[1], n_img))
        plt.subplots(n_img, n_img, figsize=(10, 10))
        # plt.tight_layout()
        for i in range(n_img):
            for j in range(n_img):
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])

                plt.subplot(n_img, n_img, i * n_img + j + 1)
                h = torch.tensor([[h1[i, j], h2[i, j]]]).type(torch.get_default_dtype())
                x = model1.decoder_layers[-1].calc_mean(h).squeeze().cpu().numpy()
                plt.imshow(-x.reshape(28, 28), cmap="gray")
        plt.savefig(figure_path.replace(os.path.basename(figure_path), f'vae_{os.path.basename(figure_path)}'))
        plt.show()

        # For IWAE
        h1_bounds = -8.0, 5.0
        h2_bounds = -3.5, 6.5
        h1, h2 = np.meshgrid(np.linspace(h1_bounds[0], h1_bounds[1], n_img),
                             np.linspace(h2_bounds[0], h2_bounds[1], n_img))
        plt.subplots(n_img, n_img, figsize=(10, 10))
        # plt.tight_layout()
        for i in range(n_img):
            for j in range(n_img):
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])

                plt.subplot(n_img, n_img, i * n_img + j + 1)
                h = torch.tensor([[h1[i, j], h2[i, j]]]).type(torch.get_default_dtype())
                x = model2.decoder_layers[-1].calc_mean(h).squeeze().cpu().numpy()
                plt.imshow(-x.reshape(28, 28), cmap="gray")
        plt.savefig(figure_path.replace(os.path.basename(figure_path), f'iwae_{os.path.basename(figure_path)}'))
        plt.show()


def plot_100_samples(*args, figure_path):
    test_dataloader = dataloaders.mnist.MnistDataloader(
        train_not_test=False,
        batch_size=400,
        pin_memory=True,
        shuffle=False)

    model1: VAE = load_checkpoint(checkpoint1_path)
    model2: IWAE = load_checkpoint(checkpoint2_path)

    # Find boundaries
    # for i, batch in enumerate(test_dataloader):
    #     h = model1.encoder(batch)
    #     print(h.min(dim=0).values, h.max(dim=0).values)
    #     if i > 5:
    #         break
    with torch.no_grad():
        pass


if __name__ == '__main__':
    DownloadableDataset.set_data_directory('../../data')

    # load_and_plot_samples("../../checkpoints/mnist_iwae_k01_L2_bs100_final.pkl", 100)

    # plot_true_posteriors_and_reconstructed_samples(
    #     "../../checkpoints/mnist_vae2d_k01_L1_bs400_final.pkl",
    #     "../../checkpoints/mnist_iwae2d_k50_L1_bs400_final.pkl",
    #     "../../figures/posteriors.pdf",
    #     return_mean=True)

    plot_lerp_samples(
        "../../checkpoints/mnist_vae2d_k01_L1_bs400_final.pkl",
        "../../checkpoints/mnist_iwae2d_k50_L1_bs400_final.pkl",
        "../../figures/lerp.pdf",
        n_img=20)

    # plot_100_samples(
    #     "../../checkpoints/mnist_vae2d_k01_L1_bs400_final.pkl",
    #     "../../checkpoints/mnist_iwae2d_k50_L1_bs400_final.pkl",
    # )
