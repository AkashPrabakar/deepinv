r"""
Self-supervised denoising with the CURE loss (Rician noise).
====================================================================================================

This example shows how to train a denoiser network in a fully self-supervised way
using the CURE loss for Rician / chi-square noise. No clean reference images are
required during training.

**Noise model.** Given a clean image :math:`x`, the Rician-noisy magnitude observation is

.. math::

    z = \sqrt{(x + \sigma\varepsilon_1)^2 + (\sigma\varepsilon_2)^2},
    \quad \varepsilon_1, \varepsilon_2 \sim \mathcal{N}(0, I).

The network operates in the **squared domain** :math:`y = z^2`, which satisfies
:math:`\mathbb{E}[y] = x^2 + 2\sigma^2`. The network estimates :math:`x^2`;
the final reconstruction is :math:`\hat{x} = \sqrt{f(y)^+}`.

**CURE loss** (Chi-Square Unbiased Risk Estimate):

.. math::

    \mathrm{CURE} = \frac{1}{N}\|f(y) - (y - 2\sigma^2)\|^2
    - \frac{4\sigma^2}{N}\mathbf{1}^\top v
    + \frac{8\sigma^2}{N}\, v^\top \mathrm{diag}(J_f),
    \quad v = y - \sigma^2.

:math:`\mathbb{E}[\mathrm{CURE}] = \mathbb{E}[\mathrm{MSE}]` under chi-square noise.

"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import deepinv as dinv
from deepinv.utils import get_cache_home

# %%
# Setup paths and device.
# ---------------------------------------------------------------

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"
CKPT_DIR = BASE_DIR / "ckpts"
ORIGINAL_DATA_DIR = get_cache_home() / "datasets" / "MNIST"

torch.manual_seed(0)

device = dinv.utils.get_device()

# Non-blocking transfers cause NaN on MPS (pin_memory unsupported on Apple Silicon).
non_blocking = device.type != "mps"

# %%
# Define squared-Rician physics
# ---------------------------------------------------------------
# CURE operates in the :math:`y = z^2` domain. We compose Rician noise with a
# squaring step so the dataset stores :math:`y = z^2` directly as the measurement.

sigma = 0.05


class RicianSquaredNoise(dinv.physics.noise.NoiseModel):
    r"""Squared Rician noise: :math:`y = z^2` where :math:`z \sim \text{Rician}(x, \sigma)`."""

    def __init__(self, sigma: float):
        super().__init__()
        self._rician = dinv.physics.RicianNoise(sigma)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._rician(x, **kwargs) ** 2


physics = dinv.physics.Denoising(RicianSquaredNoise(sigma))

# %%
# Load base image datasets
# ---------------------------------------------------------------

operation = "denoising"
train_dataset_name = "MNIST"

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    root=ORIGINAL_DATA_DIR, train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root=ORIGINAL_DATA_DIR, train=False, transform=transform, download=True
)

# %%
# Generate a dataset of squared-Rician noisy images
# ---------------------------------------------------------------
#
# .. note::
#
#       We use a subset of the whole training set to reduce the computational load of the example.
#       We recommend using the whole set by setting ``n_images_max=None`` to get the best results.

num_workers = 4 if torch.cuda.is_available() else 0

n_images_max = (
    100 if torch.cuda.is_available() else 5
)

measurement_dir = DATA_DIR / train_dataset_name / operation
deepinv_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=physics,
    device=device,
    save_dir=measurement_dir,
    train_datapoints=n_images_max,
    test_datapoints=n_images_max,
    num_workers=num_workers,
    dataset_filename="demo_cure",
)

train_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=True)
test_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False)

# %%
# Set up the denoiser network
# ---------------------------------------------------------------
# The network maps :math:`y = z^2 \to \hat{x}^2`. The final estimate of :math:`x`
# is recovered as :math:`\hat{x} = \sqrt{\hat{x}^2_+}` at evaluation time.

model = dinv.models.ArtifactRemoval(
    dinv.models.UNet(in_channels=1, out_channels=1, scales=2).to(device)
)

# %%
# Set up the training parameters
# ---------------------------------------------------------------
# We use :class:`deepinv.loss.CURELoss` as the self-supervised training loss.
# No clean images are used during training.

epochs = 10
learning_rate = 5e-4
batch_size = 32 if torch.cuda.is_available() else 1

loss = dinv.loss.CURELoss(sigma=sigma, method="mc_1side", M=10)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

# %%
# Train the network
# ---------------------------------------------------------------
# Training uses only the noisy squared observations :math:`y = z^2`; the CURE loss
# provides an unbiased estimate of the MSE without any ground-truth images.

verbose = True

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)

trainer = dinv.Trainer(
    model=model,
    physics=physics,
    epochs=epochs,
    scheduler=scheduler,
    losses=loss,
    optimizer=optimizer,
    device=device,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    compute_eval_losses=True,
    early_stop_on_losses=True,
    metrics=None,  # no supervised metrics during self-supervised training
    early_stop=2,
    plot_images=False,
    save_path=str(CKPT_DIR / operation),
    verbose=verbose,
    show_progress_bar=False,
    non_blocking_transfers=non_blocking,
)

model = trainer.train()

# %%
# Evaluate PSNR
# ---------------------------------------------------------------
# The model outputs :math:`\hat{x}^2`; we recover :math:`\hat{x} = \sqrt{\hat{x}^2_+}`
# before computing PSNR against the clean image :math:`x`.

model.eval()
psnr_net, psnr_noisy = [], []

for x, y in test_dataloader:
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        x_sq_hat = model(y, physics)                  # estimate of x²
        x_hat = x_sq_hat.clamp(0).sqrt()              # estimate of x
        z = y.clamp(0).sqrt()                          # Rician magnitude (noisy input)
    psnr_net.append(dinv.metric.PSNR()(x_hat, x).mean().item())
    psnr_noisy.append(dinv.metric.PSNR()(z, x).mean().item())

print(f"\nTest results:")
print(f"PSNR no learning: {np.mean(psnr_noisy):.3f} +- {np.std(psnr_noisy):.3f}")
print(f"PSNR (CURE):      {np.mean(psnr_net):.3f} +- {np.std(psnr_net):.3f}")
