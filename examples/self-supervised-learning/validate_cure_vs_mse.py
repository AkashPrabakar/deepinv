r"""
Supervised MSE baseline for Rician denoising on the Shepp-Logan phantom.
====================================================================================================

This script trains a denoiser with the supervised MSE loss in the :math:`y = z^2` domain,
serving as a baseline to validate the CURE demo. If CURE is correct, a model trained
with CURELoss (no clean images) should converge to the same PSNR as this MSE baseline.

The network estimates :math:`\hat{x}^2` from :math:`y = z^2`; the final reconstruction
is :math:`\hat{x} = \sqrt{\hat{x}^2_+}`.

"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import deepinv as dinv

# %%
# Setup paths and device.
# ---------------------------------------------------------------

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"
CKPT_DIR = BASE_DIR / "ckpts"

torch.manual_seed(0)

device = dinv.utils.get_device()

# Non-blocking transfers cause NaN on MPS (pin_memory unsupported on Apple Silicon).
non_blocking = device.type != "mps"

# %%
# Define squared-Rician physics (same as CURE demo)
# ---------------------------------------------------------------

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
# Load Shepp-Logan phantom dataset
# ---------------------------------------------------------------
# SheppLoganDataset(n_data=N) returns one item with N channels stacked, so we
# extract the single phantom and tile it into a proper per-image dataset.

phantom = dinv.utils.SheppLoganDataset(size=64, n_data=1)[0][:1]  # (1, 64, 64)
train_dataset = torch.utils.data.TensorDataset(phantom.unsqueeze(0))  # single image
test_dataset  = torch.utils.data.TensorDataset(phantom.unsqueeze(0))

# %%
# Generate noisy dataset
# ---------------------------------------------------------------

operation = "denoising"
num_workers = 4 if torch.cuda.is_available() else 0

measurement_dir = DATA_DIR / "SheppLogan" / operation
deepinv_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=physics,
    device=device,
    save_dir=measurement_dir,
    train_datapoints=1,
    test_datapoints=1,
    num_workers=0,  # TensorDataset not pickleable with multiprocessing
    dataset_filename="demo_mse_baseline",
)

train_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=True)
test_dataset  = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False)

# %%
# Set up the denoiser network
# ---------------------------------------------------------------

model = dinv.models.ArtifactRemoval(
    dinv.models.UNet(in_channels=1, out_channels=1, scales=2).to(device)
)

# %%
# Set up training with supervised MSE loss in x² domain
# ---------------------------------------------------------------
# The model estimates :math:`x^2` from :math:`y = z^2`, so the supervised loss is
# :math:`\|f(y) - x^2\|^2`. SupLoss compares against the stored clean image ``x``,
# so we wrap it to square the ground truth before comparing.

class SquaredMSELoss(dinv.loss.Loss):
    """Supervised MSE in the x² domain: ||f(y) - x²||²."""

    def forward(self, x_net, x, **kwargs):
        return ((x_net - x ** 2) ** 2).reshape(x.size(0), -1).mean(dim=1)


epochs = 50
learning_rate = 5e-4
batch_size = 32 if torch.cuda.is_available() else 2

loss = SquaredMSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

# %%
# Train the network
# ---------------------------------------------------------------

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
    metrics=None,
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
# The model outputs :math:`\hat{x}^2`; convert back to image domain for PSNR.

model.eval()
psnr_net, psnr_noisy = [], []

for x, y in test_dataloader:
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        x_sq_hat = model(y, physics)
        x_hat    = x_sq_hat.clamp(0).sqrt()
        z        = y.clamp(0).sqrt()
    psnr_net.append(dinv.metric.PSNR()(x_hat, x).mean().item())
    psnr_noisy.append(dinv.metric.PSNR()(z, x).mean().item())

print(f"\nTest results (MSE supervised baseline):")
print(f"PSNR no learning: {np.mean(psnr_noisy):.3f} +- {np.std(psnr_noisy):.3f}")
print(f"PSNR (MSE):       {np.mean(psnr_net):.3f} +- {np.std(psnr_net):.3f}")
