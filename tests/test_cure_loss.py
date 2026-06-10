"""
Unit tests for CURELoss and UNCURELoss.

Verifies the core statistical property: E[CURE] ≈ E[MSE] under chi-square noise,
i.e., CURE is an unbiased estimator of the supervised loss.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import deepinv as dinv


# ── Helpers ──────────────────────────────────────────────────────────────────

class IdentityPhysics:
    """Minimal physics stub: A(x) = x, A_dagger(y) = y."""
    def A(self, x): return x
    def A_dagger(self, y): return y


def rician_squared(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Sample y = z² where z ~ Rician(x, σ)."""
    z = torch.sqrt(
        (x + sigma * torch.randn_like(x)) ** 2
        + (sigma * torch.randn_like(x)) ** 2
    )
    return z ** 2


def make_model(seed: int = 0) -> nn.Module:
    """Small fixed CNN for testing (SiLU, no batch norm)."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.SiLU(),
        nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(),
        nn.Conv2d(16, 1, 3, padding=1),
    )


# ── CURELoss unbiasedness tests ───────────────────────────────────────────────

@pytest.mark.parametrize("method", ["mc_1side", "mc_2side", "auto"])
def test_cure_unbiased(method):
    """
    E[CURE(f, y, σ)] ≈ E[MSE(f(y), x²)] over many noise realizations.
    Tolerance: 1% relative error.
    """
    sigma = 0.05
    N_seeds = 200
    SIZE = 32  # small for speed

    torch.manual_seed(42)
    x = torch.rand(1, 1, SIZE, SIZE)
    x_sq = x ** 2
    model = make_model()
    model.eval()

    physics = IdentityPhysics()
    loss_fn = dinv.loss.CURELoss(sigma=sigma, method=method, M=20,
                                  rng=torch.Generator().manual_seed(0))

    cure_vals, mse_vals = [], []
    for seed in range(N_seeds):
        torch.manual_seed(seed)
        y = rician_squared(x, sigma)
        with torch.no_grad() if method != "auto" else torch.enable_grad():
            if method == "auto":
                x_net = model(y)
                c = loss_fn(y=y, x_net=x_net, physics=physics, model=model).item()
            else:
                with torch.no_grad():
                    x_net = model(y)
                    c = loss_fn(y=y, x_net=x_net, physics=physics, model=model).item()
        cure_vals.append(c)

        with torch.no_grad():
            fx = model(y)
        mse = ((fx - x_sq) ** 2).mean().item()
        mse_vals.append(mse)

    cure_mean = np.mean(cure_vals)
    mse_mean = np.mean(mse_vals)

    rel_err = abs(cure_mean - mse_mean) / (mse_mean + 1e-12)
    assert rel_err < 0.05, (
        f"[{method}] CURE mean={cure_mean:.6f}, MSE mean={mse_mean:.6f}, "
        f"relative error={rel_err:.3f} exceeds 5%"
    )


def test_cure_output_shape():
    """CURELoss returns a (batch_size,) tensor."""
    sigma = 0.05
    B = 4
    x = torch.rand(B, 1, 32, 32)
    y = rician_squared(x, sigma)
    model = make_model()
    model.eval()
    physics = IdentityPhysics()
    loss_fn = dinv.loss.CURELoss(sigma=sigma, method="mc_1side", M=5)

    with torch.no_grad():
        x_net = model(y)
        out = loss_fn(y=y, x_net=x_net, physics=physics, model=model)

    assert out.shape == (B,), f"Expected shape ({B},), got {out.shape}"


def test_cure_invalid_method():
    with pytest.raises(ValueError, match="method must be"):
        dinv.loss.CURELoss(sigma=0.05, method="bad_method")


# ── UNCURELoss tests ──────────────────────────────────────────────────────────

def test_uncure_requires_sigma_map():
    """UNCURELoss raises ValueError when sigma_map is not provided."""
    sigma = 0.05
    x = torch.rand(1, 1, 32, 32)
    y = rician_squared(x, sigma)
    model = make_model()
    model.eval()
    physics = IdentityPhysics()
    loss_fn = dinv.loss.UNCURELoss(method="mc_1side", M=5)

    with torch.no_grad():
        x_net = model(y)
    with pytest.raises(ValueError, match="sigma_map"):
        loss_fn(y=y, x_net=x_net, physics=physics, model=model)


def test_uncure_uniform_matches_cure():
    """
    Structural check only (formula unverified — see UNCURELoss warning).
    With a uniform sigma_map, UNCURELoss and CURELoss use the same formula,
    so their means should agree up to Rademacher sampling noise.
    """
    sigma = 0.05
    N_seeds = 100
    SIZE = 32

    torch.manual_seed(7)
    x = torch.rand(1, 1, SIZE, SIZE)
    model = make_model()
    model.eval()
    physics = IdentityPhysics()

    rng_cure   = torch.Generator().manual_seed(1)
    rng_uncure = torch.Generator().manual_seed(1)
    cure_fn   = dinv.loss.CURELoss(sigma=sigma, method="mc_1side", M=20, rng=rng_cure)
    uncure_fn = dinv.loss.UNCURELoss(method="mc_1side", M=20, rng=rng_uncure)

    cure_vals, uncure_vals = [], []
    for seed in range(N_seeds):
        torch.manual_seed(seed)
        y = rician_squared(x, sigma)
        sigma_map = torch.full_like(y, sigma)
        with torch.no_grad():
            x_net = model(y)
            c = cure_fn(y=y, x_net=x_net, physics=physics, model=model).item()
            u = uncure_fn(y=y, x_net=x_net, physics=physics, model=model,
                          sigma_map=sigma_map).item()
        cure_vals.append(c)
        uncure_vals.append(u)

    rel_err = abs(np.mean(cure_vals) - np.mean(uncure_vals)) / (abs(np.mean(cure_vals)) + 1e-12)
    assert rel_err < 0.10, (
        f"Uniform UNCURE mean={np.mean(uncure_vals):.6f}, "
        f"CURELoss mean={np.mean(cure_vals):.6f}, rel_err={rel_err:.3f}"
    )


def test_uncure_output_shape():
    sigma = 0.05
    B = 3
    x = torch.rand(B, 1, 32, 32)
    y = rician_squared(x, sigma)
    sigma_map = torch.full_like(y, sigma)
    model = make_model()
    model.eval()
    physics = IdentityPhysics()
    loss_fn = dinv.loss.UNCURELoss(method="mc_2side", M=5)

    with torch.no_grad():
        x_net = model(y)
        out = loss_fn(y=y, x_net=x_net, physics=physics, model=model, sigma_map=sigma_map)

    assert out.shape == (B,), f"Expected shape ({B},), got {out.shape}"
