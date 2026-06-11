from __future__ import annotations

import torch
import torch.nn as nn
from deepinv.loss.loss import Loss

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepinv.physics import Physics


class CURELoss(Loss):
    r"""
    CURE loss for Rician noise (Chi-Square Unbiased Risk Estimate).

    CURE is the analog of SURE for Rician / chi-square noise. It provides an
    unbiased estimate of the MSE without requiring clean reference images,
    enabling fully unsupervised training of denoisers and inverse-problem networks
    when only noisy magnitude MRI observations are available.

    **Noise model.**  Given a clean image :math:`x`, the Rician-noisy magnitude
    observation is

    .. math::

        z = \sqrt{(x + \sigma\varepsilon_1)^2 + (\sigma\varepsilon_2)^2},
        \quad \varepsilon_1, \varepsilon_2 \sim \mathcal{N}(0, I).

    Squaring gives the chi-square domain observation :math:`y = z^2`, which satisfies

    .. math::

        \mathbb{E}[y] = x^2 + 2\sigma^2 = x^2 + K', \quad K' = 2\sigma^2.

    The network :math:`f` takes :math:`y` as input and estimates :math:`x^2`.

    **CURE formula** (derived as :math:`\sigma^4 \times` canonical CURE,
    Theorem 4.2 in :footcite:t:`carlavan2012cure`):

    .. math::

        \mathrm{CURE} = \frac{1}{N}\|f(y) - (y - K')\|^2
        - \frac{4\sigma^2}{N} \mathbf{1}^\top v
        + \frac{8\sigma^2}{N}\, v^\top \mathrm{diag}(J_f)

    where :math:`v = y - \sigma^2` and :math:`J_f` is the Jacobian of :math:`f`.
    The third term is estimated via Hutchinson's identity
    :math:`\mathbb{E}_b[(v \odot b)^\top J_f b] = v^\top \mathrm{diag}(J_f)`
    with :math:`b \sim \mathrm{Rademacher}\{-1, +1\}`.

    The :math:`\sigma^2` factors on terms 2 and 3 are critical: without them the
    divergence correction is :math:`1/\sigma^2` times too large, causing training
    to diverge.

    The second-order (Hessian) correction term is omitted by default. For smooth
    activations (e.g. SiLU, GELU) on inputs in :math:`[0,1]`, the resulting bias
    is :math:`\mathcal{O}(\sigma^4) \approx 10^{-6}`, which is negligible.

    :param float sigma: Rician noise standard deviation :math:`\sigma`.
    :param str method: Divergence estimator. One of:

        - ``'mc_1side'``: one-sided finite differences (``M+1`` forward passes).
        - ``'mc_2side'``: central finite differences (``2M+1`` forward passes, lower bias).
        - ``'auto'``: exact Jacobian-vector product via :func:`torch.autograd.grad`
          (no finite-difference bias; requires ``create_graph=True``).

    :param float tau: Finite-difference step size. Only used for ``'mc_1side'`` and
        ``'mc_2side'``. Default ``0.01`` is suited to inputs in :math:`[0, 1]`.
    :param int M: Number of Rademacher vectors to average for the Hutchinson estimator.
        Higher values reduce variance at the cost of :math:`M` extra forward passes.
        Default ``10`` balances cost and stability.
    :param torch.Generator rng: Optional random number generator for reproducibility.

    **Example** (unsupervised denoiser training)::

        import torch
        import deepinv as dinv

        sigma = 0.05
        physics = dinv.physics.Denoising()
        model = dinv.models.DnCNN(in_channels=1, out_channels=1)
        loss = dinv.loss.CURELoss(sigma=sigma, method='auto', M=10)

        z = physics(x)        # Rician noisy magnitude (from e.g. MRI scanner)
        y = z ** 2            # squared domain
        x_net = model(y)
        loss_val = loss(y=y, x_net=x_net, physics=physics, model=model).mean()
        loss_val.backward()

    .. note::

        The input ``y`` to the network and to this loss must be the **squared**
        magnitude :math:`y = z^2`, not the raw Rician magnitude :math:`z`.
        PSNR evaluation converts back via :math:`\hat{x} = \sqrt{f(y).clamp(0)}`.

    .. footbibliography::
    """

    def __init__(
        self,
        sigma: float,
        method: str = "mc_1side",
        tau: float = 1e-2,
        M: int = 10,
        rng: torch.Generator = None,
    ):
        super().__init__()
        if method not in ("mc_1side", "mc_2side", "auto"):
            raise ValueError(f"method must be 'mc_1side', 'mc_2side', or 'auto', got '{method}'")
        self.sigma2 = sigma ** 2
        self.method = method
        self.tau = tau
        self.M = M
        self.rng = rng

    def _rademacher(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        return torch.randint(0, 2, shape, device=device, dtype=torch.float32, generator=self.rng) * 2 - 1

    def forward(
        self,
        y: torch.Tensor,
        x_net: torch.Tensor,
        physics: Physics,
        model: nn.Module,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the CURE loss.

        :param torch.Tensor y: Squared Rician observations :math:`y = z^2`.
        :param torch.Tensor x_net: Network estimate :math:`f(y)` of :math:`x^2`.
        :param deepinv.physics.Physics physics: Forward operator (used for API
            compatibility; CURE operates directly in measurement space).
        :param torch.nn.Module model: Reconstruction network :math:`f`.
        :return: :class:`torch.Tensor` of shape ``(batch_size,)`` — per-sample loss.
        """
        B = y.size(0)
        N = y[0].numel()
        k_prime = 2.0 * self.sigma2   # K' = 2σ²
        v = y - self.sigma2            # v = y − K'/2 = y − σ²

        # term1: ||f(y) − (y − K')||² / N
        term1 = ((x_net - (y - k_prime)) ** 2).reshape(B, -1).mean(dim=1)

        # term2: −(4σ²/N) · Σ v  (no model, analytic)
        term2 = (-4.0 * self.sigma2) * v.reshape(B, -1).mean(dim=1)

        # term3: (8σ²/N) · vᵀ diag(J_f)  estimated via Hutchinson
        term3 = self._hutchinson(model, y, v, B, N, physics)

        return term1 + term2 + term3

    def _hutchinson(
        self,
        model: nn.Module,
        y: torch.Tensor,
        v: torch.Tensor,
        B: int,
        N: int,
        physics: "Physics" = None,
    ) -> torch.Tensor:
        coeff1 = 8.0 * self.sigma2 / N
        coeff2 = 4.0 * self.sigma2 / N  # central diff: halved coefficient

        def f(x):
            return model(x, physics) if physics is not None else model(x)

        t3 = torch.zeros(B, device=y.device)

        if self.method == "auto":
            y_in = y.detach().requires_grad_(True)
            fy = f(y_in)
            for _ in range(self.M):
                b = self._rademacher(y.shape, y.device)
                Jft_b = torch.autograd.grad(
                    fy, y_in, b, retain_graph=True, create_graph=True
                )[0]
                t3 = t3 + coeff1 * ((v * b) * Jft_b).reshape(B, -1).sum(dim=1)

        elif self.method == "mc_1side":
            fy = f(y)
            for _ in range(self.M):
                b = self._rademacher(y.shape, y.device)
                fyp = f(y + self.tau * b)
                t3 = t3 + coeff1 * ((v * b) * (fyp - fy) / self.tau).reshape(B, -1).sum(dim=1)

        else:  # mc_2side
            fy = f(y)
            for _ in range(self.M):
                b = self._rademacher(y.shape, y.device)
                fyp = f(y + self.tau * b)
                fym = f(y - self.tau * b)
                t3 = t3 + coeff2 * ((v * b) * (fyp - fym) / self.tau).reshape(B, -1).sum(dim=1)

        return t3 / self.M


class UNCURELoss(Loss):
    r"""
    UNCURE loss for spatially-varying Rician noise (multi-coil MRI).

    .. warning::

        **Draft / unverified.** The formula below is an intuitive extrapolation of
        :class:`CURELoss` to the spatially-varying case. It has *not* been rigorously
        derived from Theorem 4.2 of :footcite:t:`carlavan2012cure` for non-i.i.d.
        chi-square observations. Use :class:`CURELoss` for uniform-noise experiments.
        This class will be updated once the derivation is complete (Phase 4).

    Extension of :class:`CURELoss` to the case where the noise standard deviation
    varies spatially, as in parallel MRI where the g-factor map :math:`g(i)` modulates
    the noise level at each pixel:

    .. math::

        \sigma_i = \sigma_0 \cdot g(i).

    After sum-of-squares coil combination, pixel :math:`i` follows a scaled
    chi-square distribution with local parameter :math:`K'_i = 2\sigma_i^2`.
    The *proposed* UNCURE formula replaces the uniform :math:`\sigma^2` weight with a
    pixel-wise :math:`\sigma_i^2` weight:

    .. math::

        \mathrm{UNCURE} = \frac{1}{N}\|f(y) - (y - K')\|^2
        - \frac{4}{N} \sum_i \sigma_i^2 v_i
        + \frac{8}{N} \sum_i \sigma_i^2 v_i\, [J_f]_{ii}

    where :math:`v_i = y_i - \sigma_i^2`, :math:`K'_i = 2\sigma_i^2`.

    :param str method: Divergence estimator: ``'mc_1side'``, ``'mc_2side'``, or ``'auto'``.
    :param float tau: Finite-difference step size. Default ``0.01``.
    :param int M: Number of Rademacher vectors for Hutchinson estimator. Default ``10``.
    :param torch.Generator rng: Optional random number generator.

    .. note::

        Pass the noise map as ``sigma_map`` (shape matching ``y``) in the ``forward`` call.
        The map should contain pixel-wise :math:`\sigma_i` values, not :math:`\sigma_i^2`.
    """

    def __init__(
        self,
        method: str = "mc_1side",
        tau: float = 1e-2,
        M: int = 10,
        rng: torch.Generator = None,
    ):
        super().__init__()
        if method not in ("mc_1side", "mc_2side", "auto"):
            raise ValueError(f"method must be 'mc_1side', 'mc_2side', or 'auto', got '{method}'")
        self.method = method
        self.tau = tau
        self.M = M
        self.rng = rng

    def _rademacher(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        return torch.randint(0, 2, shape, device=device, dtype=torch.float32, generator=self.rng) * 2 - 1

    def forward(
        self,
        y: torch.Tensor,
        x_net: torch.Tensor,
        physics: Physics,
        model: nn.Module,
        sigma_map: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the UNCURE loss.

        :param torch.Tensor y: Squared Rician observations :math:`y = z^2`.
        :param torch.Tensor x_net: Network estimate :math:`f(y)` of :math:`x^2`.
        :param deepinv.physics.Physics physics: Forward operator.
        :param torch.nn.Module model: Reconstruction network.
        :param torch.Tensor sigma_map: Pixel-wise noise std map :math:`\sigma_i`,
            same shape as ``y``. Required — raises ``ValueError`` if not provided.
        :return: :class:`torch.Tensor` of shape ``(batch_size,)`` — per-sample loss.
        """
        if sigma_map is None:
            raise ValueError(
                "UNCURELoss requires a pixel-wise sigma_map. "
                "For uniform noise, use CURELoss instead."
            )

        B = y.size(0)
        N = y[0].numel()
        sigma2_map = sigma_map ** 2          # σᵢ²
        k_prime = 2.0 * sigma2_map           # K'ᵢ = 2σᵢ²
        v = y - sigma2_map                   # vᵢ = yᵢ − σᵢ²

        # term1: ||f(y) − (y − K')||² / N  (pixel-wise K')
        term1 = ((x_net - (y - k_prime)) ** 2).reshape(B, -1).mean(dim=1)

        # term2: −(4/N) · Σᵢ σᵢ² · vᵢ
        term2 = -4.0 * (sigma2_map * v).reshape(B, -1).mean(dim=1)

        # term3: (8/N) · Σᵢ σᵢ² · vᵢ · [J_f]ᵢᵢ  via Hutchinson
        term3 = self._hutchinson(model, y, v, sigma2_map, B, N, physics)

        return term1 + term2 + term3

    def _hutchinson(
        self,
        model: nn.Module,
        y: torch.Tensor,
        v: torch.Tensor,
        sigma2_map: torch.Tensor,
        B: int,
        N: int,
        physics: "Physics" = None,
    ) -> torch.Tensor:
        # weight = σᵢ² · vᵢ  (replaces uniform σ² · v in CURELoss)
        sv = sigma2_map * v

        coeff1 = 8.0 / N
        coeff2 = 4.0 / N

        def f(x):
            return model(x, physics) if physics is not None else model(x)

        t3 = torch.zeros(B, device=y.device)

        if self.method == "auto":
            y_in = y.detach().requires_grad_(True)
            fy = f(y_in)
            for _ in range(self.M):
                b = self._rademacher(y.shape, y.device)
                Jft_b = torch.autograd.grad(
                    fy, y_in, b, retain_graph=True, create_graph=True
                )[0]
                t3 = t3 + coeff1 * ((sv * b) * Jft_b).reshape(B, -1).sum(dim=1)

        elif self.method == "mc_1side":
            fy = f(y)
            for _ in range(self.M):
                b = self._rademacher(y.shape, y.device)
                fyp = f(y + self.tau * b)
                t3 = t3 + coeff1 * ((sv * b) * (fyp - fy) / self.tau).reshape(B, -1).sum(dim=1)

        else:  # mc_2side
            fy = f(y)
            for _ in range(self.M):
                b = self._rademacher(y.shape, y.device)
                fyp = f(y + self.tau * b)
                fym = f(y - self.tau * b)
                t3 = t3 + coeff2 * ((sv * b) * (fyp - fym) / self.tau).reshape(B, -1).sum(dim=1)

        return t3 / self.M
