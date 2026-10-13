"""Loss functions shared across the four tasks.

SSIM is implemented here rather than pulled from a package so that the exact
window, padding and stabilising constants are visible and can be defended in
the report (and so the ONNX graph has no third-party custom ops in it).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------
def _gaussian_window(window_size: int, sigma: float, channels: int,
                     device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g[:, None] @ g[None, :]
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0,
         window_size: int = 11, sigma: float = 1.5,
         size_average: bool = True) -> torch.Tensor:
    """Mean structural similarity between two (N, C, H, W) batches.

    Uses the standard Wang et al. (2004) formulation: an 11x11 Gaussian window
    with sigma = 1.5, K1 = 0.01, K2 = 0.03.
    """
    c = x.shape[1]
    window = _gaussian_window(window_size, sigma, c, x.device, x.dtype)
    pad = window_size // 2

    mu_x = F.conv2d(x, window, padding=pad, groups=c)
    mu_y = F.conv2d(y, window, padding=pad, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=c) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=c) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=c) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / (denominator + 1e-12)

    if size_average:
        return ssim_map.mean()
    return ssim_map.flatten(1).mean(1)


class SSIMLoss(nn.Module):
    def forward(self, x, y):
        return 1.0 - ssim(x, y)


class ReconstructionLoss(nn.Module):
    """alpha * L1 + (1 - alpha) * (1 - SSIM).

    alpha near 1 favours pixel fidelity, alpha near 0 favours structure.
    The assignment suggests starting at 0.8; the final value is chosen by the
    Optuna study, not assumed.
    """

    def __init__(self, alpha: float = 0.8):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        s = 1.0 - ssim(pred, target)
        return self.alpha * l1 + (1.0 - self.alpha) * s, l1.detach(), s.detach()


# ---------------------------------------------------------------------------
# routing balance (Task 3)
# ---------------------------------------------------------------------------
def balance_loss(weights: torch.Tensor) -> torch.Tensor:
    """sum_i (mean_batch(w_i) - 1/K)^2.

    Penalises a gate that collapses onto a single branch.  Because the training
    batches are class balanced, the ideal average weight per branch is 1/K, so
    the uniform target is the correct one here.  An entropy regulariser was the
    alternative considered - see the report for why this one was kept.
    """
    k = weights.shape[1]
    mean_w = weights.mean(dim=0)
    return ((mean_w - 1.0 / k) ** 2).sum()


def routing_entropy(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean per-sample entropy of the routing weights. Diagnostic, not a loss.

    Low entropy means the gate behaves like hard routing; high entropy means it
    genuinely blends experts.
    """
    return -(weights * (weights + eps).log()).sum(dim=1).mean()
