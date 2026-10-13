"""Soft mixture-of-experts restoration (Task 3).

Four branches, in the fixed CORRUPTION_NAMES order:

    0  identity     (clean bypass, no parameters)
    1  salt & pepper specialist
    2  blur specialist
    3  occlusion specialist

    w      = softmax(gate(x) / T)
    x_hat  = w0 * x + w1 * E_sp(x) + w2 * E_blur(x) + w3 * E_occ(x)

The whole thing is differentiable, so the gate and the experts can be trained
through the final reconstruction error.  The temperature T controls how peaked
the routing is: small T approaches the hard routing of Task 2, large T spreads
weight across branches.

The forward pass runs all three experts on every input.  That is the price of
soft routing and it is worth stating plainly in the report: inference cost is
roughly 3x the hard-routed system, in exchange for graceful behaviour when the
gate is uncertain or the corruption is mixed.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..data.corruptions import CORRUPTION_NAMES
from .autoencoder import DenoisingAutoencoder
from .classifier import CorruptionClassifier

BRANCH_NAMES = CORRUPTION_NAMES  # identity, salt_pepper, blur, occlusion


class SoftMoERestorer(nn.Module):
    def __init__(self, gate: CorruptionClassifier,
                 experts: List[DenoisingAutoencoder],
                 temperature: float = 1.0):
        super().__init__()
        if len(experts) != 3:
            raise ValueError("expected exactly three experts: salt, blur, occlusion")
        self.gate = gate
        self.experts = nn.ModuleList(experts)
        self.temperature = float(temperature)

    def set_expert_grad(self, requires_grad: bool) -> None:
        """Warm-up helper: freeze the experts and train the gate alone first."""
        for p in self.experts.parameters():
            p.requires_grad_(requires_grad)

    def forward(self, x):
        logits = self.gate(x)
        weights = torch.softmax(logits / self.temperature, dim=1)

        branches = [x] + [expert(x) for expert in self.experts]
        stacked = torch.stack(branches, dim=1)              # (N, 4, 3, H, W)
        w = weights.view(weights.shape[0], weights.shape[1], 1, 1, 1)
        recon = (stacked * w).sum(dim=1)
        return recon, weights, logits


class SoftMoEExport(nn.Module):
    """ONNX export wrapper: returns (reconstruction, routing weights)."""

    def __init__(self, moe: SoftMoERestorer):
        super().__init__()
        self.moe = moe

    def forward(self, x):
        recon, weights, _ = self.moe(x)
        return recon, weights


@torch.no_grad()
def branch_weight_report(model: SoftMoERestorer, loader, device,
                         severity_key=None) -> dict:
    """Average routing weight per branch, grouped by true corruption type.

    This is what fills the routing heat map in the report and answers the
    question the assignment asks explicitly: has any expert gone dead, and
    does any expert dominate inputs it has nothing to do with?
    """
    from collections import defaultdict
    sums = defaultdict(lambda: torch.zeros(len(BRANCH_NAMES)))
    counts = defaultdict(int)

    model.eval()
    for batch in loader:
        corrupt, _clean, labels, _idx = batch
        _recon, weights, _logits = model(corrupt.to(device))
        weights = weights.detach().cpu()
        for i, lab in enumerate(labels.tolist()):
            key = CORRUPTION_NAMES[lab]
            sums[key] += weights[i]
            counts[key] += 1

    return {k: (sums[k] / max(counts[k], 1)).tolist() for k in sums}
