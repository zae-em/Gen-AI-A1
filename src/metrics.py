"""Evaluation metrics for restoration and classification."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

from .losses import ssim


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0
         ) -> torch.Tensor:
    """Per-image PSNR in dB for a (N, C, H, W) batch."""
    mse = ((pred - target) ** 2).flatten(1).mean(1)
    return 10.0 * torch.log10((data_range ** 2) / (mse + 1e-12))


def ssim_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ssim(pred, target, size_average=False)


class RestorationAccumulator:
    """Collects PSNR/SSIM broken down by corruption type and severity.

    The assignment asks for results separated by corruption *and* by low /
    medium / high severity, so the breakdown is accumulated during evaluation
    rather than reconstructed afterwards.
    """

    def __init__(self):
        self._psnr: Dict[tuple, List[float]] = defaultdict(list)
        self._ssim: Dict[tuple, List[float]] = defaultdict(list)

    def update(self, pred, target, corruptions: List[str], severities: List[str]):
        p = psnr(pred, target).detach().cpu().numpy()
        s = ssim_per_image(pred, target).detach().cpu().numpy()
        for i, (c, sev) in enumerate(zip(corruptions, severities)):
            key = (c, sev if sev is not None else "-")
            self._psnr[key].append(float(p[i]))
            self._ssim[key].append(float(s[i]))
            self._psnr[(c, "all")].append(float(p[i]))
            self._ssim[(c, "all")].append(float(s[i]))
            self._psnr[("overall", "all")].append(float(p[i]))
            self._ssim[("overall", "all")].append(float(s[i]))

    def summary(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for key in sorted(self._psnr):
            name = f"{key[0]}/{key[1]}"
            out[name] = {
                "psnr": float(np.mean(self._psnr[key])),
                "ssim": float(np.mean(self._ssim[key])),
                "n": len(self._psnr[key]),
            }
        return out

    def as_table(self) -> str:
        rows = self.summary()
        width = max(len(k) for k in rows) + 2
        lines = [f"{'condition'.ljust(width)}{'PSNR':>9}{'SSIM':>9}{'n':>8}"]
        lines.append("-" * (width + 26))
        for k, v in rows.items():
            lines.append(f"{k.ljust(width)}{v['psnr']:>9.3f}{v['ssim']:>9.4f}{v['n']:>8d}")
        return "\n".join(lines)


def classification_report_dict(y_true, y_pred, labels, target_names):
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 precision_recall_fscore_support)
    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=target_names, output_dict=True,
                                   zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    return {
        "report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalised": cm_norm.tolist(),
        "accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
    }
