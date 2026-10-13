"""Figure helpers for the report: comparison grids, error maps, routing heat maps."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import torch                           # noqa: E402


def _to_numpy(img: torch.Tensor) -> np.ndarray:
    return img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def comparison_grid(clean: torch.Tensor, corrupt: torch.Tensor,
                    recon: torch.Tensor, out_path: str | Path,
                    titles: Optional[Sequence[str]] = None,
                    max_rows: int = 12) -> Path:
    """Four columns per row: clean target, corrupted input, output, |error|.

    The error map uses a shared 0-0.5 scale across every row so that panels
    from different rows are directly comparable - a per-row autoscale makes a
    good reconstruction look as red as a bad one.
    """
    n = min(clean.shape[0], max_rows)
    fig, axes = plt.subplots(n, 4, figsize=(9, 2.3 * n))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["clean target", "corrupted input", "reconstruction", "|error|"]

    for r in range(n):
        err = (clean[r] - recon[r]).abs().mean(0).detach().cpu().numpy()
        panels = [_to_numpy(clean[r]), _to_numpy(corrupt[r]), _to_numpy(recon[r]), err]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            if c == 3:
                im = ax.imshow(panel, cmap="inferno", vmin=0.0, vmax=0.5)
                if r == 0:
                    fig.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.imshow(panel)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9)
        if titles is not None and r < len(titles):
            axes[r, 0].set_ylabel(titles[r], fontsize=8)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def routing_heatmap(weights_by_class: dict, out_path: str | Path,
                    branch_names: Sequence[str]) -> Path:
    """Rows = true corruption, columns = branch. Cell = mean routing weight."""
    rows = list(weights_by_class.keys())
    mat = np.array([weights_by_class[r] for r in rows])

    fig, ax = plt.subplots(figsize=(1.4 * len(branch_names) + 2, 1.0 * len(rows) + 2))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(branch_names)), branch_names, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("routing branch"); ax.set_ylabel("true corruption")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def confusion_matrix_figure(cm_norm, class_names: Sequence[str],
                            out_path: str | Path) -> Path:
    cm = np.asarray(cm_norm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def curve_figure(history: dict, out_path: str | Path, ylabel: str = "loss") -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, values in history.items():
        ax.plot(range(1, len(values) + 1), values, label=name, linewidth=1.6)
    ax.set_xlabel("epoch"); ax.set_ylabel(ylabel); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def sketch_grid(photos: torch.Tensor, targets: torch.Tensor, generated: torch.Tensor,
                out_path: str | Path, styles: Optional[List[int]] = None,
                max_rows: int = 8) -> Path:
    """Task 4 qualitative grid. Inputs are in [-1, 1]."""
    n = min(photos.shape[0], max_rows)
    fig, axes = plt.subplots(n, 3, figsize=(7, 2.3 * n))
    if n == 1:
        axes = axes[None, :]
    for r in range(n):
        for c, batch in enumerate((photos, targets, generated)):
            img = ((batch[r] + 1) / 2).detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            axes[r, c].imshow(img)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(["photograph", "target sketch", "generated"][c],
                                     fontsize=9)
        if styles is not None:
            axes[r, 0].set_ylabel(f"style {styles[r] + 1}", fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
