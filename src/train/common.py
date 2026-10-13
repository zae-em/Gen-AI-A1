"""Shared plumbing for the Task 1-3 training scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..data.corruptions import CORRUPTION_NAMES
from ..data.datasets import (PetRestorationDataset, dataset_from_manifest,
                             make_split)
from ..losses import ssim
from ..metrics import psnr
from ..utils.seed import worker_init_fn


def build_loaders(data_root: str, val_manifest: str, batch_size: int,
                  image_size: int = 128, mode: str = "random",
                  force_kind: Optional[str] = None, num_workers: int = 4,
                  augment: bool = True):
    """Training loader (runtime corruption) + validation loader (manifest).

    `mode` selects how training corruptions are drawn:
      'random'   - Task 1 and Task 3 (uniform over the four conditions)
      'balanced' - Task 2 classifier (class-balanced batches)
      'forced'   - Task 2 specialists (one corruption only)
    """
    train_paths, _ = make_split(data_root, image_size)
    train_ds = PetRestorationDataset(train_paths, mode=mode, image_size=image_size,
                                     force_kind=force_kind, augment=augment)
    val_ds = dataset_from_manifest(val_manifest, image_size)

    if force_kind is not None:
        # specialists only ever see their own corruption at validation time
        keep = [i for i, e in enumerate(val_ds.manifest) if e["corruption"] == force_kind]
        val_ds.manifest = [val_ds.manifest[i] for i in keep]

    common = dict(num_workers=num_workers, pin_memory=True,
                  worker_init_fn=worker_init_fn, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader


def combined_score(psnr_value: float, ssim_value: float) -> float:
    """Single scalar for model selection and for the Optuna objective.

    PSNR is divided by 50 dB so that the two terms sit on comparable scales
    (a good restoration lands around 25-30 dB, i.e. 0.5-0.6 after scaling,
    against an SSIM of roughly 0.8-0.9). Both are maximised.
    """
    return ssim_value + psnr_value / 50.0


@torch.no_grad()
def evaluate_reconstruction(model, loader, device) -> Dict[str, float]:
    model.eval()
    tot_psnr = tot_ssim = 0.0
    n = 0
    per_class_psnr = {name: [0.0, 0] for name in CORRUPTION_NAMES}

    for corrupt, clean, labels, _idx in loader:
        corrupt, clean = corrupt.to(device), clean.to(device)
        recon = model(corrupt)
        if isinstance(recon, tuple):
            recon = recon[0]
        recon = recon.clamp(0, 1)

        p = psnr(recon, clean)
        s = ssim(recon, clean, size_average=False)
        tot_psnr += float(p.sum()); tot_ssim += float(s.sum()); n += clean.shape[0]
        for i, lab in enumerate(labels.tolist()):
            per_class_psnr[CORRUPTION_NAMES[lab]][0] += float(p[i])
            per_class_psnr[CORRUPTION_NAMES[lab]][1] += 1

    out = {"val_psnr": tot_psnr / max(n, 1), "val_ssim": tot_ssim / max(n, 1)}
    out["val_score"] = combined_score(out["val_psnr"], out["val_ssim"])
    for name, (total, count) in per_class_psnr.items():
        if count:
            out[f"val_psnr_{name}"] = total / count
    return out


def save_checkpoint(path: str | Path, model, config: dict, extra: Optional[dict] = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "config": config}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
