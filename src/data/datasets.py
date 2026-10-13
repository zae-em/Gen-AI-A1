"""Oxford-IIIT Pet data pipeline shared by Tasks 1-3.

The split is created once, from the official *trainval* collection, with
numpy's RandomState(42) so that Tasks 1, 2 and 3 all see exactly the same
images.  The official *test* collection is never touched by training or by
hyper-parameter selection.

Training corruptions are sampled at load time (a fresh corruption every time
an image is read).  Validation and test corruptions come from a manifest so
that every evaluation run is comparable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from .corruptions import (CORRUPTION_NAMES, CorruptionSpec, apply_spec,
                          sample_spec)

IMAGE_SIZE = 128
SPLIT_SEED = 42
VAL_FRACTION = 0.2


def _base_transform(image_size: int = IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),          # -> float32 in [0, 1]
    ])


def load_pet_paths(root: str, split: str) -> List[Path]:
    """Return the image paths of an official Oxford-IIIT Pet split.

    `split` is 'trainval' or 'test'.  torchvision downloads and verifies the
    archive; we only use it to resolve file paths so that the rest of the
    pipeline stays framework-agnostic.
    """
    ds = OxfordIIITPet(root=root, split=split, download=True)
    return [Path(p) for p in ds._images]


def make_split(root: str, image_size: int = IMAGE_SIZE
               ) -> Tuple[List[Path], List[Path]]:
    """80/20 train/val split of the official trainval collection, seed 42."""
    paths = load_pet_paths(root, "trainval")
    rng = np.random.RandomState(SPLIT_SEED)
    order = rng.permutation(len(paths))
    n_val = int(round(VAL_FRACTION * len(paths)))
    val_idx, train_idx = order[:n_val], order[n_val:]
    train = [paths[i] for i in sorted(train_idx)]
    val = [paths[i] for i in sorted(val_idx)]
    return train, val


class PetRestorationDataset(Dataset):
    """Clean target + corrupted input + corruption label.

    Parameters
    ----------
    paths       : image files for this split.
    mode        : 'random'   - fresh corruption per read (training)
                  'balanced' - as above but the corruption class is forced to
                               index % 4, so every batch is class balanced.
                               Length is 4x the number of images.
                  'forced'   - always `force_kind` (specialist training).
                  'manifest' - corruption comes from a stored manifest.
    """

    def __init__(self, paths: List[Path], mode: str = "random",
                 image_size: int = IMAGE_SIZE,
                 force_kind: Optional[str] = None,
                 manifest: Optional[List[dict]] = None,
                 augment: bool = False):
        self.paths = list(paths)
        self.mode = mode
        self.image_size = image_size
        self.force_kind = force_kind
        self.manifest = manifest
        self.augment = augment
        self.transform = _base_transform(image_size)

        if mode == "manifest" and manifest is None:
            raise ValueError("manifest mode requires a manifest")

    def __len__(self) -> int:
        if self.mode == "manifest":
            return len(self.manifest)
        if self.mode == "balanced":
            return len(self.paths) * len(CORRUPTION_NAMES)
        return len(self.paths)

    def _read(self, path: Path) -> torch.Tensor:
        with Image.open(path) as im:
            return self.transform(im.convert("RGB"))

    def __getitem__(self, idx: int):
        if self.mode == "manifest":
            entry = self.manifest[idx]
            clean = self._read(self.paths[entry["image_index"]])
            spec = CorruptionSpec.from_dict(entry["spec"])
            gen = torch.Generator().manual_seed(int(entry.get("seed", idx)))
            corrupt = apply_spec(clean, spec, gen)
            return corrupt, clean, spec.label, idx

        if self.mode == "balanced":
            img_idx, kind_idx = divmod(idx, len(CORRUPTION_NAMES))
            forced = CORRUPTION_NAMES[kind_idx]
        else:
            img_idx = idx
            forced = self.force_kind          # None in 'random' mode

        clean = self._read(self.paths[img_idx])

        if self.augment:
            if torch.rand(1).item() < 0.5:
                clean = torch.flip(clean, dims=[2])

        spec = sample_spec(self.image_size, self.image_size, force_kind=forced)
        corrupt = apply_spec(clean, spec)
        return corrupt, clean, spec.label, idx


def load_manifest(path: str | Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def dataset_from_manifest(manifest_file: str | Path,
                          image_size: int = IMAGE_SIZE) -> PetRestorationDataset:
    data = load_manifest(manifest_file)
    paths = [Path(p) for p in data["paths"]]
    return PetRestorationDataset(paths, mode="manifest", image_size=image_size,
                                 manifest=data["entries"])
