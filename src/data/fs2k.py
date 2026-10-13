"""FS2K paired photograph / sketch dataset for Task 4.

FS2K ships two annotation files, `anno_train.json` and `anno_test.json`.  Each
record carries an `image_name` such as `photo/1/image0001` and a `style` field
in {0, 1, 2}.  Photographs and sketches live in parallel directories and do not
always share a file extension, so paths are resolved by globbing.

15% of the official training portion is held out as validation, stratified by
style, with seed 42.  The official test portion is never used for training or
hyper-parameter selection.

Spatial augmentation is applied *identically* to the photograph and the sketch
- the pair must stay pixel-aligned or the L1 term becomes meaningless.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

STYLE_NAMES = ("Style 1", "Style 2", "Style 3")
N_STYLES = 3
IMAGE_SIZE = 128
SPLIT_SEED = 42
VAL_FRACTION = 0.15


def _resolve(root: Path, rel: str, folder: str) -> Optional[Path]:
    """Find `<root>/<folder>/<...>` regardless of extension or naming quirk."""
    rel = rel.replace("\\", "/")
    parts = rel.split("/")
    if parts[0] in ("photo", "sketch"):
        parts = parts[1:]
    stem = "/".join(parts)
    # sketches are named `sketchNNNN` where photos are `imageNNNN`
    candidates = [stem]
    if folder == "sketch":
        candidates.append(stem.replace("image", "sketch"))
    else:
        candidates.append(stem.replace("sketch", "image"))
    for cand in candidates:
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            p = root / folder / f"{cand}{ext}"
            if p.exists():
                return p
        hits = sorted((root / folder).glob(f"{cand}.*"))
        if hits:
            return hits[0]
    return None


def read_records(root: str | Path, split: str = "train") -> List[dict]:
    root = Path(root)
    anno = root / f"anno_{split}.json"
    if not anno.exists():
        raise FileNotFoundError(
            f"{anno} not found. Download FS2K and point --fs2k-root at the "
            "folder containing anno_train.json, anno_test.json, photo/ and sketch/.")
    with open(anno) as fh:
        raw = json.load(fh)

    records = []
    missing = 0
    for r in raw:
        photo = _resolve(root, r["image_name"], "photo")
        sketch = _resolve(root, r["image_name"], "sketch")
        if photo is None or sketch is None:
            missing += 1
            continue
        records.append({"photo": photo, "sketch": sketch, "style": int(r["style"])})
    if missing:
        print(f"[fs2k] warning: {missing} records skipped (file not found)")
    return records


def stratified_split(records: List[dict], val_fraction: float = VAL_FRACTION,
                     seed: int = SPLIT_SEED) -> Tuple[List[dict], List[dict]]:
    """Hold out `val_fraction` per style so all three styles stay represented."""
    rng = np.random.RandomState(seed)
    train, val = [], []
    for style in range(N_STYLES):
        idx = [i for i, r in enumerate(records) if r["style"] == style]
        order = rng.permutation(len(idx))
        n_val = int(round(val_fraction * len(idx)))
        for j, o in enumerate(order):
            (val if j < n_val else train).append(records[idx[o]])
    rng.shuffle(train)
    return train, val


class FS2KDataset(Dataset):
    """Returns (photo, sketch, style) with photo/sketch in [-1, 1]."""

    def __init__(self, records: List[dict], image_size: int = IMAGE_SIZE,
                 augment: bool = False):
        self.records = records
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, path: Path) -> Image.Image:
        with Image.open(path) as im:
            return im.convert("RGB").copy()

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        photo = self._load(rec["photo"])
        sketch = self._load(rec["sketch"])

        size = self.image_size
        if self.augment:
            # resize a little large, then take the SAME crop from both images
            up = int(size * 1.12)
            photo = TF.resize(photo, [up, up], TF.InterpolationMode.BICUBIC)
            sketch = TF.resize(sketch, [up, up], TF.InterpolationMode.BICUBIC)
            top = random.randint(0, up - size)
            left = random.randint(0, up - size)
            photo = TF.crop(photo, top, left, size, size)
            sketch = TF.crop(sketch, top, left, size, size)
            if random.random() < 0.5:
                photo, sketch = TF.hflip(photo), TF.hflip(sketch)
        else:
            photo = TF.resize(photo, [size, size], TF.InterpolationMode.BICUBIC)
            sketch = TF.resize(sketch, [size, size], TF.InterpolationMode.BICUBIC)

        photo_t = TF.to_tensor(photo) * 2.0 - 1.0
        sketch_t = TF.to_tensor(sketch) * 2.0 - 1.0
        return photo_t, sketch_t, int(rec["style"])


def build_datasets(root: str | Path, image_size: int = IMAGE_SIZE):
    train_recs = read_records(root, "train")
    test_recs = read_records(root, "test")
    train_recs, val_recs = stratified_split(train_recs)
    return (FS2KDataset(train_recs, image_size, augment=True),
            FS2KDataset(val_recs, image_size, augment=False),
            FS2KDataset(test_recs, image_size, augment=False))
