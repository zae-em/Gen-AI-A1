"""Build the deterministic validation and test corruption manifests.

Run once, commit the JSON files (they are small - specs, not pixels):

    python -m src.data.manifest --root data/ --out data/manifests

Validation manifest: one corruption per image, classes cycled so the
validation set is exactly class balanced.

Test manifest: for every test image, one clean entry plus three fixed
severities for each of the three corruptions (10 entries per image).
Severities are the ones fixed by the assignment:
    salt & pepper  p = 0.03 / 0.08 / 0.15
    blur           (k, sigma) = (3, 0.7) / (5, 1.5) / (7, 2.5)
    occlusion      ~10% (1 rect) / ~20% (2 rects) / ~35% (3 rects)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .corruptions import (CORRUPTION_NAMES, SEVERITY_NAMES, coverage_of,
                          sample_test_spec)
from .datasets import IMAGE_SIZE, SPLIT_SEED, load_pet_paths, make_split


def build_val_manifest(paths, image_size: int = IMAGE_SIZE, seed: int = SPLIT_SEED):
    entries = []
    for i, _ in enumerate(paths):
        kind = CORRUPTION_NAMES[i % len(CORRUPTION_NAMES)]
        severity_idx = i % len(SEVERITY_NAMES)
        entry_seed = seed * 100003 + i
        gen = torch.Generator().manual_seed(entry_seed)
        spec = sample_test_spec(image_size, image_size, kind, severity_idx, gen)
        spec.seed = entry_seed
        entries.append({"image_index": i, "seed": entry_seed,
                        "corruption": kind,
                        "severity": spec.severity,
                        "spec": spec.to_dict()})
    return entries


def build_test_manifest(paths, image_size: int = IMAGE_SIZE, seed: int = SPLIT_SEED):
    entries = []
    for i, _ in enumerate(paths):
        entry_seed = seed * 100019 + i
        gen = torch.Generator().manual_seed(entry_seed)

        spec = sample_test_spec(image_size, image_size, "clean", 0, gen)
        spec.seed = entry_seed
        entries.append({"image_index": i, "seed": entry_seed, "corruption": "clean",
                        "severity": None, "spec": spec.to_dict()})

        for kind in ("salt_pepper", "blur", "occlusion"):
            for s_idx in range(len(SEVERITY_NAMES)):
                spec = sample_test_spec(image_size, image_size, kind, s_idx, gen)
                spec.seed = entry_seed
                rec = {"image_index": i, "seed": entry_seed, "corruption": kind,
                       "severity": SEVERITY_NAMES[s_idx], "spec": spec.to_dict()}
                if kind == "occlusion":
                    rec["realised_coverage"] = round(
                        coverage_of(spec.rects, image_size, image_size), 4)
                entries.append(rec)
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="torchvision dataset root")
    ap.add_argument("--out", default="data/manifests")
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _, val_paths = make_split(args.root, args.image_size)
    test_paths = load_pet_paths(args.root, "test")

    val = {"image_size": args.image_size, "split": "val",
           "paths": [str(p) for p in val_paths],
           "entries": build_val_manifest(val_paths, args.image_size)}
    test = {"image_size": args.image_size, "split": "test",
            "paths": [str(p) for p in test_paths],
            "entries": build_test_manifest(test_paths, args.image_size)}

    with open(out / "val_manifest.json", "w") as fh:
        json.dump(val, fh)
    with open(out / "test_manifest.json", "w") as fh:
        json.dump(test, fh)

    print(f"val  : {len(val['entries'])} entries over {len(val_paths)} images")
    print(f"test : {len(test['entries'])} entries over {len(test_paths)} images")


if __name__ == "__main__":
    main()
