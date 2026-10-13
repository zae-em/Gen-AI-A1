"""Task 1 final evaluation on the untouched official test split.

    python -m src.eval.evaluate_restoration \
        --checkpoint artifacts/task1/task1_universal_ae.pt \
        --test-manifest data/manifests/test_manifest.json

Produces:
  * a PSNR/SSIM table broken down by corruption *and* severity;
  * a 12-row qualitative panel (clean / corrupted / restored / |error|);
  * the four worst reconstructions, which are the failure cases the report
    has to discuss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.datasets import dataset_from_manifest
from ..losses import ssim
from ..metrics import RestorationAccumulator, psnr
from ..models.autoencoder import build_autoencoder
from ..train.common import load_checkpoint
from ..utils.seed import pick_device, set_seed
from ..utils.viz import comparison_grid


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=64, num_workers=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    acc = RestorationAccumulator()
    worst = []          # (score, index) - kept small with a running trim
    best_examples = []

    for corrupt, clean, _labels, idx in tqdm(loader, desc="evaluating"):
        corrupt, clean = corrupt.to(device), clean.to(device)
        recon = model(corrupt)
        if isinstance(recon, tuple):
            recon = recon[0]
        recon = recon.clamp(0, 1)

        entries = [dataset.manifest[int(i)] for i in idx]
        acc.update(recon, clean,
                   [e["corruption"] for e in entries],
                   [e.get("severity") for e in entries])

        s = ssim(recon, clean, size_average=False).cpu()
        p = psnr(recon, clean).cpu()
        for i in range(clean.shape[0]):
            worst.append((float(s[i]), float(p[i]), int(idx[i])))
        worst.sort(key=lambda t: t[0])
        worst = worst[:64]

    return acc, worst


def render_examples(model, dataset, indices, device, out_path, titles=None):
    batch = [dataset[i] for i in indices]
    corrupt = torch.stack([b[0] for b in batch])
    clean = torch.stack([b[1] for b in batch])
    with torch.no_grad():
        recon = model(corrupt.to(device))
        if isinstance(recon, tuple):
            recon = recon[0]
        recon = recon.clamp(0, 1).cpu()
    return comparison_grid(clean, corrupt, recon, out_path, titles=titles,
                           max_rows=len(indices))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/task1/task1_universal_ae.pt")
    ap.add_argument("--test-manifest", default="data/manifests/test_manifest.json")
    ap.add_argument("--out-dir", default="artifacts/task1/eval")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    set_seed(42)
    device = pick_device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(args.checkpoint)
    model = build_autoencoder(ckpt["config"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])

    dataset = dataset_from_manifest(args.test_manifest)
    acc, worst = evaluate(model, dataset, device, args.batch_size, args.num_workers)

    print("\n" + acc.as_table())
    summary = acc.summary()
    with open(out_dir / "task1_test_metrics.json", "w") as fh:
        json.dump({"config": ckpt["config"], "metrics": summary,
                   "compression_ratio": model.compression_ratio()}, fh, indent=2)

    # twelve representative examples: spread across corruption and severity
    picks, seen = [], set()
    for i, entry in enumerate(dataset.manifest):
        key = (entry["corruption"], entry.get("severity"))
        if key not in seen:
            seen.add(key); picks.append(i)
        if len(picks) >= 12:
            break
    titles = [f"{dataset.manifest[i]['corruption']}/{dataset.manifest[i].get('severity')}"
              for i in picks]
    render_examples(model, dataset, picks, device,
                    out_dir / "task1_representative.png", titles)

    # four worst cases by SSIM
    fail_idx = [w[2] for w in worst[:4]]
    fail_titles = [f"{dataset.manifest[i]['corruption']}/"
                   f"{dataset.manifest[i].get('severity')}" for i in fail_idx]
    render_examples(model, dataset, fail_idx, device,
                    out_dir / "task1_failures.png", fail_titles)

    with open(out_dir / "task1_failure_cases.json", "w") as fh:
        json.dump([{"index": i, "ssim": s, "psnr": p,
                    "entry": dataset.manifest[i]} for s, p, i in worst[:12]],
                  fh, indent=2)
    print(f"\nwrote results to {out_dir}")


if __name__ == "__main__":
    main()
