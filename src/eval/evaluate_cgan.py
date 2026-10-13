"""Task 4 evaluation on the official FS2K test split.

    python -m src.eval.evaluate_cgan --checkpoint artifacts/task4/task4_generator.pt

Reports L1, SSIM and PSNR overall and per style, and renders a per-style grid.
It also runs a style-swap check: the same photograph generated under all three
conditions.  If the three outputs are near-identical the conditioning is being
ignored, which is the single most common failure of this architecture and has
to be reported honestly if it happens.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.fs2k import STYLE_NAMES, build_datasets
from ..losses import ssim
from ..metrics import psnr
from ..models.pix2pix import build_generator
from ..train.common import load_checkpoint
from ..utils.seed import pick_device, set_seed
from ..utils.viz import sketch_grid


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/task4/task4_generator.pt")
    ap.add_argument("--fs2k-root", default="data/FS2K")
    ap.add_argument("--out-dir", default="artifacts/task4/eval")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    set_seed(42)
    device = pick_device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(args.checkpoint)
    generator = build_generator(ckpt["config"]).to(device).eval()
    generator.load_state_dict(ckpt["state_dict"])

    _train, _val, test_ds = build_datasets(args.fs2k_root, ckpt["config"]["image_size"])
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    stats = defaultdict(lambda: {"l1": 0.0, "ssim": 0.0, "psnr": 0.0, "n": 0})
    worst = []

    for photo, sketch, style in tqdm(loader, desc="task4 test"):
        photo, sketch, style = photo.to(device), sketch.to(device), style.to(device)
        fake = generator(photo, style)
        f01, s01 = (fake + 1) / 2, (sketch + 1) / 2

        l1 = (fake - sketch).abs().flatten(1).mean(1)
        s = ssim(f01, s01, size_average=False)
        p = psnr(f01.clamp(0, 1), s01)

        for i in range(photo.shape[0]):
            for key in (f"style{int(style[i]) + 1}", "overall"):
                stats[key]["l1"] += float(l1[i])
                stats[key]["ssim"] += float(s[i])
                stats[key]["psnr"] += float(p[i])
                stats[key]["n"] += 1
            worst.append((float(s[i]), int(style[i])))

    summary = {k: {"l1": v["l1"] / v["n"], "ssim": v["ssim"] / v["n"],
                   "psnr": v["psnr"] / v["n"], "n": v["n"]}
               for k, v in stats.items()}
    print(f"{'split':>10}{'L1':>9}{'SSIM':>9}{'PSNR':>9}{'n':>7}")
    for k in sorted(summary):
        v = summary[k]
        print(f"{k:>10}{v['l1']:>9.4f}{v['ssim']:>9.4f}{v['psnr']:>9.3f}{v['n']:>7d}")

    # per-style qualitative grid
    picks, seen = [], set()
    for i, rec in enumerate(test_ds.records):
        if rec["style"] not in seen:
            seen.add(rec["style"]); picks.append(i)
        if len(seen) == len(STYLE_NAMES):
            break
    batch = [test_ds[i] for i in picks]
    photos = torch.stack([b[0] for b in batch]).to(device)
    sketches = torch.stack([b[1] for b in batch])
    styles = torch.tensor([b[2] for b in batch], device=device)
    fake = generator(photos, styles).cpu()
    sketch_grid(photos.cpu(), sketches, fake, out_dir / "task4_per_style.png",
                styles=styles.cpu().tolist())

    # style-swap check on one photograph
    probe = photos[:1]
    swaps = [generator(probe, torch.tensor([s], device=device)).cpu()
             for s in range(len(STYLE_NAMES))]
    diffs = {f"style1_vs_style{s + 1}": float((swaps[0] - swaps[s]).abs().mean())
             for s in range(1, len(STYLE_NAMES))}
    print(f"\nstyle-swap mean absolute differences: {diffs}")
    sketch_grid(torch.cat([probe.cpu()] * len(STYLE_NAMES)),
                torch.cat([sketches[:1]] * len(STYLE_NAMES)),
                torch.cat(swaps), out_dir / "task4_style_swap.png",
                styles=list(range(len(STYLE_NAMES))))

    with open(out_dir / "task4_test_metrics.json", "w") as fh:
        json.dump({"summary": summary, "style_swap_differences": diffs}, fh, indent=2)
    print(f"\nwrote results to {out_dir}")


if __name__ == "__main__":
    main()
