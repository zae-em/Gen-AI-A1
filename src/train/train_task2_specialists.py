"""Task 2, stage 2 - three specialist denoising autoencoders.

    python -m src.train.train_task2_specialists --experts salt_pepper blur occlusion

Each specialist sees only its own corruption, and each one gets its own set of
parameters.  The architecture is shared (one Optuna study, as the assignment
permits, to keep the compute budget sane) but the weights are trained
independently - `--config` supplies the shared architecture, the loop trains
one model per corruption.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from ..losses import ReconstructionLoss
from ..models.autoencoder import build_autoencoder
from ..utils.seed import pick_device, set_seed
from ..utils.tracking import ExperimentTracker
from ..utils.viz import comparison_grid
from .common import build_loaders, evaluate_reconstruction, save_checkpoint

EXPERT_KINDS = ("salt_pepper", "blur", "occlusion")

DEFAULT_CONFIG = {
    "base_channels": 32,
    "latent_dim": 128,
    "dropout": 0.1,
    "depth": 4,
    "image_size": 128,
    "use_skip": False,
    "lr": 2e-3,
    "batch_size": 64,
    "alpha": 0.8,
    "weight_decay": 1e-5,
}


def train_one(kind: str, config: dict, args, trial=None) -> dict:
    set_seed(args.seed)
    device = pick_device(args.device)

    train_loader, val_loader = build_loaders(
        args.data_root, args.val_manifest, config["batch_size"], config["image_size"],
        mode="forced", force_kind=kind, num_workers=args.num_workers)

    model = build_autoencoder(config).to(device)
    criterion = ReconstructionLoss(config["alpha"])
    optimiser = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    tracker = None
    if not args.no_tracking:
        tracker = ExperimentTracker(args.tracker, args.project,
                                    run_name=f"task2-specialist-{kind}",
                                    config={**config, "expert": kind},
                                    output_dir=args.out_dir)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best_score, best_metrics = -1e9, {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        bar = tqdm(train_loader, desc=f"{kind} epoch {epoch}/{args.epochs}",
                   disable=args.quiet)
        for corrupt, clean, _labels, _idx in bar:
            corrupt, clean = corrupt.to(device), clean.to(device)
            loss, _l1, _s = criterion(model(corrupt), clean)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            running += float(loss)
            if not args.quiet:
                bar.set_postfix(loss=f"{float(loss):.4f}")
        scheduler.step()

        metrics = evaluate_reconstruction(model, val_loader, device)
        metrics["train_loss"] = running / max(len(train_loader), 1)
        if tracker:
            tracker.log_metrics(metrics, step=epoch)
        if not args.quiet:
            print(f"  [{kind}] epoch {epoch}: PSNR {metrics['val_psnr']:.2f} dB | "
                  f"SSIM {metrics['val_ssim']:.4f}")

        if metrics["val_score"] > best_score:
            best_score, best_metrics = metrics["val_score"], dict(metrics)
            save_checkpoint(out_dir / f"task2_specialist_{kind}.pt", model, config,
                            {"epoch": epoch, "metrics": metrics, "expert": kind})

        if trial is not None:
            import optuna
            trial.report(metrics["val_score"], epoch)
            if trial.should_prune():
                if tracker:
                    tracker.finish()
                raise optuna.TrialPruned()

    corrupt, clean, _labels, _idx = next(iter(val_loader))
    model.eval()
    with torch.no_grad():
        recon = model(corrupt.to(device)).clamp(0, 1).cpu()
    grid = comparison_grid(clean, corrupt, recon, out_dir / f"task2_{kind}_samples.png")

    if tracker:
        tracker.log_image(grid, f"task2 {kind} samples")
        tracker.finish()
    return best_metrics


def build_argparser():
    ap = argparse.ArgumentParser(description="Task 2 specialist autoencoders")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--val-manifest", default="data/manifests/val_manifest.json")
    ap.add_argument("--config", default=None, help="shared architecture from Optuna")
    ap.add_argument("--out-dir", default="artifacts/task2")
    ap.add_argument("--experts", nargs="+", default=list(EXPERT_KINDS),
                    choices=list(EXPERT_KINDS))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--tracker", default="mlflow", choices=["mlflow", "wandb"])
    ap.add_argument("--project", default="genai-assignment-1")
    ap.add_argument("--no-tracking", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main():
    args = build_argparser().parse_args()
    config = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config) as fh:
            config.update(json.load(fh))

    summary = {}
    for kind in args.experts:
        print(f"\n=== specialist: {kind} ===")
        summary[kind] = train_one(kind, config, args)

    out = Path(args.out_dir) / "task2_specialists_summary.json"
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
