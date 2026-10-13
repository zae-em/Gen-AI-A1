"""Task 1 - universal multi-corruption denoising autoencoder.

    python -m src.train.train_task1 \
        --data-root data --val-manifest data/manifests/val_manifest.json \
        --epochs 60 --config configs/task1_best.json

`--config` is the JSON that `src/tune/optuna_task1.py` writes for the best
trial.  Without it the script falls back to the assignment's suggested
starting point, which is fine for a smoke test but should not be what ends up
in the report.
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
from ..utils.viz import comparison_grid, curve_figure
from .common import build_loaders, evaluate_reconstruction, save_checkpoint

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


def train(config: dict, args, trial=None) -> dict:
    """Returns the best validation metrics. Shared with the Optuna objective."""
    set_seed(args.seed)
    device = pick_device(args.device)

    train_loader, val_loader = build_loaders(
        args.data_root, args.val_manifest, config["batch_size"],
        config["image_size"], mode="random", num_workers=args.num_workers)

    model = build_autoencoder(config).to(device)
    criterion = ReconstructionLoss(config["alpha"])
    optimiser = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    tracker = None
    if not args.no_tracking:
        tracker = ExperimentTracker(args.tracker, args.project,
                                    run_name=args.run_name or "task1-universal-ae",
                                    config=config, output_dir=args.out_dir)
        tracker.log_params({"compression_ratio": round(model.compression_ratio(), 3),
                            "latent_spatial": model.latent_spatial(),
                            "parameters": sum(p.numel() for p in model.parameters())})

    history = {"train_loss": [], "val_psnr": [], "val_ssim": []}
    best_score, best_metrics = -1e9, {}
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        bar = tqdm(train_loader, desc=f"task1 epoch {epoch}/{args.epochs}",
                   disable=args.quiet)
        for corrupt, clean, _labels, _idx in bar:
            corrupt, clean = corrupt.to(device), clean.to(device)
            recon = model(corrupt)
            loss, l1_val, ssim_val = criterion(recon, clean)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            running += float(loss)
            if not args.quiet:
                bar.set_postfix(loss=f"{float(loss):.4f}", l1=f"{float(l1_val):.4f}",
                                ssim_term=f"{float(ssim_val):.4f}")
        scheduler.step()

        train_loss = running / max(len(train_loader), 1)
        metrics = evaluate_reconstruction(model, val_loader, device)
        metrics["train_loss"] = train_loss
        metrics["lr"] = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["val_psnr"].append(metrics["val_psnr"])
        history["val_ssim"].append(metrics["val_ssim"])
        if tracker:
            tracker.log_metrics(metrics, step=epoch)
        if not args.quiet:
            print(f"  epoch {epoch}: loss {train_loss:.4f} | "
                  f"PSNR {metrics['val_psnr']:.2f} dB | SSIM {metrics['val_ssim']:.4f}")

        if metrics["val_score"] > best_score:
            best_score, best_metrics = metrics["val_score"], dict(metrics)
            save_checkpoint(out_dir / "task1_universal_ae.pt", model, config,
                            {"epoch": epoch, "metrics": metrics})

        if trial is not None:                      # Optuna pruning hook
            import optuna
            trial.report(metrics["val_score"], epoch)
            if trial.should_prune():
                if tracker:
                    tracker.finish()
                raise optuna.TrialPruned()

    # qualitative panel from the fixed validation manifest
    corrupt, clean, _labels, _idx = next(iter(val_loader))
    model.eval()
    with torch.no_grad():
        recon = model(corrupt.to(device)).clamp(0, 1).cpu()
    grid = comparison_grid(clean, corrupt, recon, out_dir / "task1_samples.png")
    curves = curve_figure({k: v for k, v in history.items() if k != "train_loss"},
                          out_dir / "task1_val_curves.png", ylabel="validation metric")
    curve_figure({"train_loss": history["train_loss"]},
                 out_dir / "task1_train_loss.png")

    if tracker:
        tracker.log_image(grid, "task1 samples")
        tracker.log_image(curves, "task1 validation curves")
        tracker.log_artifact(out_dir / "task1_universal_ae.pt")
        tracker.finish()

    with open(out_dir / "task1_history.json", "w") as fh:
        json.dump({"history": history, "best": best_metrics}, fh, indent=2)
    return best_metrics


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Task 1 universal denoising autoencoder")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--val-manifest", default="data/manifests/val_manifest.json")
    ap.add_argument("--config", default=None, help="JSON produced by the Optuna study")
    ap.add_argument("--out-dir", default="artifacts/task1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--tracker", default="mlflow", choices=["mlflow", "wandb"])
    ap.add_argument("--project", default="genai-assignment-1")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-tracking", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main():
    args = build_argparser().parse_args()
    config = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config) as fh:
            config.update(json.load(fh))
    best = train(config, args)
    print("\nbest validation metrics:")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
