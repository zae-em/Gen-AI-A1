"""Task 4 - style-conditioned face-to-sketch conditional GAN.

    python -m src.train.train_task4_cgan --fs2k-root data/FS2K --epochs 200

Losses are logged separately (D real, D fake, G adversarial, G L1) because a
single combined GAN loss curve tells you almost nothing about whether training
is healthy.  The same fixed validation photographs are rendered every
`--sample-every` epochs so the generator's progress can be watched over time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.fs2k import build_datasets
from ..losses import ssim
from ..models.pix2pix import (build_discriminator, build_generator, init_weights)
from ..utils.seed import pick_device, set_seed
from ..utils.tracking import ExperimentTracker
from ..utils.viz import curve_figure, sketch_grid
from .common import save_checkpoint

DEFAULT_CONFIG = {
    "base_channels": 64,
    "style_dim": 32,
    "dropout": 0.5,
    "lr_g": 2e-4,
    "lr_d": 2e-4,
    "beta1": 0.5,
    "beta2": 0.999,
    "batch_size": 16,
    "lambda_l1": 100.0,
    "image_size": 128,
    "label_smoothing": 0.9,     # one-sided smoothing on the real label
}


@torch.no_grad()
def validate(generator, loader, device):
    generator.eval()
    l1_total, ssim_total, n = 0.0, 0.0, 0
    for photo, sketch, style in loader:
        photo, sketch, style = photo.to(device), sketch.to(device), style.to(device)
        fake = generator(photo, style)
        l1_total += float(F.l1_loss(fake, sketch, reduction="sum")) / fake[0].numel()
        # SSIM wants [0, 1]
        ssim_total += float(ssim((fake + 1) / 2, (sketch + 1) / 2,
                                 size_average=False).sum())
        n += photo.shape[0]
    return {"val_l1": l1_total / max(n, 1), "val_ssim": ssim_total / max(n, 1)}


def train(config: dict, args, trial=None) -> dict:
    set_seed(args.seed)
    device = pick_device(args.device)

    train_ds, val_ds, _test_ds = build_datasets(args.fs2k_root, config["image_size"])
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                              num_workers=args.num_workers, drop_last=True,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"train {len(train_ds)} | val {len(val_ds)} pairs")

    generator = build_generator(config).to(device)
    discriminator = build_discriminator(config).to(device)
    init_weights(generator); init_weights(discriminator)

    opt_g = torch.optim.Adam(generator.parameters(), lr=config["lr_g"],
                             betas=(config["beta1"], config["beta2"]))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config["lr_d"],
                             betas=(config["beta1"], config["beta2"]))
    adversarial = nn.BCEWithLogitsLoss()

    tracker = None
    if not args.no_tracking:
        tracker = ExperimentTracker(args.tracker, args.project,
                                    run_name=args.run_name or "task4-cgan",
                                    config=config, output_dir=args.out_dir)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_dir / "samples"; sample_dir.mkdir(exist_ok=True)

    # one fixed validation batch, so progress panels are comparable over time
    fixed_photo, fixed_sketch, fixed_style = next(iter(val_loader))
    fixed_photo, fixed_style = fixed_photo.to(device), fixed_style.to(device)

    history = {"d_real": [], "d_fake": [], "g_adv": [], "g_l1": [], "val_l1": []}
    best_l1, best_metrics = 1e9, {}

    for epoch in range(1, args.epochs + 1):
        generator.train(); discriminator.train()
        sums = {"d_real": 0.0, "d_fake": 0.0, "g_adv": 0.0, "g_l1": 0.0}
        batches = 0

        bar = tqdm(train_loader, desc=f"task4 epoch {epoch}/{args.epochs}",
                   disable=args.quiet)
        for photo, sketch, style in bar:
            photo, sketch = photo.to(device), sketch.to(device)
            style = style.to(device)
            fake = generator(photo, style)

            # --- discriminator ---------------------------------------------
            pred_real = discriminator(photo, style, sketch)
            pred_fake = discriminator(photo, style, fake.detach())
            real_target = torch.full_like(pred_real, config["label_smoothing"])
            loss_d_real = adversarial(pred_real, real_target)
            loss_d_fake = adversarial(pred_fake, torch.zeros_like(pred_fake))
            loss_d = 0.5 * (loss_d_real + loss_d_fake)

            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # --- generator --------------------------------------------------
            pred_fake_for_g = discriminator(photo, style, fake)
            loss_g_adv = adversarial(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
            loss_g_l1 = F.l1_loss(fake, sketch)
            loss_g = loss_g_adv + config["lambda_l1"] * loss_g_l1

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            sums["d_real"] += float(loss_d_real)
            sums["d_fake"] += float(loss_d_fake)
            sums["g_adv"] += float(loss_g_adv)
            sums["g_l1"] += float(loss_g_l1)
            batches += 1
            if not args.quiet:
                bar.set_postfix(d=f"{float(loss_d):.3f}", g_adv=f"{float(loss_g_adv):.3f}",
                                g_l1=f"{float(loss_g_l1):.3f}")

        stats = {k: v / max(batches, 1) for k, v in sums.items()}
        val_metrics = validate(generator, val_loader, device)
        metrics = {**stats, **val_metrics}

        for k in ("d_real", "d_fake", "g_adv", "g_l1"):
            history[k].append(stats[k])
        history["val_l1"].append(val_metrics["val_l1"])

        if tracker:
            tracker.log_metrics(metrics, step=epoch)
        if not args.quiet:
            print(f"  epoch {epoch}: D(real) {stats['d_real']:.3f} "
                  f"D(fake) {stats['d_fake']:.3f} G(adv) {stats['g_adv']:.3f} "
                  f"G(L1) {stats['g_l1']:.4f} | val L1 {val_metrics['val_l1']:.4f}")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            generator.eval()
            with torch.no_grad():
                fake_fixed = generator(fixed_photo, fixed_style).cpu()
            path = sketch_grid(fixed_photo.cpu(), fixed_sketch, fake_fixed,
                               sample_dir / f"epoch_{epoch:04d}.png",
                               styles=fixed_style.cpu().tolist())
            if tracker:
                tracker.log_image(path, f"epoch {epoch}")

        if val_metrics["val_l1"] < best_l1:
            best_l1, best_metrics = val_metrics["val_l1"], dict(metrics)
            save_checkpoint(out_dir / "task4_generator.pt", generator, config,
                            {"epoch": epoch, "metrics": metrics})
            save_checkpoint(out_dir / "task4_discriminator.pt", discriminator, config,
                            {"epoch": epoch})

        if trial is not None:
            import optuna
            trial.report(val_metrics["val_l1"], epoch)
            if trial.should_prune():
                if tracker:
                    tracker.finish()
                raise optuna.TrialPruned()

    curve_figure({k: history[k] for k in ("d_real", "d_fake", "g_adv")},
                 out_dir / "task4_adversarial_curves.png", ylabel="loss")
    curve_figure({"g_l1": history["g_l1"], "val_l1": history["val_l1"]},
                 out_dir / "task4_l1_curves.png", ylabel="L1")
    with open(out_dir / "task4_history.json", "w") as fh:
        json.dump({"history": history, "best": best_metrics}, fh, indent=2)

    if tracker:
        tracker.log_artifact(out_dir / "task4_history.json")
        tracker.finish()
    return best_metrics


def build_argparser():
    ap = argparse.ArgumentParser(description="Task 4 face-to-sketch cGAN")
    ap.add_argument("--fs2k-root", default="data/FS2K")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="artifacts/task4")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--sample-every", type=int, default=10)
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
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
