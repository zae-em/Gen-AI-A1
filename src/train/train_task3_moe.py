"""Task 3 - jointly trained soft mixture-of-experts restoration.

    python -m src.train.train_task3_moe \
        --classifier artifacts/task2/task2_classifier.pt \
        --specialist-dir artifacts/task2 --warmup-epochs 3 --epochs 25

Two stages, as required:
  warm-up   experts frozen, only the gate learns to produce useful weights
            under the *reconstruction* objective rather than pure
            classification;
  joint     everything unfrozen at a smaller learning rate.

Joint loss:
    L = l1_w * L1 + ssim_w * (1 - SSIM) + ce_w * CE(gate, true label)
        + bal_w * balance(w)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..data.corruptions import CORRUPTION_NAMES
from ..losses import balance_loss, routing_entropy, ssim
from ..models.autoencoder import build_autoencoder
from ..models.classifier import build_classifier
from ..models.moe import BRANCH_NAMES, SoftMoERestorer, branch_weight_report
from ..utils.seed import pick_device, set_seed
from ..utils.tracking import ExperimentTracker
from ..utils.viz import comparison_grid, routing_heatmap
from .common import (build_loaders, evaluate_reconstruction, load_checkpoint,
                     save_checkpoint)

EXPERT_KINDS = ("salt_pepper", "blur", "occlusion")

DEFAULT_CONFIG = {
    "temperature": 1.0,
    "l1_w": 0.8,
    "ssim_w": 0.2,
    "ce_w": 0.1,
    "bal_w": 0.01,
    "lr_warmup": 1e-3,
    "lr_joint": 1e-4,
    "batch_size": 32,
    "image_size": 128,
    "weight_decay": 1e-5,
}


def build_moe(args, config, device) -> SoftMoERestorer:
    """Assemble the MoE from the Task 2 checkpoints - never from scratch."""
    cls_ckpt = load_checkpoint(args.classifier)
    gate = build_classifier(cls_ckpt["config"])
    gate.load_state_dict(cls_ckpt["state_dict"])

    experts = []
    for kind in EXPERT_KINDS:
        path = Path(args.specialist_dir) / f"task2_specialist_{kind}.pt"
        ckpt = load_checkpoint(path)
        expert = build_autoencoder(ckpt["config"])
        expert.load_state_dict(ckpt["state_dict"])
        experts.append(expert)

    return SoftMoERestorer(gate, experts, config["temperature"]).to(device)


def joint_loss(recon, clean, logits, weights, labels, config):
    l1 = F.l1_loss(recon, clean)
    ssim_term = 1.0 - ssim(recon, clean)
    ce = nn.functional.cross_entropy(logits, labels)
    bal = balance_loss(weights)
    total = (config["l1_w"] * l1 + config["ssim_w"] * ssim_term
             + config["ce_w"] * ce + config["bal_w"] * bal)
    return total, {"l1": float(l1), "ssim_term": float(ssim_term),
                   "ce": float(ce), "balance": float(bal)}


def run_epoch(model, loader, optimiser, config, device, desc, quiet):
    model.train()
    totals = {"loss": 0.0, "l1": 0.0, "ssim_term": 0.0, "ce": 0.0,
              "balance": 0.0, "entropy": 0.0}
    batches = 0
    bar = tqdm(loader, desc=desc, disable=quiet)
    for corrupt, clean, labels, _idx in bar:
        corrupt, clean, labels = corrupt.to(device), clean.to(device), labels.to(device)
        recon, weights, logits = model(corrupt)
        loss, parts = joint_loss(recon, clean, logits, weights, labels, config)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimiser.step()

        totals["loss"] += float(loss)
        for k, v in parts.items():
            totals[k] += v
        totals["entropy"] += float(routing_entropy(weights.detach()))
        batches += 1
        if not quiet:
            bar.set_postfix(loss=f"{float(loss):.4f}", ce=f"{parts['ce']:.3f}")
    return {k: v / max(batches, 1) for k, v in totals.items()}


def train(config: dict, args, trial=None) -> dict:
    set_seed(args.seed)
    device = pick_device(args.device)

    train_loader, val_loader = build_loaders(
        args.data_root, args.val_manifest, config["batch_size"],
        config["image_size"], mode="balanced", num_workers=args.num_workers)

    model = build_moe(args, config, device)
    tracker = None
    if not args.no_tracking:
        tracker = ExperimentTracker(args.tracker, args.project,
                                    run_name=args.run_name or "task3-soft-moe",
                                    config=config, output_dir=args.out_dir)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_psnr": [], "val_ssim": [], "routing_entropy": []}
    best_score, best_metrics = -1e9, {}
    step = 0

    # ---- stage 1: warm-up, experts frozen ---------------------------------
    model.set_expert_grad(False)
    warm_opt = torch.optim.AdamW(model.gate.parameters(), lr=config["lr_warmup"],
                                 weight_decay=config["weight_decay"])
    for epoch in range(1, args.warmup_epochs + 1):
        step += 1
        stats = run_epoch(model, train_loader, warm_opt, config, device,
                          f"warmup {epoch}/{args.warmup_epochs}", args.quiet)
        metrics = evaluate_reconstruction(model, val_loader, device)
        metrics.update({f"train_{k}": v for k, v in stats.items()})
        metrics["stage"] = 0
        if tracker:
            tracker.log_metrics(metrics, step=step)
        if not args.quiet:
            print(f"  warmup {epoch}: PSNR {metrics['val_psnr']:.2f} | "
                  f"SSIM {metrics['val_ssim']:.4f} | H {stats['entropy']:.3f}")

    # ---- stage 2: joint fine-tuning ---------------------------------------
    model.set_expert_grad(True)
    joint_opt = torch.optim.AdamW(model.parameters(), lr=config["lr_joint"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(joint_opt, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        step += 1
        stats = run_epoch(model, train_loader, joint_opt, config, device,
                          f"joint {epoch}/{args.epochs}", args.quiet)
        scheduler.step()

        metrics = evaluate_reconstruction(model, val_loader, device)
        metrics.update({f"train_{k}": v for k, v in stats.items()})
        metrics["stage"] = 1

        history["train_loss"].append(stats["loss"])
        history["val_psnr"].append(metrics["val_psnr"])
        history["val_ssim"].append(metrics["val_ssim"])
        history["routing_entropy"].append(stats["entropy"])
        if tracker:
            tracker.log_metrics(metrics, step=step)
        if not args.quiet:
            print(f"  joint {epoch}: PSNR {metrics['val_psnr']:.2f} | "
                  f"SSIM {metrics['val_ssim']:.4f} | H {stats['entropy']:.3f}")

        if metrics["val_score"] > best_score:
            best_score, best_metrics = metrics["val_score"], dict(metrics)
            save_checkpoint(out_dir / "task3_soft_moe.pt", model, config,
                            {"epoch": epoch, "metrics": metrics,
                             "temperature": config["temperature"]})

        if trial is not None:
            import optuna
            trial.report(metrics["val_score"], step)
            if trial.should_prune():
                if tracker:
                    tracker.finish()
                raise optuna.TrialPruned()

    # ---- routing analysis --------------------------------------------------
    weights_by_class = branch_weight_report(model, val_loader, device)
    heat = routing_heatmap(weights_by_class, out_dir / "task3_routing_heatmap.png",
                           BRANCH_NAMES)

    dead = [BRANCH_NAMES[i] for i in range(len(BRANCH_NAMES))
            if max(w[i] for w in weights_by_class.values()) < 0.10]
    analysis = {"mean_weights_by_true_corruption": weights_by_class,
                "possibly_inactive_branches": dead,
                "best_metrics": best_metrics, "history": history}
    with open(out_dir / "task3_routing_analysis.json", "w") as fh:
        json.dump(analysis, fh, indent=2)

    corrupt, clean, _labels, _idx = next(iter(val_loader))
    model.eval()
    with torch.no_grad():
        recon, _w, _l = model(corrupt.to(device))
    comparison_grid(clean, corrupt, recon.clamp(0, 1).cpu(),
                    out_dir / "task3_samples.png")

    if tracker:
        tracker.log_image(heat, "task3 routing heatmap")
        tracker.log_artifact(out_dir / "task3_routing_analysis.json")
        tracker.finish()

    print("\nmean routing weight by true corruption:")
    print("  " + " ".join(f"{b:>12}" for b in BRANCH_NAMES))
    for kind in CORRUPTION_NAMES:
        if kind in weights_by_class:
            row = weights_by_class[kind]
            print(f"{kind:>12} " + " ".join(f"{v:>12.3f}" for v in row))
    if dead:
        print(f"\nWARNING: branches never exceeding 0.10 mean weight: {dead}")
    return best_metrics


def build_argparser():
    ap = argparse.ArgumentParser(description="Task 3 soft mixture-of-experts")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--val-manifest", default="data/manifests/val_manifest.json")
    ap.add_argument("--classifier", default="artifacts/task2/task2_classifier.pt")
    ap.add_argument("--specialist-dir", default="artifacts/task2")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="artifacts/task3")
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=25)
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
    train(config, args)


if __name__ == "__main__":
    main()
