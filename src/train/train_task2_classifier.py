"""Task 2, stage 1 - four-class corruption classifier.

    python -m src.train.train_task2_classifier --epochs 30

Batches are class balanced by construction: the dataset is indexed as
(image, corruption) pairs so every image appears once per corruption class in
an epoch.  That is cheaper and more exact than a WeightedRandomSampler, and it
removes the bias the assignment warns about.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..data.corruptions import CORRUPTION_NAMES
from ..metrics import classification_report_dict
from ..models.classifier import build_classifier
from ..utils.seed import pick_device, set_seed
from ..utils.tracking import ExperimentTracker
from ..utils.viz import confusion_matrix_figure, curve_figure
from .common import build_loaders, save_checkpoint

DEFAULT_CONFIG = {
    "base_channels": 32,
    "dropout": 0.3,
    "depth": 4,
    "image_size": 128,
    "lr": 1e-3,
    "batch_size": 64,
    "weight_decay": 1e-4,
    "label_smoothing": 0.0,
}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for corrupt, _clean, labels, _idx in loader:
        logits = model(corrupt.to(device))
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(labels.tolist())
    return classification_report_dict(y_true, y_pred,
                                      labels=list(range(len(CORRUPTION_NAMES))),
                                      target_names=list(CORRUPTION_NAMES))


def train(config: dict, args, trial=None) -> dict:
    set_seed(args.seed)
    device = pick_device(args.device)

    train_loader, val_loader = build_loaders(
        args.data_root, args.val_manifest, config["batch_size"],
        config["image_size"], mode="balanced", num_workers=args.num_workers)

    model = build_classifier(config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))
    optimiser = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    tracker = None
    if not args.no_tracking:
        tracker = ExperimentTracker(args.tracker, args.project,
                                    run_name=args.run_name or "task2-classifier",
                                    config=config, output_dir=args.out_dir)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_accuracy": [], "val_macro_f1": []}
    best_f1, best_report = -1.0, {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, correct, seen = 0.0, 0, 0
        bar = tqdm(train_loader, desc=f"task2-cls epoch {epoch}/{args.epochs}",
                   disable=args.quiet)
        for corrupt, _clean, labels, _idx in bar:
            corrupt, labels = corrupt.to(device), labels.to(device)
            logits = model(corrupt)
            loss = criterion(logits, labels)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

            running += float(loss)
            correct += int((logits.argmax(1) == labels).sum())
            seen += labels.numel()
            if not args.quiet:
                bar.set_postfix(loss=f"{float(loss):.4f}", acc=f"{correct / seen:.3f}")
        scheduler.step()

        report = evaluate(model, val_loader, device)
        metrics = {"train_loss": running / max(len(train_loader), 1),
                   "train_accuracy": correct / max(seen, 1),
                   "val_accuracy": report["accuracy"],
                   "val_macro_f1": report["macro_f1"],
                   "val_macro_precision": report["macro_precision"],
                   "val_macro_recall": report["macro_recall"]}
        for name in CORRUPTION_NAMES:
            metrics[f"val_f1_{name}"] = report["report"][name]["f1-score"]

        history["train_loss"].append(metrics["train_loss"])
        history["val_accuracy"].append(metrics["val_accuracy"])
        history["val_macro_f1"].append(metrics["val_macro_f1"])
        if tracker:
            tracker.log_metrics(metrics, step=epoch)
        if not args.quiet:
            print(f"  epoch {epoch}: acc {report['accuracy']:.4f} | "
                  f"macro-F1 {report['macro_f1']:.4f}")

        if report["macro_f1"] > best_f1:
            best_f1, best_report = report["macro_f1"], report
            save_checkpoint(out_dir / "task2_classifier.pt", model, config,
                            {"epoch": epoch, "report": report})

        if trial is not None:
            import optuna
            trial.report(report["macro_f1"], epoch)
            if trial.should_prune():
                if tracker:
                    tracker.finish()
                raise optuna.TrialPruned()

    cm_fig = confusion_matrix_figure(best_report["confusion_matrix_normalised"],
                                     CORRUPTION_NAMES,
                                     out_dir / "task2_confusion_matrix.png")
    curve_figure({k: v for k, v in history.items() if k != "train_loss"},
                 out_dir / "task2_classifier_curves.png", ylabel="validation metric")
    with open(out_dir / "task2_classifier_report.json", "w") as fh:
        json.dump(best_report, fh, indent=2)

    if tracker:
        tracker.log_image(cm_fig, "task2 confusion matrix")
        tracker.log_artifact(out_dir / "task2_classifier_report.json")
        tracker.finish()

    print("\nnormalised confusion matrix (rows = true):")
    for name, row in zip(CORRUPTION_NAMES, best_report["confusion_matrix_normalised"]):
        print(f"  {name:>12}: " + "  ".join(f"{v:.3f}" for v in row))
    return best_report


def build_argparser():
    ap = argparse.ArgumentParser(description="Task 2 corruption classifier")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--val-manifest", default="data/manifests/val_manifest.json")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="artifacts/task2")
    ap.add_argument("--epochs", type=int, default=30)
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
