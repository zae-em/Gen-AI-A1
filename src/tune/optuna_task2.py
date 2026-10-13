"""Optuna studies for Task 2.

Two separate studies:
  --stage classifier  learning rate, batch size, channel width, dropout,
                      weight decay; objective = validation macro-F1.
  --stage specialists one shared architecture search (the assignment allows
                      this to keep the compute feasible); objective = mean
                      combined score across the three corruptions.

    python -m src.tune.optuna_task2 --stage classifier --n-trials 25 --epochs 6
    python -m src.tune.optuna_task2 --stage specialists --n-trials 20 --epochs 6
"""
from __future__ import annotations

import argparse

from ..train.train_task2_classifier import DEFAULT_CONFIG as CLS_CONFIG
from ..train.train_task2_classifier import build_argparser as cls_args
from ..train.train_task2_classifier import train as train_classifier
from ..train.train_task2_specialists import DEFAULT_CONFIG as SPEC_CONFIG
from ..train.train_task2_specialists import train_one
from .base import run


def suggest_classifier(trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "base_channels": trial.suggest_categorical("base_channels", [16, 24, 32, 48]),
        "depth": trial.suggest_int("depth", 3, 5),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.15),
    }


def suggest_specialist(trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "latent_dim": trial.suggest_categorical("latent_dim", [64, 128, 256]),
        "base_channels": trial.suggest_categorical("base_channels", [16, 32, 48]),
        "alpha": trial.suggest_float("alpha", 0.3, 0.95),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
    }


def main():
    ap = argparse.ArgumentParser(parents=[cls_args()], add_help=False,
                                 conflict_handler="resolve")
    ap.add_argument("--stage", choices=["classifier", "specialists"],
                    default="classifier")
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--study-out", default="artifacts/optuna")
    ap.add_argument("--probe-expert", default="blur",
                    help="corruption used for the shared specialist search")
    args = ap.parse_args()
    args.no_tracking = True
    args.quiet = True
    args.out_dir = f"{args.study_out}/task2_trials"

    if args.stage == "classifier":
        def objective(trial):
            config = {**CLS_CONFIG, **suggest_classifier(trial)}
            report = train_classifier(config, args, trial=trial)
            return report["macro_f1"]
        run("task2_classifier", objective, args.n_trials, args.study_out)
    else:
        def objective(trial):
            config = {**SPEC_CONFIG, **suggest_specialist(trial)}
            # search on one representative corruption, then reuse the winning
            # architecture for all three specialists (documented compromise)
            best = train_one(args.probe_expert, config, args, trial=trial)
            return best["val_score"]
        run("task2_specialists", objective, args.n_trials, args.study_out)


if __name__ == "__main__":
    main()
