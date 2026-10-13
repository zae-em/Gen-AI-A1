"""Optuna study for Task 1.

Searches the learning rate, batch size, bottleneck dimension, encoder channel
width, dropout and the L1/SSIM weight alpha - the six the assignment names -
plus weight decay.  The objective is the combined validation score
(SSIM + PSNR/50), maximised.

    python -m src.tune.optuna_task1 --n-trials 30 --epochs 8
"""
from __future__ import annotations

import argparse

from ..train.train_task1 import DEFAULT_CONFIG, build_argparser, train
from .base import run


def suggest(trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "latent_dim": trial.suggest_categorical("latent_dim", [32, 64, 128, 256]),
        "base_channels": trial.suggest_categorical("base_channels", [16, 32, 48, 64]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "alpha": trial.suggest_float("alpha", 0.3, 0.95),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }


def main():
    ap = argparse.ArgumentParser(parents=[build_argparser()], add_help=False,
                                 conflict_handler="resolve")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--study-out", default="artifacts/optuna")
    args = ap.parse_args()
    args.no_tracking = True      # per-trial runs would flood the tracker
    args.quiet = True
    args.out_dir = f"{args.study_out}/task1_trials"

    def objective(trial):
        config = {**DEFAULT_CONFIG, **suggest(trial)}
        best = train(config, args, trial=trial)
        return best["val_score"]

    run("task1_universal_ae", objective, args.n_trials, args.study_out)


if __name__ == "__main__":
    main()
