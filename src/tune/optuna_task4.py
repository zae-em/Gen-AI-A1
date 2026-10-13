"""Optuna study for Task 4.

Searches both learning rates, batch size, base channel count, dropout, style
embedding dimension and lambda_L1.  GAN trials are expensive, so the study runs
with a short epoch budget (--epochs) and the winning configuration is then
retrained for the full schedule with train_task4_cgan.py.

Objective: validation L1 between generated and ground-truth sketch, minimised.
L1 is used rather than the adversarial loss because a discriminator loss can
look excellent while the generator produces nothing resembling the target.

    python -m src.tune.optuna_task4 --n-trials 15 --epochs 12
"""
from __future__ import annotations

import argparse

from ..train.train_task4_cgan import DEFAULT_CONFIG, build_argparser, train
from .base import run


def suggest(trial) -> dict:
    return {
        "lr_g": trial.suggest_float("lr_g", 5e-5, 5e-4, log=True),
        "lr_d": trial.suggest_float("lr_d", 5e-5, 5e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "base_channels": trial.suggest_categorical("base_channels", [32, 48, 64]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "style_dim": trial.suggest_categorical("style_dim", [8, 16, 32, 64]),
        "lambda_l1": trial.suggest_float("lambda_l1", 20.0, 200.0, log=True),
        "beta1": trial.suggest_float("beta1", 0.3, 0.9),
    }


def main():
    ap = argparse.ArgumentParser(parents=[build_argparser()], add_help=False,
                                 conflict_handler="resolve")
    ap.add_argument("--n-trials", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--study-out", default="artifacts/optuna")
    args = ap.parse_args()
    args.no_tracking = True
    args.quiet = True
    args.out_dir = f"{args.study_out}/task4_trials"
    args.sample_every = 10 ** 6      # no sample grids during the search

    def objective(trial):
        config = {**DEFAULT_CONFIG, **suggest(trial)}
        best = train(config, args, trial=trial)
        return best["val_l1"]

    run("task4_cgan", objective, args.n_trials, args.study_out, direction="minimize")


if __name__ == "__main__":
    main()
