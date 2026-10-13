"""Optuna study for Task 3 joint fine-tuning.

Searches the joint learning rate, gate temperature, classification weight,
balance weight and the L1/SSIM split.  Trials that collapse the routing are
pruned by the objective itself: if any branch falls below a floor of mean
weight, the trial is reported as failed rather than being allowed to win on
reconstruction alone.

    python -m src.tune.optuna_task3 --n-trials 20 --epochs 5 --warmup-epochs 1
"""
from __future__ import annotations

import argparse

from ..train.train_task3_moe import DEFAULT_CONFIG, build_argparser, train
from .base import run


def suggest(trial) -> dict:
    l1_w = trial.suggest_float("l1_w", 0.4, 1.0)
    return {
        "lr_joint": trial.suggest_float("lr_joint", 1e-5, 5e-4, log=True),
        "lr_warmup": trial.suggest_float("lr_warmup", 1e-4, 3e-3, log=True),
        "temperature": trial.suggest_float("temperature", 0.3, 3.0, log=True),
        "ce_w": trial.suggest_float("ce_w", 0.0, 0.5),
        "bal_w": trial.suggest_float("bal_w", 1e-4, 0.5, log=True),
        "l1_w": l1_w,
        "ssim_w": 1.0 - l1_w,
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }


def main():
    ap = argparse.ArgumentParser(parents=[build_argparser()], add_help=False,
                                 conflict_handler="resolve")
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--study-out", default="artifacts/optuna")
    args = ap.parse_args()
    args.no_tracking = True
    args.quiet = True
    args.out_dir = f"{args.study_out}/task3_trials"

    def objective(trial):
        config = {**DEFAULT_CONFIG, **suggest(trial)}
        best = train(config, args, trial=trial)
        return best["val_score"]

    run("task3_soft_moe", objective, args.n_trials, args.study_out)


if __name__ == "__main__":
    main()
