"""Shared Optuna scaffolding.

Every study uses a SQLite storage file so a run can be interrupted and
resumed, and a MedianPruner so that obviously bad configurations stop early
instead of burning the epoch budget.  `summarise` writes both the best
configuration (consumed by the training scripts via --config) and a trials
table for the report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import optuna


def make_study(name: str, storage_dir: str | Path = "optuna",
               direction: str = "maximize", seed: int = 42,
               n_startup_trials: int = 5, n_warmup_steps: int = 3) -> optuna.Study:
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{storage_dir / 'studies.db'}",
        direction=direction,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials,
                                           n_warmup_steps=n_warmup_steps),
    )


def summarise(study: optuna.Study, out_dir: str | Path, name: str) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best = study.best_trial
    with open(out_dir / f"{name}_best_config.json", "w") as fh:
        json.dump(best.params, fh, indent=2)

    rows = []
    for t in study.trials:
        rows.append({"number": t.number, "state": str(t.state),
                     "value": t.value, "params": t.params})
    with open(out_dir / f"{name}_trials.json", "w") as fh:
        json.dump({"n_trials": len(study.trials),
                   "n_pruned": sum(1 for t in study.trials
                                   if t.state == optuna.trial.TrialState.PRUNED),
                   "n_complete": sum(1 for t in study.trials
                                     if t.state == optuna.trial.TrialState.COMPLETE),
                   "best_value": best.value, "best_params": best.params,
                   "trials": rows}, fh, indent=2)

    try:
        import optuna.visualization.matplotlib as ovm
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for fn, label in ((ovm.plot_optimization_history, "history"),
                          (ovm.plot_param_importances, "importances")):
            try:
                fn(study)
                plt.tight_layout()
                plt.savefig(out_dir / f"{name}_{label}.png", dpi=150,
                            bbox_inches="tight")
                plt.close("all")
            except Exception as exc:                        # noqa: BLE001
                print(f"[optuna] could not plot {label}: {exc}")
    except Exception as exc:                                # noqa: BLE001
        print(f"[optuna] plotting unavailable: {exc}")

    print(f"\nbest value {best.value:.5f} (trial {best.number})")
    print(json.dumps(best.params, indent=2))
    return best.params


def run(study_name: str, objective: Callable, n_trials: int, out_dir: str | Path,
        direction: str = "maximize", storage_dir: str | Path = "optuna",
        timeout: int | None = None) -> dict:
    study = make_study(study_name, storage_dir, direction)
    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   gc_after_trial=True)
    return summarise(study, out_dir, study_name)
