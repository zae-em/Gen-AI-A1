"""One wrapper over MLflow and Weights & Biases.

Every training and tuning script logs through this, so switching backends is
a flag (--tracker wandb) rather than an edit.  If neither package is
installed the run still proceeds and metrics go to stdout - useful when a
quick sanity run should not create an experiment record.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentTracker:
    def __init__(self, backend: str = "mlflow", project: str = "genai-assignment-1",
                 run_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None,
                 output_dir: str | Path = "runs"):
        self.backend = backend
        self.project = project
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._impl = None
        self._active = False
        self._start(config or {})

    def _start(self, config: Dict[str, Any]) -> None:
        if self.backend == "mlflow":
            try:
                import mlflow
                mlflow.set_experiment(self.project)
                mlflow.start_run(run_name=self.run_name)
                mlflow.log_params(_flatten(config))
                self._impl, self._active = mlflow, True
            except Exception as exc:                      # noqa: BLE001
                print(f"[tracking] mlflow unavailable ({exc}); logging to stdout")
        elif self.backend == "wandb":
            try:
                import wandb
                wandb.init(project=self.project, name=self.run_name, config=config)
                self._impl, self._active = wandb, True
            except Exception as exc:                      # noqa: BLE001
                print(f"[tracking] wandb unavailable ({exc}); logging to stdout")
        if config:
            with open(self.output_dir / "config.json", "w") as fh:
                json.dump(config, fh, indent=2, default=str)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        if not self._active:
            tag = f"step {step}" if step is not None else ""
            print(f"[metrics] {tag} " + " ".join(f"{k}={v:.5f}" for k, v in clean.items()))
            return
        if self.backend == "mlflow":
            self._impl.log_metrics(clean, step=step)
        else:
            self._impl.log(clean, step=step)

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self._active:
            print(f"[params] {params}")
            return
        if self.backend == "mlflow":
            self._impl.log_params(_flatten(params))
        else:
            self._impl.config.update(params, allow_val_change=True)

    def log_artifact(self, path: str | Path) -> None:
        if not self._active:
            return
        if self.backend == "mlflow":
            self._impl.log_artifact(str(path))
        else:
            self._impl.save(str(path))

    def log_image(self, path: str | Path, caption: str = "") -> None:
        if not self._active:
            return
        if self.backend == "mlflow":
            self._impl.log_artifact(str(path))
        else:
            self._impl.log({caption or Path(path).stem: self._impl.Image(str(path))})

    def finish(self) -> None:
        if not self._active:
            return
        if self.backend == "mlflow":
            self._impl.end_run()
        else:
            self._impl.finish()


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out
