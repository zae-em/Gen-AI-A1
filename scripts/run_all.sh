#!/usr/bin/env bash
# End-to-end pipeline: tune, train, evaluate, export, verify.
# Expect this to take many GPU-hours. Trim --n-trials and --epochs for a
# faster pass; the numbers below are the ones used for the reported results.
set -euo pipefail

DATA=${DATA:-data}
VAL_MANIFEST=$DATA/manifests/val_manifest.json
OPT=artifacts/optuna

echo "### Task 1 ###"
python -m src.tune.optuna_task1 --n-trials 30 --epochs 8 --data-root "$DATA"
python -m src.train.train_task1 --epochs 60 --data-root "$DATA" \
  --config $OPT/task1_universal_ae_best_config.json
python -m src.eval.evaluate_restoration

echo "### Task 2 ###"
python -m src.tune.optuna_task2 --stage classifier --n-trials 25 --epochs 6 --data-root "$DATA"
python -m src.train.train_task2_classifier --epochs 30 --data-root "$DATA" \
  --config $OPT/task2_classifier_best_config.json
python -m src.tune.optuna_task2 --stage specialists --n-trials 20 --epochs 6 --data-root "$DATA"
python -m src.train.train_task2_specialists --epochs 50 --data-root "$DATA" \
  --config $OPT/task2_specialists_best_config.json
python -m src.eval.evaluate_routing --mode hard

echo "### Task 3 ###"
python -m src.tune.optuna_task3 --n-trials 20 --epochs 5 --warmup-epochs 1 --data-root "$DATA"
python -m src.train.train_task3_moe --warmup-epochs 3 --epochs 25 --data-root "$DATA" \
  --config $OPT/task3_soft_moe_best_config.json
python -m src.eval.evaluate_routing --mode soft

echo "### Task 4 ###"
python -m src.tune.optuna_task4 --n-trials 15 --epochs 12
python -m src.train.train_task4_cgan --epochs 200 \
  --config $OPT/task4_cgan_best_config.json
python -m src.eval.evaluate_cgan

echo "### Export ###"
python -m src.export.export_onnx --all --out-dir models/onnx
python -m src.export.verify_onnx --onnx-dir models/onnx

echo "All done. Start the application with: docker compose up --build"
