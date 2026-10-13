# Restoration Bench

All seven trained models have been imported from
[AbdullahMasood73/genai-assignment-1](https://github.com/AbdullahMasood73/genai-assignment-1).
See [the import details](docs/IMPORTED_MODELS.md) and
[the completed assignment report](imports/genai-assignment-1/report/report.pdf).

Four generative systems behind one browser application:

1. **Universal Restoration** — one denoising autoencoder for clean images,
   salt-and-pepper noise, Gaussian blur and rectangular occlusion, with no
   indication of which corruption was applied.
2. **Hard-Routed Restoration** — a four-class corruption classifier that selects
   one of three specialist autoencoders, with an identity bypass for clean input.
3. **Soft Mixture-of-Experts Restoration** — a differentiable gate that blends
   all four branches, initialised from the components above and fine-tuned jointly.
4. **Face-to-Sketch Generator** — a style-conditioned pix2pix-style conditional
   GAN trained on FS2K.

Inference runs on ONNX Runtime inside a container. `docker compose up --build`
starts everything.

---

## Quick start for an evaluator

```bash
git clone <REPO-URL> && cd <REPO>

# put the seven .onnx files in models/onnx/ (see models/onnx/README.md)
docker compose up --build
```

Then open **http://localhost:3000**. The backend is on
http://localhost:8000, with interactive API docs at
http://localhost:8000/docs.

`GET /api/health` reports which models loaded. The interface shows the same
status in the sidebar, so a missing model produces a readable message rather
than a failed request.

Nothing else is required: no virtual environment, no VS Code, no manually run
Python scripts.

---

## Repository layout

```
src/data/          corruption operators, datasets, deterministic manifests
src/models/        autoencoder, classifier, soft MoE, pix2pix
src/train/         one training script per task
src/tune/          one Optuna study per task
src/eval/          test-set evaluation and failure-case extraction
src/export/        ONNX export and PyTorch-vs-ONNX verification
backend/           FastAPI service + Dockerfile
frontend/          React + Tailwind + Vite + nginx Dockerfile
report/            IEEE LaTeX skeleton
scripts/           dataset download, smoke test, full pipeline
configs/           starting configurations (Optuna overwrites these choices)
```

---

## Reproducing the models

### 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate

# install PyTorch for your hardware first: https://pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### 2. Check the code before downloading anything

```bash
python scripts/smoke_test.py
```

Builds every model on random tensors, checks the corruption statistics against
the specification, verifies SSIM and the routing weights, and round-trips three
models through ONNX. Under a minute on CPU, and it catches shape and export
bugs before a training run wastes an afternoon on them.

### 3. Data

```bash
bash scripts/download_data.sh data
```

Oxford-IIIT Pet downloads automatically through torchvision. FS2K needs a
manual download from <https://github.com/DengPingFan/FS2K>; unpack it so that
`data/FS2K/anno_train.json`, `anno_test.json`, `photo/` and `sketch/` exist.

The script then writes the deterministic corruption manifests to
`data/manifests/`. These are small (specs, not pixels) and **should be
committed** — every reported number depends on them.

### 4. Train

The whole pipeline:

```bash
bash scripts/run_all.sh
```

Or one stage at a time:

```bash
# Task 1
python -m src.tune.optuna_task1 --n-trials 30 --epochs 8
python -m src.train.train_task1 --epochs 60 \
  --config artifacts/optuna/task1_universal_ae_best_config.json
python -m src.eval.evaluate_restoration

# Task 2
python -m src.tune.optuna_task2 --stage classifier --n-trials 25 --epochs 6
python -m src.train.train_task2_classifier --epochs 30 \
  --config artifacts/optuna/task2_classifier_best_config.json
python -m src.tune.optuna_task2 --stage specialists --n-trials 20 --epochs 6
python -m src.train.train_task2_specialists --epochs 50 \
  --config artifacts/optuna/task2_specialists_best_config.json
python -m src.eval.evaluate_routing --mode hard

# Task 3 (needs the Task 2 checkpoints)
python -m src.tune.optuna_task3 --n-trials 20 --epochs 5 --warmup-epochs 1
python -m src.train.train_task3_moe --warmup-epochs 3 --epochs 25 \
  --config artifacts/optuna/task3_soft_moe_best_config.json
python -m src.eval.evaluate_routing --mode soft

# Task 4
python -m src.tune.optuna_task4 --n-trials 15 --epochs 12
python -m src.train.train_task4_cgan --epochs 200 \
  --config artifacts/optuna/task4_cgan_best_config.json
python -m src.eval.evaluate_cgan
```

Order matters in one place only: Task 3 loads the trained classifier and the
three specialists from Task 2 and will not start without them.

### 5. Export and verify

```bash
python -m src.export.export_onnx --all --out-dir models/onnx
python -m src.export.verify_onnx --onnx-dir models/onnx
```

`verify_onnx` exits non-zero if any model drifts more than `1e-4` from its
PyTorch original, and writes `models/onnx/verification_report.json` — the table
the report needs.

---

## Experiment tracking

Every training script logs through one wrapper, so the backend is a flag:

```bash
python -m src.train.train_task1 --tracker mlflow        # default
python -m src.train.train_task1 --tracker wandb --project genai-assignment-1
python -m src.train.train_task1 --no-tracking           # quick sanity runs
```

For MLflow, `mlflow ui` in the repository root opens the run history.

---

## Development without Docker

```bash
# backend
cd backend && pip install -r requirements.txt
MODEL_DIR=../models/onnx uvicorn app.main:app --reload --port 8000

# frontend (separate terminal)
cd frontend && npm install && npm run dev
```

Vite proxies `/api` to `http://localhost:8000`; override with
`VITE_API_TARGET`.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | which models are loaded |
| POST | `/api/corrupt` | apply a corruption without running a model |
| POST | `/api/universal` | Task 1 |
| POST | `/api/hard-routed` | Task 2 — returns probabilities, prediction, expert |
| POST | `/api/soft-moe` | Task 3 — returns all four routing weights |
| POST | `/api/face-to-sketch` | Task 4 — takes `style` ∈ {0, 1, 2} |

The restoration endpoints accept an optional `corruption`
(`salt_pepper`, `blur`, `occlusion`, or `none`) and `severity`
(`low`, `medium`, `high`), so an evaluator can start from a clean photograph
and choose the damage.

```bash
curl -F "file=@cat.jpg" -F "corruption=blur" -F "severity=high" \
  http://localhost:8000/api/hard-routed
```

---

## Implementation notes worth knowing

**Corruption class order is fixed** as `clean, salt_pepper, blur, occlusion` in
`src/data/corruptions.py`. Classifier logits, gate weights, ONNX outputs and the
frontend labels all depend on it. Do not reorder it.

**Occlusion coverage is measured, not assumed.** Rectangles may overlap, so
realised coverage sits slightly below the nominal target. The manifest records
the true value under `realised_coverage`; quote that number, not the nominal one.

**Salt-and-pepper is per pixel, not per channel.** A corrupted pixel is
(0,0,0) or (1,1,1), which is how impulse noise actually appears and keeps the
artefact unambiguous for the classifier.

**The classifier's first block is stride 1** so isolated noisy pixels survive
into the feature maps. An aggressive stride-2 stem averages them away and the
clean-versus-noise confusion rises sharply at low `p`.

**Soft routing costs three expert evaluations per image** because every branch
runs. That is inherent to the approach, not an implementation flaw.

**Skip connections are off by default.** Full-resolution skips would let the
network copy the corrupted input to the output and defeat the bottleneck. A
single coarse skip at the deepest encoder stage is available behind
`use_skip=True` if you want to measure the trade-off.

---

## Troubleshooting

**A workspace says a model is missing.** Check `GET /api/health`. Either export
the ONNX files or drop the released ones into `models/onnx/`, then restart the
backend container.

**`verify_onnx` fails on one model.** Almost always a module left in training
mode during export — batch-norm statistics differ between train and eval. The
export helper calls `.eval()`, so check for a custom wrapper that does not.

**Task 3 will not start.** It needs
`artifacts/task2/task2_classifier.pt` and the three
`task2_specialist_*.pt` files. Train Task 2 first.

**FS2K paths do not resolve.** The loader globs for extensions and handles the
`image####` / `sketch####` naming difference, but it prints a count of skipped
records. If that count is large, check the directory layout against the
structure described above.

**Out of memory.** Lower `batch_size` in the relevant config. Task 3 is the
heaviest because three experts and a gate are resident at once.
