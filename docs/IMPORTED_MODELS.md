# Imported assignment models and report

Source: https://github.com/AbdullahMasood73/genai-assignment-1

Pinned commit: `4e841d6f966dcfcc64cd04019619609c855df7fa`.

All seven trained ONNX models are installed in `models/onnx/`. Their SHA-256
hashes match the upstream model manifest, which marks them as verified and
non-synthetic. The original files and report evidence are preserved under
`imports/genai-assignment-1/`.

| Assignment component | Original model | Installed model | Selected epoch |
|---|---|---|---:|
| Task 1: universal restoration | universal.onnx | task1_universal_ae.onnx | 22 |
| Task 2: corruption classifier | classifier.onnx | task2_classifier.onnx | 10 |
| Task 2: salt-and-pepper specialist | salt.onnx | task2_specialist_salt_pepper.onnx | 20 |
| Task 2: blur specialist | blur.onnx | task2_specialist_blur.onnx | 18 |
| Task 2: occlusion specialist | occlusion.onnx | task2_specialist_occlusion.onnx | 18 |
| Task 3: soft mixture of experts | soft.onnx | task3_soft_moe.onnx | 11 |
| Task 4: sketch generator | gan.onnx | task4_generator.onnx | 12 |

The import includes:

- The completed PDF, LaTeX report, metadata, and all report figures in
  `imports/genai-assignment-1/report/`.
- All published training artifacts, selected configurations, Optuna studies,
  MLflow database and artifacts, test and validation results, and provenance in
  `imports/genai-assignment-1/artifacts/`.
- The original training notebook, model/training/evaluation/export source,
  scripts, dependency specifications, samples, and supporting documentation.

The upstream repository does not publish the PyTorch `.pt` checkpoints or
datasets. The ONNX exports are sufficient to run all four app workspaces.
To retrain these imported architectures, use the original source under
`imports/genai-assignment-1/restoration/` and its original instructions.
The local `src/` directory contains the earlier scaffold's architectures.

The app adapter now handles the imported classifier's single logits output
with softmax and the imported GAN's `image` input and `[0, 1]` output range.
The class order remains clean, salt-and-pepper, blur, occlusion.

The original report records 36,690 test pet inputs and 1,046 face pairs, with
96.8% classifier accuracy and mean sketch SSIM of 0.455. These are the
repository's recorded evaluation results; importing and checking inference
does not rerun that dataset evaluation. Full metrics and limitations are in
the preserved report and `artifacts/results/test/`.

To repeat the download, run `.venv/Scripts/python.exe scripts/import_assignment_models.py`.
Existing files are checked against the pinned Git blob hashes and reused.
Installed model names and SHA-256 hashes are recorded in
`models/onnx/import-provenance.json`.

Local validation passed for all seven graphs with batch sizes 1 and 2,
12 restoration requests covering all four input conditions, and three sketch
styles through the frontend's API proxy. The classifier probabilities match
the original logits, and each sketch matches direct ONNX inference exactly
after PNG encoding. All three styles produce distinct outputs. The running
API reports every model available. Results are in `docs/import-validation.json`.
