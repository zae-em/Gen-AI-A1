# What this repository does not contain

Update: trained ONNX models and the upstream completed report, results, and
training evidence have now been imported. See [IMPORTED_MODELS.md](IMPORTED_MODELS.md).
The notes below describe the original scaffold before that import.

This is working scaffolding: the architectures, the data pipeline, the training
and tuning loops, the evaluation, the export path, the application and the
report structure. It is not a finished submission, and a few parts of the
assignment cannot be produced by anyone but you.

## Nobody can hand you these

**Trained weights.** Every model here is untrained. Training needs the two
datasets and a GPU. Until you run the scripts there are no `.pt` files, no
`.onnx` files and the application will report every model as missing.

**Your results.** Every table in `report/main.tex` is marked
`\PLACEHOLDER`. The numbers have to come from your runs. Invented numbers are
the one thing guaranteed to fail the viva, because the evaluator can rerun the
evaluation scripts.

**The Optuna evidence.** The search spaces are defined, but the assignment asks
for completed trials, the best trial, and the selected configuration. That means
actually running the studies. `artifacts/optuna/*_trials.json` is written for
you to quote from.

**The Google Stitch design.** The assignment requires the interface to be
designed in Stitch first, with evidence in the report. The React implementation
here gives you something concrete to design against, but the Stitch artefact has
to be yours.

**The demonstration video.** Five to seven minutes, uploaded to YouTube, showing
startup and all four workspaces. It has to show your running system.

**The analysis.** The assignment is explicit that the report must interpret
every table and figure — what the result demonstrates, why a failure occurred,
how the evidence shaped your conclusions. That is the part being graded, and it
depends on having watched your own training runs.

## What you should verify before trusting any of this

The assignment says you remain responsible for correctness and must understand
everything you submit. Concretely:

1. **Run `python scripts/smoke_test.py`.** It checks corruption statistics
   against the specification, SSIM behaviour, every model's shapes, the routing
   weights, and ONNX agreement. If something here is wrong, that is where it
   will show.

2. **Check the corruption operators against the assignment text yourself.**
   Read `src/data/corruptions.py` next to the specification table. Two choices
   in there are interpretations, not transcriptions, and you should be able to
   defend or change them:
   - salt-and-pepper is applied per pixel rather than per channel;
   - occlusion rectangles may overlap, so realised coverage runs below the
     nominal target (the manifest records the measured value).

3. **Confirm the FS2K annotation format.** `src/data/fs2k.py` assumes records
   with `image_name` and `style` fields and resolves paths by globbing. Open
   `anno_train.json` and check. The loader prints how many records it skipped —
   if that number is not near zero, the path resolution needs adjusting for your
   copy of the dataset.

4. **Read the architectures.** You will be asked to explain the bottleneck, why
   skip connections are limited, why the gate is initialised from the classifier,
   and why the discriminator needs the style embedding. The reasoning is in the
   module docstrings, but reading it is not the same as being able to defend it.

5. **Verify the ONNX exports numerically.** `src/export/verify_onnx.py` is not
   decorative — the assignment requires this check, and a batch-norm layer left
   in training mode is a real failure mode that it catches.

## Suggested order of work

1. Smoke test, then read `corruptions.py` and `datasets.py` end to end.
2. Download the data, build the manifests, inspect a few corrupted samples
   visually before training anything.
3. Task 1 with a short run (5 epochs, no Optuna) to confirm the loop works and
   the loss falls.
4. Full Optuna study and full training for Task 1, then evaluation.
5. Task 2, then Task 3 (which depends on it), then Task 4.
6. Export, verify, start the containers, check all four workspaces.
7. Stitch design, screenshots, the video.
8. Write the report last, with the numbers in front of you.

Tasks 1 and 2 are independent of Task 4, so if you have limited GPU time, start
the cGAN training in parallel — it is the longest run of the four.

## On the AI-use appendix

The assignment permits AI assistance and requires you to declare it. If you used
this scaffolding, say so, name what it produced, and record what you tested and
what you changed. The evaluator is looking for evidence that you checked the
output rather than a claim that you wrote everything yourself. A short, specific
appendix — files touched, bugs found, decisions you overrode — reads far better
than a vague one.
