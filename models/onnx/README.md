# ONNX models

All seven trained models from `AbdullahMasood73/genai-assignment-1` have been
imported here and checked against its manifest. See
[`docs/IMPORTED_MODELS.md`](../../docs/IMPORTED_MODELS.md) for their source,
filename mapping, and the preserved completed report and experiment evidence.

The trained models are not committed to git. Get them one of two ways.

**Download the release.** Put every `.onnx` file from the GitHub release
directly in this folder. Expected filenames:

    task1_universal_ae.onnx
    task2_classifier.onnx
    task2_specialist_salt_pepper.onnx
    task2_specialist_blur.onnx
    task2_specialist_occlusion.onnx
    task3_soft_moe.onnx
    task4_generator.onnx

**Or export them yourself** after training:

    python -m src.export.export_onnx --all --out-dir models/onnx
    python -m src.export.verify_onnx --onnx-dir models/onnx

`GET /api/health` lists exactly which files the backend can see, so if a
workspace reports a missing model, check that endpoint first.
