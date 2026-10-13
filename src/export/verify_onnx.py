"""Check that every ONNX model agrees with its PyTorch original.

    python -m src.export.verify_onnx --onnx-dir models/onnx

Runs the same random inputs through both and reports the maximum absolute
difference.  The assignment asks for this check explicitly; the tolerance is
1e-4, which is loose enough for float32 graph optimisation and tight enough to
catch a genuinely wrong export (wrong normalisation, a dropped sigmoid, a
frozen batch-norm that was not in eval mode).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..models.autoencoder import build_autoencoder
from ..models.classifier import ClassifierWithProbs, build_classifier
from ..models.moe import SoftMoEExport, SoftMoERestorer
from ..models.pix2pix import build_generator
from ..train.common import load_checkpoint

EXPERT_KINDS = ("salt_pepper", "blur", "occlusion")


def compare(torch_out, ort_out, name, tolerance):
    if isinstance(torch_out, torch.Tensor):
        torch_out = [torch_out]
    diffs = [float(np.abs(t.detach().numpy() - o).max())
             for t, o in zip(torch_out, ort_out)]
    worst = max(diffs)
    ok = worst < tolerance
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<32} max|diff| = {worst:.3e}")
    return {"model": name, "max_abs_diff": worst, "passed": bool(ok)}


def main():
    import onnxruntime as ort

    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default="models/onnx")
    ap.add_argument("--task1", default="artifacts/task1/task1_universal_ae.pt")
    ap.add_argument("--classifier", default="artifacts/task2/task2_classifier.pt")
    ap.add_argument("--specialist-dir", default="artifacts/task2")
    ap.add_argument("--moe", default="artifacts/task3/task3_soft_moe.pt")
    ap.add_argument("--generator", default="artifacts/task4/task4_generator.pt")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(0)
    onnx_dir = Path(args.onnx_dir)
    x = torch.randn(args.batch, 3, args.image_size, args.image_size)
    results = []

    def session(path):
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    print("verifying ONNX exports against PyTorch:")

    p = onnx_dir / "task1_universal_ae.onnx"
    if p.exists():
        ckpt = load_checkpoint(args.task1)
        m = build_autoencoder(ckpt["config"]); m.load_state_dict(ckpt["state_dict"]); m.eval()
        with torch.no_grad():
            t = m(x)
        o = session(p).run(None, {"image": x.numpy()})
        results.append(compare(t, o, "task1_universal_ae", args.tolerance))

    p = onnx_dir / "task2_classifier.onnx"
    if p.exists():
        ckpt = load_checkpoint(args.classifier)
        base = build_classifier(ckpt["config"]); base.load_state_dict(ckpt["state_dict"])
        m = ClassifierWithProbs(base); m.eval()
        with torch.no_grad():
            t = list(m(x))
        o = session(p).run(None, {"image": x.numpy()})
        results.append(compare(t, o, "task2_classifier", args.tolerance))

    for kind in EXPERT_KINDS:
        p = onnx_dir / f"task2_specialist_{kind}.onnx"
        c = Path(args.specialist_dir) / f"task2_specialist_{kind}.pt"
        if p.exists() and c.exists():
            ckpt = load_checkpoint(c)
            m = build_autoencoder(ckpt["config"]); m.load_state_dict(ckpt["state_dict"]); m.eval()
            with torch.no_grad():
                t = m(x)
            o = session(p).run(None, {"image": x.numpy()})
            results.append(compare(t, o, f"task2_specialist_{kind}", args.tolerance))

    p = onnx_dir / "task3_soft_moe.onnx"
    if p.exists():
        ckpt = load_checkpoint(args.moe)
        gate = build_classifier(load_checkpoint(args.classifier)["config"])
        experts = [build_autoencoder(load_checkpoint(
            Path(args.specialist_dir) / f"task2_specialist_{k}.pt")["config"])
            for k in EXPERT_KINDS]
        moe = SoftMoERestorer(gate, experts, ckpt.get("temperature", 1.0))
        moe.load_state_dict(ckpt["state_dict"])
        m = SoftMoEExport(moe); m.eval()
        with torch.no_grad():
            t = list(m(x))
        o = session(p).run(None, {"image": x.numpy()})
        results.append(compare(t, o, "task3_soft_moe", args.tolerance))

    p = onnx_dir / "task4_generator.onnx"
    if p.exists():
        ckpt = load_checkpoint(args.generator)
        m = build_generator(ckpt["config"]); m.load_state_dict(ckpt["state_dict"]); m.eval()
        style = torch.zeros(args.batch, dtype=torch.int64)
        with torch.no_grad():
            t = m(x, style)
        o = session(p).run(None, {"photo": x.numpy(), "style": style.numpy()})
        results.append(compare(t, o, "task4_generator", args.tolerance))

    out = onnx_dir / "verification_report.json"
    with open(out, "w") as fh:
        json.dump({"tolerance": args.tolerance, "results": results}, fh, indent=2)

    failed = [r for r in results if not r["passed"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} models within tolerance")
    print(f"report written to {out}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
