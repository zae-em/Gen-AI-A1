"""Export every inference model to ONNX.

    python -m src.export.export_onnx --all --out-dir models/onnx

Exports:
    task1_universal_ae.onnx     image -> restored image
    task2_classifier.onnx       image -> (logits, probabilities)
    task2_specialist_*.onnx     image -> restored image  (three files)
    task3_soft_moe.onnx         image -> (restored image, routing weights)
    task4_generator.onnx        (photo, style_index) -> sketch

The Task 3 export deliberately contains the whole pipeline - gate, softmax,
three experts and the weighted sum - so the backend does one session run and
cannot drift out of step with the trained model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..models.autoencoder import build_autoencoder
from ..models.classifier import ClassifierWithProbs, build_classifier
from ..models.moe import SoftMoEExport, SoftMoERestorer
from ..models.pix2pix import build_generator
from ..train.common import load_checkpoint

EXPERT_KINDS = ("salt_pepper", "blur", "occlusion")
OPSET = 17


def _export(model, dummy, path: Path, input_names, output_names, dynamic_axes):
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(model, dummy, str(path), export_params=True,
                      opset_version=OPSET, do_constant_folding=True,
                      input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes)
    size_mb = path.stat().st_size / 1e6
    print(f"  wrote {path}  ({size_mb:.1f} MB)")
    return path


def export_autoencoder(ckpt_path, out_path, size=128):
    ckpt = load_checkpoint(ckpt_path)
    model = build_autoencoder(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    dummy = torch.randn(1, 3, size, size)
    return _export(model, dummy, Path(out_path), ["image"], ["restored"],
                   {"image": {0: "batch"}, "restored": {0: "batch"}})


def export_classifier(ckpt_path, out_path, size=128):
    ckpt = load_checkpoint(ckpt_path)
    base = build_classifier(ckpt["config"])
    base.load_state_dict(ckpt["state_dict"])
    model = ClassifierWithProbs(base)
    dummy = torch.randn(1, 3, size, size)
    return _export(model, dummy, Path(out_path), ["image"],
                   ["logits", "probabilities"],
                   {"image": {0: "batch"}, "logits": {0: "batch"},
                    "probabilities": {0: "batch"}})


def export_moe(moe_ckpt, classifier_ckpt, specialist_dir, out_path, size=128):
    ckpt = load_checkpoint(moe_ckpt)
    gate = build_classifier(load_checkpoint(classifier_ckpt)["config"])
    experts = [build_autoencoder(
        load_checkpoint(Path(specialist_dir) / f"task2_specialist_{k}.pt")["config"])
        for k in EXPERT_KINDS]
    moe = SoftMoERestorer(gate, experts, ckpt.get("temperature", 1.0))
    moe.load_state_dict(ckpt["state_dict"])
    dummy = torch.randn(1, 3, size, size)
    return _export(SoftMoEExport(moe), dummy, Path(out_path), ["image"],
                   ["restored", "weights"],
                   {"image": {0: "batch"}, "restored": {0: "batch"},
                    "weights": {0: "batch"}})


def export_generator(ckpt_path, out_path, size=128):
    ckpt = load_checkpoint(ckpt_path)
    model = build_generator(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    dummy = (torch.randn(1, 3, size, size), torch.zeros(1, dtype=torch.int64))
    return _export(model, dummy, Path(out_path), ["photo", "style"], ["sketch"],
                   {"photo": {0: "batch"}, "style": {0: "batch"},
                    "sketch": {0: "batch"}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="models/onnx")
    ap.add_argument("--task1", default="artifacts/task1/task1_universal_ae.pt")
    ap.add_argument("--classifier", default="artifacts/task2/task2_classifier.pt")
    ap.add_argument("--specialist-dir", default="artifacts/task2")
    ap.add_argument("--moe", default="artifacts/task3/task3_soft_moe.pt")
    ap.add_argument("--generator", default="artifacts/task4/task4_generator.pt")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["task1", "task2", "task3", "task4"])
    args = ap.parse_args()

    out = Path(args.out_dir)
    wanted = set(args.only or ["task1", "task2", "task3", "task4"])
    if args.all:
        wanted = {"task1", "task2", "task3", "task4"}

    if "task1" in wanted and Path(args.task1).exists():
        print("task 1:")
        export_autoencoder(args.task1, out / "task1_universal_ae.onnx", args.image_size)

    if "task2" in wanted:
        print("task 2:")
        if Path(args.classifier).exists():
            export_classifier(args.classifier, out / "task2_classifier.onnx",
                              args.image_size)
        for kind in EXPERT_KINDS:
            p = Path(args.specialist_dir) / f"task2_specialist_{kind}.pt"
            if p.exists():
                export_autoencoder(p, out / f"task2_specialist_{kind}.onnx",
                                   args.image_size)

    if "task3" in wanted and Path(args.moe).exists():
        print("task 3:")
        export_moe(args.moe, args.classifier, args.specialist_dir,
                   out / "task3_soft_moe.onnx", args.image_size)

    if "task4" in wanted and Path(args.generator).exists():
        print("task 4:")
        export_generator(args.generator, out / "task4_generator.onnx", args.image_size)

    print(f"\nONNX models in {out}")


if __name__ == "__main__":
    main()
