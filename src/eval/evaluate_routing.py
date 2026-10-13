"""Tasks 2 and 3 final evaluation.

    python -m src.eval.evaluate_routing --mode hard   # oracle + predicted
    python -m src.eval.evaluate_routing --mode soft

Hard routing is scored twice, as the assignment requires:

  oracle    the expert is chosen from the deterministic test manifest label.
            This measures how good the specialists are.
  predicted the expert is chosen by the trained classifier.
            This measures the operational system.

The gap between the two is the cost of classifier error, and the script dumps
every case where oracle routing succeeded and predicted routing did not, which
is the material for that part of the discussion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.corruptions import CORRUPTION_NAMES
from ..data.datasets import dataset_from_manifest
from ..losses import ssim
from ..metrics import RestorationAccumulator, classification_report_dict, psnr
from ..models.autoencoder import build_autoencoder
from ..models.classifier import build_classifier
from ..models.moe import BRANCH_NAMES, SoftMoERestorer
from ..train.common import load_checkpoint
from ..utils.seed import pick_device, set_seed
from ..utils.viz import (comparison_grid, confusion_matrix_figure,
                         routing_heatmap)

EXPERT_KINDS = ("salt_pepper", "blur", "occlusion")


def load_specialists(directory, device):
    experts = {}
    for kind in EXPERT_KINDS:
        ckpt = load_checkpoint(Path(directory) / f"task2_specialist_{kind}.pt")
        model = build_autoencoder(ckpt["config"]).to(device).eval()
        model.load_state_dict(ckpt["state_dict"])
        experts[kind] = model
    return experts


def route(corrupt, decisions, experts):
    """Apply the per-sample routing decision. Class 0 (clean) bypasses."""
    out = corrupt.clone()
    for cls_idx in range(1, len(CORRUPTION_NAMES)):
        mask = decisions == cls_idx
        if mask.any():
            kind = CORRUPTION_NAMES[cls_idx]
            out[mask] = experts[kind](corrupt[mask])
    return out.clamp(0, 1)


@torch.no_grad()
def evaluate_hard(args, device):
    cls_ckpt = load_checkpoint(args.classifier)
    classifier = build_classifier(cls_ckpt["config"]).to(device).eval()
    classifier.load_state_dict(cls_ckpt["state_dict"])
    experts = load_specialists(args.specialist_dir, device)

    dataset = dataset_from_manifest(args.test_manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    oracle_acc, predicted_acc = RestorationAccumulator(), RestorationAccumulator()
    y_true, y_pred, regressions = [], [], []

    for corrupt, clean, labels, idx in tqdm(loader, desc="hard routing"):
        corrupt, clean = corrupt.to(device), clean.to(device)
        labels = labels.to(device)

        pred = classifier(corrupt).argmax(1)
        y_true.extend(labels.cpu().tolist()); y_pred.extend(pred.cpu().tolist())

        out_oracle = route(corrupt, labels, experts)
        out_pred = route(corrupt, pred, experts)

        entries = [dataset.manifest[int(i)] for i in idx]
        kinds = [e["corruption"] for e in entries]
        sevs = [e.get("severity") for e in entries]
        oracle_acc.update(out_oracle, clean, kinds, sevs)
        predicted_acc.update(out_pred, clean, kinds, sevs)

        s_o = ssim(out_oracle, clean, size_average=False).cpu()
        s_p = ssim(out_pred, clean, size_average=False).cpu()
        for i in range(clean.shape[0]):
            if int(pred[i]) != int(labels[i]):
                regressions.append({
                    "index": int(idx[i]),
                    "true": CORRUPTION_NAMES[int(labels[i])],
                    "predicted": CORRUPTION_NAMES[int(pred[i])],
                    "severity": sevs[i],
                    "ssim_oracle": float(s_o[i]),
                    "ssim_predicted": float(s_p[i]),
                    "ssim_drop": float(s_o[i] - s_p[i]),
                })

    report = classification_report_dict(y_true, y_pred,
                                        labels=list(range(len(CORRUPTION_NAMES))),
                                        target_names=list(CORRUPTION_NAMES))
    regressions.sort(key=lambda r: -r["ssim_drop"])
    return oracle_acc, predicted_acc, report, regressions, dataset, experts, classifier


@torch.no_grad()
def evaluate_soft(args, device):
    ckpt = load_checkpoint(args.moe)
    cls_ckpt = load_checkpoint(args.classifier)
    gate = build_classifier(cls_ckpt["config"])
    experts = []
    for kind in EXPERT_KINDS:
        spec = load_checkpoint(Path(args.specialist_dir) / f"task2_specialist_{kind}.pt")
        experts.append(build_autoencoder(spec["config"]))
    model = SoftMoERestorer(gate, experts, ckpt.get("temperature", 1.0)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dataset = dataset_from_manifest(args.test_manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    acc = RestorationAccumulator()
    weight_sums, weight_counts = {}, {}
    blended = []

    for corrupt, clean, _labels, idx in tqdm(loader, desc="soft routing"):
        corrupt, clean = corrupt.to(device), clean.to(device)
        recon, weights, _logits = model(corrupt)
        recon = recon.clamp(0, 1)

        entries = [dataset.manifest[int(i)] for i in idx]
        kinds = [e["corruption"] for e in entries]
        sevs = [e.get("severity") for e in entries]
        acc.update(recon, clean, kinds, sevs)

        w = weights.cpu()
        for i, (k, sev) in enumerate(zip(kinds, sevs)):
            key = f"{k}/{sev if sev else '-'}"
            weight_sums.setdefault(key, torch.zeros(len(BRANCH_NAMES)))
            weight_sums[key] += w[i]
            weight_counts[key] = weight_counts.get(key, 0) + 1
            top2 = torch.topk(w[i], 2).values
            if float(top2[1]) > 0.25:      # genuinely blended, not near-hard
                blended.append({"index": int(idx[i]), "corruption": k,
                                "severity": sev, "weights": w[i].tolist()})

    mean_weights = {k: (weight_sums[k] / weight_counts[k]).tolist()
                    for k in weight_sums}
    return acc, mean_weights, blended, dataset, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hard", "soft"], default="hard")
    ap.add_argument("--classifier", default="artifacts/task2/task2_classifier.pt")
    ap.add_argument("--specialist-dir", default="artifacts/task2")
    ap.add_argument("--moe", default="artifacts/task3/task3_soft_moe.pt")
    ap.add_argument("--test-manifest", default="data/manifests/test_manifest.json")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    set_seed(42)
    device = pick_device(args.device)
    out_dir = Path(args.out_dir or
                   ("artifacts/task2/eval" if args.mode == "hard" else "artifacts/task3/eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "hard":
        oracle, predicted, report, regressions, dataset, experts, classifier = \
            evaluate_hard(args, device)
        print("\n--- oracle routing ---\n" + oracle.as_table())
        print("\n--- predicted routing ---\n" + predicted.as_table())
        print(f"\nclassifier accuracy {report['accuracy']:.4f} | "
              f"macro-F1 {report['macro_f1']:.4f}")

        confusion_matrix_figure(report["confusion_matrix_normalised"],
                                CORRUPTION_NAMES,
                                out_dir / "task2_test_confusion_matrix.png")
        with open(out_dir / "task2_test_metrics.json", "w") as fh:
            json.dump({"oracle": oracle.summary(), "predicted": predicted.summary(),
                       "classifier": report,
                       "n_misrouted": len(regressions),
                       "worst_misroutes": regressions[:20]}, fh, indent=2)

        if regressions:
            idxs = [r["index"] for r in regressions[:4]]
            batch = [dataset[i] for i in idxs]
            corrupt = torch.stack([b[0] for b in batch]).to(device)
            clean = torch.stack([b[1] for b in batch])
            pred = classifier(corrupt).argmax(1)
            out = route(corrupt, pred, experts).cpu()
            comparison_grid(clean, corrupt.cpu(), out,
                            out_dir / "task2_misrouting_failures.png",
                            titles=[f"{r['true']}->{r['predicted']}"
                                    for r in regressions[:4]], max_rows=4)
            print(f"\n{len(regressions)} misrouted test items; worst drop "
                  f"{regressions[0]['ssim_drop']:.4f} SSIM")
    else:
        acc, mean_weights, blended, dataset, model = evaluate_soft(args, device)
        print("\n--- soft mixture of experts ---\n" + acc.as_table())
        routing_heatmap(mean_weights, out_dir / "task3_test_routing_heatmap.png",
                        BRANCH_NAMES)
        dead = [BRANCH_NAMES[i] for i in range(len(BRANCH_NAMES))
                if max(v[i] for v in mean_weights.values()) < 0.10]
        with open(out_dir / "task3_test_metrics.json", "w") as fh:
            json.dump({"metrics": acc.summary(),
                       "mean_weights_by_condition": mean_weights,
                       "possibly_inactive_branches": dead,
                       "n_blended": len(blended),
                       "blended_examples": blended[:20]}, fh, indent=2)
        print(f"\n{len(blended)} test items routed to two or more experts "
              f"(second weight > 0.25)")
        if dead:
            print(f"WARNING: branches that never activate: {dead}")

    print(f"\nwrote results to {out_dir}")


if __name__ == "__main__":
    main()
