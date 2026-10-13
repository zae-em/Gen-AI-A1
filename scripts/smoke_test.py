"""Run this first, before downloading anything.

    python scripts/smoke_test.py

It builds every model on random tensors and checks the shapes, the corruption
operators, the SSIM implementation, the routing weights and a full ONNX export
and reload.  Nothing here touches a dataset, so it finishes in under a minute
on CPU and catches the class of bug that otherwise shows up forty minutes into
a training run.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.corruptions import (CORRUPTION_NAMES, apply_spec, coverage_of,
                                  sample_spec, sample_test_spec)
from src.losses import ReconstructionLoss, balance_loss, routing_entropy, ssim
from src.models.autoencoder import DenoisingAutoencoder
from src.models.classifier import ClassifierWithProbs, CorruptionClassifier
from src.models.moe import SoftMoEExport, SoftMoERestorer
from src.models.pix2pix import (StylePatchDiscriminator, StyleUNetGenerator,
                                init_weights)

PASS, FAIL = "  ok  ", " FAIL "
failures = []


def check(name, condition, detail=""):
    print(f"{PASS if condition else FAIL} {name}{(' - ' + detail) if detail else ''}")
    if not condition:
        failures.append(name)


def test_corruptions():
    print("\ncorruption operators")
    img = torch.rand(3, 128, 128)

    for kind in CORRUPTION_NAMES:
        spec = sample_spec(128, 128, force_kind=kind)
        out = apply_spec(img, spec)
        check(f"{kind}: shape and range preserved",
              out.shape == img.shape and 0.0 <= float(out.min())
              and float(out.max()) <= 1.0)

    sp = sample_spec(128, 128, force_kind="salt_pepper")
    out = apply_spec(img, sp)
    changed = (out != img).any(0).float().mean().item()
    check("salt & pepper hits roughly the requested fraction",
          abs(changed - sp.prob) < 0.03, f"asked {sp.prob:.3f}, got {changed:.3f}")

    extremes = ((out == 0).all(0) | (out == 1).all(0)).float().mean().item()
    check("corrupted pixels are pure black or white", extremes >= changed - 0.01)

    for i, expected in enumerate((0.10, 0.20, 0.35)):
        spec = sample_test_spec(128, 128, "occlusion", i,
                                torch.Generator().manual_seed(i))
        got = coverage_of(spec.rects, 128, 128)
        check(f"occlusion severity {i} covers about {expected:.0%}",
              abs(got - expected) < 0.06, f"{got:.3f}")

    blur = sample_test_spec(128, 128, "blur", 2)
    blurred = apply_spec(img, blur)
    check("blur reduces local variance",
          float(blurred.var()) < float(img.var()))

    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = apply_spec(img, sample_spec(128, 128, g1, "salt_pepper"), g1)
    b = apply_spec(img, sample_spec(128, 128, g2, "salt_pepper"), g2)
    check("seeded corruption is reproducible", torch.equal(a, b))


def test_losses():
    print("\nlosses")
    x = torch.rand(4, 3, 64, 64)
    check("SSIM(x, x) == 1", abs(float(ssim(x, x)) - 1.0) < 1e-4,
          f"{float(ssim(x, x)):.6f}")
    noisy = (x + 0.3 * torch.randn_like(x)).clamp(0, 1)
    check("SSIM falls on a noisy copy", float(ssim(x, noisy)) < 0.9,
          f"{float(ssim(x, noisy)):.4f}")

    loss, l1, s = ReconstructionLoss(0.8)(noisy, x)
    check("reconstruction loss is a finite scalar",
          loss.ndim == 0 and torch.isfinite(loss))
    check("loss sits between its components",
          min(float(l1), float(s)) <= float(loss) <= max(float(l1), float(s)))

    uniform = torch.full((8, 4), 0.25)
    collapsed = torch.zeros(8, 4); collapsed[:, 1] = 1.0
    check("balance loss is zero at uniform routing",
          float(balance_loss(uniform)) < 1e-8)
    check("balance loss punishes collapse",
          float(balance_loss(collapsed)) > float(balance_loss(uniform)))
    check("entropy is higher for uniform routing",
          float(routing_entropy(uniform)) > float(routing_entropy(collapsed + 1e-6)))


def test_models():
    print("\nmodel shapes")
    x = torch.rand(2, 3, 128, 128)

    for latent in (32, 128, 256):
        ae = DenoisingAutoencoder(base_channels=16, latent_dim=latent).eval()
        with torch.no_grad():
            y = ae(x)
        check(f"autoencoder (latent {latent}) returns the input shape",
              y.shape == x.shape)
        check(f"autoencoder (latent {latent}) actually compresses",
              ae.compression_ratio() > 1.0, f"{ae.compression_ratio():.2f}x")

    ae_skip = DenoisingAutoencoder(base_channels=16, use_skip=True).eval()
    with torch.no_grad():
        check("autoencoder with the coarse skip still works",
              ae_skip(x).shape == x.shape)

    for depth in (3, 4, 5):
        clf = CorruptionClassifier(base_channels=16, depth=depth).eval()
        with torch.no_grad():
            check(f"classifier (depth {depth}) returns four logits",
                  clf(x).shape == (2, 4))

    with torch.no_grad():
        logits, probs = ClassifierWithProbs(CorruptionClassifier(16).eval())(x)
    check("classifier probabilities sum to one",
          torch.allclose(probs.sum(1), torch.ones(2), atol=1e-5))

    gate = CorruptionClassifier(16)
    experts = [DenoisingAutoencoder(base_channels=16) for _ in range(3)]
    moe = SoftMoERestorer(gate, experts, temperature=1.0).eval()
    with torch.no_grad():
        recon, weights, _ = moe(x)
    check("mixture output matches the input shape", recon.shape == x.shape)
    check("routing weights sum to one",
          torch.allclose(weights.sum(1), torch.ones(2), atol=1e-5))

    moe.set_expert_grad(False)
    frozen = all(not p.requires_grad for p in moe.experts.parameters())
    trainable = all(p.requires_grad for p in moe.gate.parameters())
    check("warm-up freezes the experts but not the gate", frozen and trainable)
    moe.set_expert_grad(True)

    photo = torch.rand(2, 3, 128, 128) * 2 - 1
    style = torch.tensor([0, 2])
    gen = StyleUNetGenerator(base_channels=16, style_dim=8).eval()
    disc = StylePatchDiscriminator(base_channels=16, style_dim=8).eval()
    init_weights(gen); init_weights(disc)
    with torch.no_grad():
        fake = gen(photo, style)
        verdict = disc(photo, style, fake)
    check("generator returns a full-resolution sketch", fake.shape == photo.shape)
    check("generator output is inside [-1, 1]",
          float(fake.min()) >= -1.0 and float(fake.max()) <= 1.0)
    check("discriminator returns a patch grid",
          verdict.ndim == 4 and verdict.shape[1] == 1, str(tuple(verdict.shape)))

    with torch.no_grad():
        a = gen(photo[:1], torch.tensor([0]))
        b = gen(photo[:1], torch.tensor([2]))
    check("style condition changes the output", float((a - b).abs().mean()) > 1e-5,
          f"mean |diff| = {float((a - b).abs().mean()):.5f}")


def test_onnx():
    print("\nONNX export")
    try:
        import onnxruntime as ort
    except ImportError:
        print("  -- onnxruntime not installed, skipping")
        return

    x = torch.rand(1, 3, 128, 128)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        ae = DenoisingAutoencoder(base_channels=16, latent_dim=64).eval()
        torch.onnx.export(ae, x, str(tmp / "ae.onnx"), opset_version=17,
                          input_names=["image"], output_names=["restored"],
                          dynamic_axes={"image": {0: "batch"},
                                        "restored": {0: "batch"}})
        with torch.no_grad():
            expected = ae(x).numpy()
        got = ort.InferenceSession(str(tmp / "ae.onnx")).run(
            None, {"image": x.numpy()})[0]
        diff = float(abs(expected - got).max())
        check("autoencoder ONNX matches PyTorch", diff < 1e-4, f"max|diff| {diff:.2e}")

        gate = CorruptionClassifier(16)
        experts = [DenoisingAutoencoder(base_channels=16, latent_dim=64)
                   for _ in range(3)]
        moe = SoftMoEExport(SoftMoERestorer(gate, experts, 1.0)).eval()
        torch.onnx.export(moe, x, str(tmp / "moe.onnx"), opset_version=17,
                          input_names=["image"], output_names=["restored", "weights"],
                          dynamic_axes={"image": {0: "batch"}})
        with torch.no_grad():
            e_recon, e_w = moe(x)
        g_recon, g_w = ort.InferenceSession(str(tmp / "moe.onnx")).run(
            None, {"image": x.numpy()})
        d = max(float(abs(e_recon.numpy() - g_recon).max()),
                float(abs(e_w.numpy() - g_w).max()))
        check("mixture ONNX matches PyTorch", d < 1e-4, f"max|diff| {d:.2e}")

        gen = StyleUNetGenerator(base_channels=16, style_dim=8).eval()
        style = torch.zeros(1, dtype=torch.int64)
        torch.onnx.export(gen, (x, style), str(tmp / "gen.onnx"), opset_version=17,
                          input_names=["photo", "style"], output_names=["sketch"],
                          dynamic_axes={"photo": {0: "batch"}, "style": {0: "batch"}})
        with torch.no_grad():
            expected = gen(x, style).numpy()
        got = ort.InferenceSession(str(tmp / "gen.onnx")).run(
            None, {"photo": x.numpy(), "style": style.numpy()})[0]
        diff = float(abs(expected - got).max())
        check("generator ONNX matches PyTorch", diff < 1e-4, f"max|diff| {diff:.2e}")


def main():
    torch.manual_seed(0)
    test_corruptions()
    test_losses()
    test_models()
    test_onnx()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for name in failures:
            print(f"  - {name}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
