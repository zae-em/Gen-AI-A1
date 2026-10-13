"""ONNX Runtime inference layer.

Sessions are created lazily and cached, so the container starts even when only
some of the models have been trained yet - `/api/health` reports exactly which
ones are present.  That matters during development and it means a missing file
produces a clear message in the interface rather than a container that will
not boot.
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGE_SIZE = 128
CORRUPTION_NAMES = ("clean", "salt_pepper", "blur", "occlusion")
BRANCH_LABELS = ("identity (clean)", "salt & pepper expert", "blur expert",
                 "occlusion expert")
STYLE_NAMES = ("Style 1", "Style 2", "Style 3")

MODEL_FILES = {
    "task1": "task1_universal_ae.onnx",
    "classifier": "task2_classifier.onnx",
    "specialist_salt_pepper": "task2_specialist_salt_pepper.onnx",
    "specialist_blur": "task2_specialist_blur.onnx",
    "specialist_occlusion": "task2_specialist_occlusion.onnx",
    "soft_moe": "task3_soft_moe.onnx",
    "generator": "task4_generator.onnx",
}


@dataclass
class InferenceResult:
    image: str                      # base64 PNG data URI
    inference_ms: float
    extra: Optional[dict] = None


class ModelRegistry:
    def __init__(self, model_dir: str | Path = "models/onnx"):
        self.model_dir = Path(model_dir)
        self._sessions: Dict[str, ort.InferenceSession] = {}

    def path(self, key: str) -> Path:
        return self.model_dir / MODEL_FILES[key]

    def available(self) -> Dict[str, bool]:
        return {k: self.path(k).exists() for k in MODEL_FILES}

    def session(self, key: str) -> ort.InferenceSession:
        if key not in self._sessions:
            p = self.path(key)
            if not p.exists():
                raise FileNotFoundError(
                    f"{p.name} is missing. Train the model and run "
                    f"`python -m src.export.export_onnx --all`, or download the "
                    f"released weights into {self.model_dir}/.")
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 2
            self._sessions[key] = ort.InferenceSession(
                str(p), opts, providers=["CPUExecutionProvider"])
        return self._sessions[key]

    def warm(self) -> List[str]:
        loaded = []
        for key, exists in self.available().items():
            if exists:
                try:
                    self.session(key)
                    loaded.append(key)
                except Exception as exc:                    # noqa: BLE001
                    print(f"[registry] could not load {key}: {exc}")
        return loaded


# ---------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------
def decode_image(raw: bytes, size: int = IMAGE_SIZE) -> np.ndarray:
    """bytes -> (1, 3, size, size) float32 in [0, 1]."""
    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB").resize((size, size), Image.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None]


def encode_image(arr: np.ndarray) -> str:
    """(3, H, W) or (1, 3, H, W) float in [0, 1] -> base64 PNG data URI."""
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.clip(arr, 0.0, 1.0).transpose(1, 2, 0)
    img = Image.fromarray((arr * 255).round().astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def error_map(a: np.ndarray, b: np.ndarray) -> str:
    """Absolute-difference heat map, mean over channels, scaled 0-0.5."""
    diff = np.abs(a[0] - b[0]).mean(0)
    scaled = np.clip(diff / 0.5, 0, 1)
    # simple black-red-yellow-white ramp, no matplotlib in the container
    r = np.clip(scaled * 3, 0, 1)
    g = np.clip(scaled * 3 - 1, 0, 1)
    bl = np.clip(scaled * 3 - 2, 0, 1)
    return encode_image(np.stack([r, g, bl]))


# ---------------------------------------------------------------------------
# runtime corruption (mirrors src/data/corruptions.py, numpy version)
# ---------------------------------------------------------------------------
def corrupt_image(arr: np.ndarray, kind: str, severity: str = "medium",
                  seed: int = 42) -> Tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    _, _, h, w = arr.shape
    out = arr.copy()

    if kind == "clean":
        return out, {"kind": "clean"}

    if kind == "salt_pepper":
        prob = {"low": 0.03, "medium": 0.08, "high": 0.15}[severity]
        selected = rng.random((h, w)) < prob
        salt = rng.random((h, w)) < 0.5
        out[0][:, selected & salt] = 1.0
        out[0][:, selected & ~salt] = 0.0
        return out, {"kind": kind, "severity": severity, "probability": prob}

    if kind == "blur":
        kernel, sigma = {"low": (3, 0.7), "medium": (5, 1.5), "high": (7, 2.5)}[severity]
        out = _gaussian_blur(out, kernel, sigma)
        return out, {"kind": kind, "severity": severity,
                     "kernel": kernel, "sigma": sigma}

    if kind == "occlusion":
        area, n = {"low": (0.10, 1), "medium": (0.20, 2), "high": (0.35, 3)}[severity]
        rects = []
        per = area * h * w / n
        for _ in range(n):
            ratio = float(rng.uniform(0.5, 2.0))
            rw = max(1, min(int(round((per * ratio) ** 0.5)), w))
            rh = max(1, min(int(round((per / ratio) ** 0.5)), h))
            x = int(rng.integers(0, max(1, w - rw + 1)))
            y = int(rng.integers(0, max(1, h - rh + 1)))
            out[0][:, y:y + rh, x:x + rw] = 0.0
            rects.append([x, y, rw, rh])
        return out, {"kind": kind, "severity": severity,
                     "target_area": area, "rectangles": rects}

    raise ValueError(f"unknown corruption: {kind}")


def _gaussian_blur(arr: np.ndarray, kernel: int, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with reflect padding - matches torchvision."""
    radius = kernel // 2
    coords = np.arange(kernel) - radius
    k = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    k = (k / k.sum()).astype(np.float32)

    out = arr[0]
    padded = np.pad(out, ((0, 0), (0, 0), (radius, radius)), mode="reflect")
    tmp = sum(k[i] * padded[:, :, i:i + out.shape[2]] for i in range(kernel))
    padded = np.pad(tmp, ((0, 0), (radius, radius), (0, 0)), mode="reflect")
    res = sum(k[i] * padded[:, i:i + out.shape[1], :] for i in range(kernel))
    return res[None].astype(np.float32)


# ---------------------------------------------------------------------------
# task pipelines
# ---------------------------------------------------------------------------
def run_universal(registry: ModelRegistry, arr: np.ndarray) -> InferenceResult:
    sess = registry.session("task1")
    start = time.perf_counter()
    out = sess.run(None, {"image": arr})[0]
    elapsed = (time.perf_counter() - start) * 1000
    return InferenceResult(encode_image(out), elapsed,
                           {"error_map": error_map(out, arr)})


def run_hard_routed(registry: ModelRegistry, arr: np.ndarray) -> InferenceResult:
    cls = registry.session("classifier")
    start = time.perf_counter()
    outputs = cls.run(None, {"image": arr})
    if len(outputs) == 1:
        # The imported assignment classifier exports logits only.
        logits = outputs[0]
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
    else:
        _logits, probs = outputs
    predicted = int(np.argmax(probs[0]))
    kind = CORRUPTION_NAMES[predicted]

    if kind == "clean":
        out = arr                                    # identity bypass
        expert_used = "identity bypass"
    else:
        out = registry.session(f"specialist_{kind}").run(None, {"image": arr})[0]
        expert_used = f"{kind} specialist"
    elapsed = (time.perf_counter() - start) * 1000

    return InferenceResult(encode_image(out), elapsed, {
        "probabilities": {n: float(p) for n, p in zip(CORRUPTION_NAMES, probs[0])},
        "predicted_corruption": kind,
        "confidence": float(probs[0][predicted]),
        "expert_used": expert_used,
        "error_map": error_map(out, arr),
    })


def run_soft_moe(registry: ModelRegistry, arr: np.ndarray) -> InferenceResult:
    sess = registry.session("soft_moe")
    start = time.perf_counter()
    out, weights = sess.run(None, {"image": arr})
    elapsed = (time.perf_counter() - start) * 1000

    w = weights[0].astype(float)
    order = np.argsort(-w)
    return InferenceResult(encode_image(out), elapsed, {
        "weights": {label: float(v) for label, v in zip(BRANCH_LABELS, w)},
        "dominant_branch": BRANCH_LABELS[int(order[0])],
        "dominant_weight": float(w[order[0]]),
        "second_branch": BRANCH_LABELS[int(order[1])],
        "second_weight": float(w[order[1]]),
        "is_blended": bool(w[order[1]] > 0.25),
        "error_map": error_map(out, arr),
    })


def run_face_to_sketch(registry: ModelRegistry, arr: np.ndarray,
                       style: int) -> InferenceResult:
    sess = registry.session("generator")
    style_arr = np.array([style], dtype=np.int64)
    image_input = next(item.name for item in sess.get_inputs() if item.name != "style")
    # The assignment GAN uses RGB [0, 1] and a sigmoid output. The original
    # scaffold exports a "photo" input in [-1, 1] and a tanh output.
    photo = (arr * 2.0 - 1.0).astype(np.float32) if image_input == "photo" else arr
    start = time.perf_counter()
    out = sess.run(None, {image_input: photo, "style": style_arr})[0]
    elapsed = (time.perf_counter() - start) * 1000
    sketch = (out + 1.0) / 2.0 if image_input == "photo" else out
    return InferenceResult(encode_image(sketch), elapsed,
                           {"style": STYLE_NAMES[style], "style_index": style})
