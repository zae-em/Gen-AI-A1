"""Corruption operators for Tasks 1-3.

Every corruption is described by a `CorruptionSpec` which is fully
serialisable.  That is what makes the validation / test manifests
deterministic: we store the spec, not the corrupted pixels.

Conventions
-----------
* Images are float32 torch tensors, shape (3, H, W), range [0, 1].
* `CORRUPTION_NAMES` fixes the class order used *everywhere*
  (classifier logits, gate weights, ONNX outputs, frontend labels).
  Do not reorder it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as TF

CORRUPTION_NAMES: Tuple[str, ...] = ("clean", "salt_pepper", "blur", "occlusion")
NAME_TO_INDEX: Dict[str, int] = {n: i for i, n in enumerate(CORRUPTION_NAMES)}

# --- training-time sampling ranges (assignment spec) ------------------------
SP_PROB_RANGE = (0.02, 0.15)
BLUR_KERNELS = (3, 5, 7)
BLUR_SIGMA_RANGE = (0.5, 2.5)
OCC_N_RECTS_RANGE = (1, 3)
OCC_AREA_RANGE = (0.10, 0.35)

# --- fixed test severities (assignment spec) --------------------------------
TEST_SP_PROBS = (0.03, 0.08, 0.15)
TEST_BLUR_CONFIGS = ((3, 0.7), (5, 1.5), (7, 2.5))
TEST_OCC_CONFIGS = ((0.10, 1), (0.20, 2), (0.35, 3))  # (area fraction, n rects)
SEVERITY_NAMES = ("low", "medium", "high")


@dataclass
class CorruptionSpec:
    """Fully-specified, reproducible description of one corruption."""

    kind: str                                   # one of CORRUPTION_NAMES
    severity: Optional[str] = None              # low / medium / high, None for clean
    prob: Optional[float] = None                # salt & pepper
    kernel: Optional[int] = None                # blur
    sigma: Optional[float] = None               # blur
    rects: List[Tuple[int, int, int, int]] = field(default_factory=list)  # x, y, w, h
    seed: Optional[int] = None

    @property
    def label(self) -> int:
        return NAME_TO_INDEX[self.kind]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["rects"] = [list(r) for r in self.rects]
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CorruptionSpec":
        d = dict(d)
        d["rects"] = [tuple(r) for r in d.get("rects", [])]
        return CorruptionSpec(**d)


# ---------------------------------------------------------------------------
# primitive operators
# ---------------------------------------------------------------------------
def apply_salt_pepper(img: torch.Tensor, prob: float,
                      generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Replace a fraction `prob` of pixels with pure black or pure white.

    The decision is made per *pixel* (not per channel), so a corrupted pixel is
    (0,0,0) or (1,1,1).  That matches how salt-and-pepper noise appears on a
    real sensor and keeps the artefact visually unambiguous for the classifier.
    """
    c, h, w = img.shape
    noise = torch.rand(h, w, generator=generator, device=img.device)
    selected = noise < prob
    salt = torch.rand(h, w, generator=generator, device=img.device) < 0.5

    out = img.clone()
    out[:, selected & salt] = 1.0
    out[:, selected & ~salt] = 0.0
    return out


def apply_blur(img: torch.Tensor, kernel: int, sigma: float) -> torch.Tensor:
    """Gaussian blur. Deterministic given (kernel, sigma) - no RNG needed."""
    return TF.gaussian_blur(img, kernel_size=[int(kernel), int(kernel)],
                            sigma=[float(sigma), float(sigma)])


def apply_occlusion(img: torch.Tensor,
                    rects: List[Tuple[int, int, int, int]]) -> torch.Tensor:
    out = img.clone()
    for (x, y, w, h) in rects:
        out[:, y:y + h, x:x + w] = 0.0
    return out


def sample_rects(height: int, width: int, n_rects: int, area_fraction: float,
                 generator: Optional[torch.Generator] = None,
                 max_tries: int = 40) -> List[Tuple[int, int, int, int]]:
    """Sample `n_rects` axis-aligned rectangles jointly covering ~`area_fraction`.

    The target area is split evenly between rectangles and each one gets a
    random aspect ratio in [0.5, 2.0].  Rectangles may overlap, so realised
    coverage can fall slightly below the target; `realised_coverage` in
    manifest.py records the true value so the report can quote measured
    numbers rather than nominal ones.
    """
    target_area = area_fraction * height * width
    per_rect = target_area / n_rects
    rects: List[Tuple[int, int, int, int]] = []

    for _ in range(n_rects):
        for _try in range(max_tries):
            ratio = 0.5 + 1.5 * torch.rand(1, generator=generator).item()
            rw = int(round((per_rect * ratio) ** 0.5))
            rh = int(round((per_rect / ratio) ** 0.5))
            rw, rh = max(1, min(rw, width)), max(1, min(rh, height))
            if rw <= width and rh <= height:
                break
        x = int(torch.randint(0, max(1, width - rw + 1), (1,), generator=generator).item())
        y = int(torch.randint(0, max(1, height - rh + 1), (1,), generator=generator).item())
        rects.append((x, y, rw, rh))
    return rects


def coverage_of(rects, height: int, width: int) -> float:
    """Realised (union, not summed) coverage fraction of a rectangle list."""
    mask = torch.zeros(height, width, dtype=torch.bool)
    for (x, y, w, h) in rects:
        mask[y:y + h, x:x + w] = True
    return mask.float().mean().item()


# ---------------------------------------------------------------------------
# spec sampling + application
# ---------------------------------------------------------------------------
def _uniform(lo: float, hi: float, generator=None) -> float:
    return lo + (hi - lo) * torch.rand(1, generator=generator).item()


def sample_spec(height: int, width: int,
                generator: Optional[torch.Generator] = None,
                force_kind: Optional[str] = None) -> CorruptionSpec:
    """Sample one of the four input conditions with equal probability.

    Used by the training data loader, where a fresh corruption is drawn every
    time an image is read.  `force_kind` is used to build balanced batches for
    the Task 2 classifier and for the per-corruption specialists.
    """
    if force_kind is None:
        idx = int(torch.randint(0, len(CORRUPTION_NAMES), (1,), generator=generator).item())
        kind = CORRUPTION_NAMES[idx]
    else:
        kind = force_kind

    if kind == "clean":
        return CorruptionSpec(kind="clean")

    if kind == "salt_pepper":
        return CorruptionSpec(kind="salt_pepper", prob=_uniform(*SP_PROB_RANGE, generator=generator))

    if kind == "blur":
        k = BLUR_KERNELS[int(torch.randint(0, len(BLUR_KERNELS), (1,), generator=generator).item())]
        return CorruptionSpec(kind="blur", kernel=int(k),
                              sigma=_uniform(*BLUR_SIGMA_RANGE, generator=generator))

    if kind == "occlusion":
        n = int(torch.randint(OCC_N_RECTS_RANGE[0], OCC_N_RECTS_RANGE[1] + 1, (1,),
                              generator=generator).item())
        area = _uniform(*OCC_AREA_RANGE, generator=generator)
        return CorruptionSpec(kind="occlusion",
                              rects=sample_rects(height, width, n, area, generator))

    raise ValueError(f"unknown corruption kind: {kind}")


def sample_test_spec(height: int, width: int, kind: str, severity_idx: int,
                     generator: Optional[torch.Generator] = None) -> CorruptionSpec:
    """Build one of the three fixed test severities for a given corruption."""
    sev = SEVERITY_NAMES[severity_idx]

    if kind == "clean":
        return CorruptionSpec(kind="clean")

    if kind == "salt_pepper":
        return CorruptionSpec(kind="salt_pepper", severity=sev,
                              prob=TEST_SP_PROBS[severity_idx])

    if kind == "blur":
        k, s = TEST_BLUR_CONFIGS[severity_idx]
        return CorruptionSpec(kind="blur", severity=sev, kernel=int(k), sigma=float(s))

    if kind == "occlusion":
        area, n = TEST_OCC_CONFIGS[severity_idx]
        return CorruptionSpec(kind="occlusion", severity=sev,
                              rects=sample_rects(height, width, n, area, generator))

    raise ValueError(f"unknown corruption kind: {kind}")


def apply_spec(img: torch.Tensor, spec: CorruptionSpec,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Apply a spec to a (3, H, W) float tensor in [0, 1]."""
    if spec.kind == "clean":
        return img.clone()
    if spec.kind == "salt_pepper":
        return apply_salt_pepper(img, float(spec.prob), generator)
    if spec.kind == "blur":
        return apply_blur(img, int(spec.kernel), float(spec.sigma))
    if spec.kind == "occlusion":
        return apply_occlusion(img, spec.rects)
    raise ValueError(f"unknown corruption kind: {spec.kind}")
