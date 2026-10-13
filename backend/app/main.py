"""FastAPI backend serving all four tasks.

Endpoints
    GET  /api/health              which ONNX models are loaded
    POST /api/corrupt             apply a corruption to an upload (no model)
    POST /api/universal           Task 1
    POST /api/hard-routed         Task 2
    POST /api/soft-moe            Task 3
    POST /api/face-to-sketch      Task 4

Uploads are validated before anything touches a model: content type, byte
size, and whether PIL can actually decode the file. A 413 or 415 with a
readable message is much easier to act on than a 500 from deep inside the
inference layer.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .inference import (CORRUPTION_NAMES, STYLE_NAMES, ModelRegistry,
                        corrupt_image, decode_image, encode_image,
                        run_face_to_sketch, run_hard_routed, run_soft_moe,
                        run_universal)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
MODEL_DIR = os.getenv("MODEL_DIR", "models/onnx")
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"}

app = FastAPI(title="Generative AI Assignment 1",
              description="Universal restoration, hard routing, soft mixture of "
                          "experts, and face-to-sketch generation.",
              version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = ModelRegistry(MODEL_DIR)


@app.on_event("startup")
def _startup():
    loaded = registry.warm()
    print(f"[startup] model dir: {MODEL_DIR}")
    print(f"[startup] loaded: {loaded or 'none'}")


async def read_image(upload: UploadFile):
    if upload.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported file type {upload.content_type}. "
                                 "Upload a PNG, JPEG, WebP or BMP image.")
    raw = await upload.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image is larger than "
                                 f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not raw:
        raise HTTPException(400, "The uploaded file is empty.")
    try:
        return decode_image(raw)
    except Exception:                                    # noqa: BLE001
        raise HTTPException(400, "That file could not be read as an image.")


def guard(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))


@app.get("/api/health")
def health():
    available = registry.available()
    return {
        "status": "ok" if all(available.values()) else "degraded",
        "models": available,
        "model_dir": MODEL_DIR,
        "corruptions": list(CORRUPTION_NAMES),
        "styles": list(STYLE_NAMES),
        "image_size": 128,
    }


@app.post("/api/corrupt")
async def corrupt(file: UploadFile = File(...),
                  corruption: str = Form("salt_pepper"),
                  severity: str = Form("medium"),
                  seed: int = Form(42)):
    """Apply a corruption in the browser flow, so an evaluator can start from a
    clean photograph and choose what damage to inflict."""
    if corruption not in CORRUPTION_NAMES:
        raise HTTPException(400, f"corruption must be one of {CORRUPTION_NAMES}")
    if severity not in ("low", "medium", "high"):
        raise HTTPException(400, "severity must be low, medium or high")
    arr = await read_image(file)
    corrupted, meta = corrupt_image(arr, corruption, severity, seed)
    return {"original": encode_image(arr), "corrupted": encode_image(corrupted),
            "settings": meta}


@app.post("/api/universal")
async def universal(file: UploadFile = File(...),
                    corruption: Optional[str] = Form(None),
                    severity: str = Form("medium"),
                    seed: int = Form(42)):
    arr = await read_image(file)
    applied = {"kind": "none (image used as uploaded)"}
    if corruption and corruption != "none":
        arr, applied = corrupt_image(arr, corruption, severity, seed)
    result = guard(run_universal, registry, arr)
    return {"input": encode_image(arr), "output": result.image,
            "inference_ms": round(result.inference_ms, 2),
            "applied_corruption": applied, **(result.extra or {})}


@app.post("/api/hard-routed")
async def hard_routed(file: UploadFile = File(...),
                      corruption: Optional[str] = Form(None),
                      severity: str = Form("medium"),
                      seed: int = Form(42)):
    arr = await read_image(file)
    applied = {"kind": "none (image used as uploaded)"}
    if corruption and corruption != "none":
        arr, applied = corrupt_image(arr, corruption, severity, seed)
    result = guard(run_hard_routed, registry, arr)
    return {"input": encode_image(arr), "output": result.image,
            "inference_ms": round(result.inference_ms, 2),
            "applied_corruption": applied, **(result.extra or {})}


@app.post("/api/soft-moe")
async def soft_moe(file: UploadFile = File(...),
                   corruption: Optional[str] = Form(None),
                   severity: str = Form("medium"),
                   seed: int = Form(42)):
    arr = await read_image(file)
    applied = {"kind": "none (image used as uploaded)"}
    if corruption and corruption != "none":
        arr, applied = corrupt_image(arr, corruption, severity, seed)
    result = guard(run_soft_moe, registry, arr)
    return {"input": encode_image(arr), "output": result.image,
            "inference_ms": round(result.inference_ms, 2),
            "applied_corruption": applied, **(result.extra or {})}


@app.post("/api/face-to-sketch")
async def face_to_sketch(file: UploadFile = File(...), style: int = Form(0)):
    if style not in (0, 1, 2):
        raise HTTPException(400, "style must be 0, 1 or 2")
    arr = await read_image(file)
    result = guard(run_face_to_sketch, registry, arr, style)
    return {"input": encode_image(arr), "output": result.image,
            "inference_ms": round(result.inference_ms, 2), **(result.extra or {})}
