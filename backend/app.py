"""FastAPI inference server for the deepfake image classifier.

Preprocessing mirrors how the model was *trained* in deepfake_image.ipynb:
  - tf.keras.preprocessing.image_dataset_from_directory loads images as RGB,
    resized to 128x128 with bilinear interpolation, with pixel values left in
    the 0-255 range (the notebook adds no Rescaling layer).
  - Class indices come from the alphabetically sorted folder names
    ["Fake", "Real"], so the sigmoid output is P(Real), not P(Fake).

Testing_v1.py got both of these wrong (BGR + /255, and "> 0.5 means fake"),
so its answers were unreliable. Both settings are overridable by env var in
case a model was trained differently.

If no model file is found the server starts in DEMO mode and returns a
deterministic placeholder score so the frontend can still be exercised.
"""

from __future__ import annotations

import hashlib
import io
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).resolve().parent.parent / "models" / "deepfake_detection_updated_v3.h5"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "128"))
# "raw" = 0-255 (matches training), "unit" = divide by 255
INPUT_SCALE = os.getenv("INPUT_SCALE", "raw")
# Which class the sigmoid output represents. Alphabetical order => "real".
POSITIVE_CLASS = os.getenv("POSITIVE_CLASS", "real").lower()
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "15"))
STATIC_DIR = Path(os.getenv("STATIC_DIR", Path(__file__).resolve().parent / "static"))

app = FastAPI(title="Deepfake Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_load_error: str | None = None


def _load_model() -> None:
    global _model, _load_error
    if not MODEL_PATH.exists():
        _load_error = f"Model file not found at {MODEL_PATH}"
        return
    try:
        from tensorflow.keras.models import load_model  # imported lazily: slow

        _model = load_model(str(MODEL_PATH), compile=False)
    except Exception as exc:  # noqa: BLE001 - surface any load failure to the UI
        _load_error = f"Failed to load model: {exc}"


_load_model()


def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    if INPUT_SCALE == "unit":
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def _demo_score(data: bytes) -> float:
    """Deterministic placeholder in [0.05, 0.95]; NOT a real prediction."""
    h = int.from_bytes(hashlib.sha256(data).digest()[:4], "big")
    return 0.05 + 0.9 * (h / 0xFFFFFFFF)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "mode": "model" if _model is not None else "demo",
        "model_file": MODEL_PATH.name,
        "detail": _load_error,
        "input_size": IMG_SIZE,
        "input_scale": INPUT_SCALE,
        "positive_class": POSITIVE_CLASS,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB:g} MB")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "Could not read that file as an image")

    started = time.perf_counter()
    if _model is not None:
        output = float(_model.predict(preprocess(image), verbose=0)[0][0])
        mode = "model"
    else:
        output = _demo_score(data)
        mode = "demo"
    elapsed_ms = (time.perf_counter() - started) * 1000

    p_real = output if POSITIVE_CLASS == "real" else 1.0 - output
    p_fake = 1.0 - p_real
    label = "fake" if p_fake > 0.5 else "real"
    return {
        "label": label,
        "fake_probability": round(p_fake, 4),
        "real_probability": round(p_real, 4),
        "confidence": round(max(p_fake, p_real), 4),
        "inference_ms": round(elapsed_ms, 1),
        "mode": mode,
        "image": {"width": image.width, "height": image.height, "format": image.format},
    }


# Serve the built frontend (production / Docker). In dev, Vite serves it.
if STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        target = STATIC_DIR / path
        if path and target.is_file():
            return FileResponse(target)
        return FileResponse(STATIC_DIR / "index.html")
