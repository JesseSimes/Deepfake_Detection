"""Export the trained Keras model's weights for in-browser inference.

Writes frontend/public/model/{weights.bin,model.json}. The browser
(frontend/src/model.ts) rebuilds the same layers with TensorFlow.js ops, so
this only supports the notebook's architecture: Conv2D/MaxPooling2D blocks,
Flatten, Dense (Dropout is a no-op at inference).

    .venv-tf\\Scripts\\python tools\\export_weights.py models\\deepfake_detection_updated_v3.h5
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import keras  # noqa: E402
import numpy as np  # noqa: E402

src = Path(sys.argv[1] if len(sys.argv) > 1 else "models/deepfake_detection_updated_v3.h5")
out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "frontend/public/model")
test_accuracy = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8727

model = keras.models.load_model(src, compile=False)
out_dir.mkdir(parents=True, exist_ok=True)

layers, blobs, offset = [], [], 0
for layer in model.layers:
    kind = type(layer).__name__
    entry = {"name": layer.name, "type": kind}
    if kind == "Conv2D":
        assert layer.padding == "valid" and tuple(layer.strides) == (1, 1), "unsupported conv config"
        entry["activation"] = layer.activation.__name__
    elif kind == "MaxPooling2D":
        assert layer.padding == "valid", "unsupported pooling config"
        entry["pool"] = list(layer.pool_size)
        entry["strides"] = list(layer.strides)
    elif kind == "Dense":
        entry["activation"] = layer.activation.__name__
    elif kind in ("Flatten", "Dropout"):
        pass
    else:
        raise SystemExit(f"Layer type {kind} isn't supported by the browser runtime")

    weights = []
    for w in layer.get_weights():
        arr = np.ascontiguousarray(w, dtype="<f4")
        weights.append({"shape": list(arr.shape), "offset": offset})
        blobs.append(arr.tobytes())
        offset += arr.size
    if weights:
        entry["weights"] = weights
    layers.append(entry)

(out_dir / "weights.bin").write_bytes(b"".join(blobs))
manifest = {
    "format": "deepfake-cnn/1",
    "source": src.name,
    "input": {"size": list(model.input_shape[1:3]), "channels": 3, "scale": "0-255 RGB"},
    "output": "sigmoid = P(Real); classes sorted ['Fake', 'Real']",
    "test_accuracy": test_accuracy,
    "weights_file": "weights.bin",
    "weights_bytes": offset * 4,
    "layers": layers,
}
(out_dir / "model.json").write_text(json.dumps(manifest, indent=2))
print(f"Wrote {out_dir}/weights.bin ({offset * 4 / 1e6:.1f} MB) and model.json, {len(layers)} layers")
