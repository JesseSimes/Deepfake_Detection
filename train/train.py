"""Retrain the deepfake image classifier.

Same recipe as deepfake_image.ipynb (same data pipeline, CNN, optimizer and
10 epochs), packaged as a script so it can run in Docker. Additions:
  - keeps the epoch with the best validation accuracy, not just the last one
  - checks the class order is ["Fake", "Real"], which the API assumes
  - writes the test-set results next to the model as JSON

Expected layout (as extracted from the Kaggle dataset):
    <data>/Train/{Fake,Real}  <data>/Validation/{Fake,Real}  <data>/Test/{Fake,Real}

    python train.py --data /data --out /models/deepfake_detection_updated_v2.h5
"""

import argparse
import json
import time
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--img-size", type=int, default=128)
args = p.parse_args()

data = Path(args.data)
size = (args.img_size, args.img_size)
print("GPUs:", tf.config.list_physical_devices("GPU") or "none (training on CPU)")


def load(split: str, shuffle: bool):
    return tf.keras.utils.image_dataset_from_directory(
        data / split, image_size=size, batch_size=args.batch_size,
        label_mode="binary", shuffle=shuffle, seed=42,
    )


train_ds = load("Train", True)
val_ds = load("Validation", False)
test_ds = load("Test", False)

class_names = train_ds.class_names
if class_names != ["Fake", "Real"]:
    raise SystemExit(f"Unexpected class folders {class_names}; the API assumes ['Fake', 'Real'].")

AUTOTUNE = tf.data.AUTOTUNE
train_ds, val_ds, test_ds = (d.prefetch(AUTOTUNE) for d in (train_ds, val_ds, test_ds))

# Identical to create_model() in the notebook. Pixels stay in 0-255 on purpose,
# matching the original training (the API preprocesses the same way).
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(*size, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),  # P(Real): class index 1
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
started = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(str(out), monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(out.with_suffix(".history.csv"))),
    ],
)

best = models.load_model(str(out))
test_loss, test_acc = best.evaluate(test_ds)
report = {
    "model_file": out.name,
    "test_accuracy": round(float(test_acc), 4),
    "test_loss": round(float(test_loss), 4),
    "best_val_accuracy": round(float(max(history.history["val_accuracy"])), 4),
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "class_names": class_names,
    "output_meaning": "sigmoid = P(Real)",
    "input": f"{size[0]}x{size[1]} RGB, 0-255",
    "train_minutes": round((time.time() - started) / 60, 1),
}
out.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
print(f"Test accuracy: {test_acc * 100:.2f}%")
