"""Write an UNTRAINED model with the notebook's exact architecture.

Only for checking the serving pipeline end to end when the real
deepfake_detection_updated_v2.h5 isn't at hand. Its predictions are noise.

    python make_dummy_model.py ../models/dummy_untrained.h5
"""

import sys

from tensorflow.keras import layers, models

out = sys.argv[1] if len(sys.argv) > 1 else "dummy_untrained.h5"
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),
])
model.save(out)
print(f"wrote {out}")
