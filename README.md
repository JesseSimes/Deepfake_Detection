# Deepfake_Detection
This is an image Deepfake Detection project. Using Tensor Flow and CNN.

The training data was downloaded from:
https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images/data

deepfake_image.ipynb file is used to train the data.

Testing_v1.py file is the testing of the model .h5 file.

Go through the report to understand the behaviour of the code.

THANK YOU!!

---

## Web app

A browser UI for the trained model: drag, drop or paste images and get a real/fake verdict, the probability, and a history of the session. **The model runs in the browser with TensorFlow.js**, so images never leave the visitor's device and the site deploys as static files (e.g. on Vercel).

```
frontend/               React + Vite UI; src/model.ts runs the CNN with TensorFlow.js
frontend/public/model/  exported weights (model.json + weights.bin, 13 MB)
tools/export_weights.py converts a Keras .h5 into those files
backend/                optional FastAPI server exposing the model as an HTTP API
models/                 .h5 files (git-ignored)
```

### Run locally

```
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The first visit downloads the 13 MB model once.

### Deploy to Vercel

`vercel.json` at the repo root builds `frontend/` and serves `frontend/dist`. Import the repository in Vercel and deploy; no environment variables or server are needed.

### Updating the model

After retraining, export the new weights and commit them:

```
.venv-tfScriptspython toolsexport_weights.py models<new_model>.h5 frontendpublicmodel <test_accuracy>
```

The browser runtime supports this notebook's architecture (Conv2D / MaxPooling2D / Flatten / Dropout / Dense). On 200 test images the browser and Keras gave the same verdict for all 200 (mean score difference 0.006, from JPEG decoding differences).

### Optional: Python API

`backend/app.py` serves the same model over HTTP (`/api/health`, `/api/predict`) for scripts or other apps. With the training environment:

```
.venv-tfScriptspip install fastapi==0.115.6 "uvicorn[standard]==0.32.1" python-multipart==0.0.20 pillow
.venv-tfScriptspython -m uvicorn backend.app:app --port 8000
```

Or with Docker: put the model in `models/` and run `docker compose up --build`. Without a model file the API runs in a labelled demo mode with placeholder scores.

### Preprocessing note (fixes vs. Testing_v1.py)

The app (browser and API) preprocesses images the way the notebook **trained** the model, which differs from `Testing_v1.py` in two ways:

- **Pixel scale and colour order.** `image_dataset_from_directory` yields RGB pixels in the 0–255 range and the model has no Rescaling layer. `Testing_v1.py` fed BGR (OpenCV) pixels divided by 255.
- **Label direction.** Keras assigns class indices alphabetically (`Fake`=0, `Real`=1), so the sigmoid output is P(Real). `Testing_v1.py` treated `> 0.5` as fake, which is inverted.

For the API, both can be overridden with env vars if a model was trained differently: `INPUT_SCALE=unit` (divide by 255) and `POSITIVE_CLASS=fake`. Other settings: `MODEL_PATH`, `MAX_UPLOAD_MB`, `CORS_ORIGINS`.

`backend/make_dummy_model.py` writes an untrained model with the same architecture, useful only for checking that the serving pipeline works.

### Retraining the model

`train/train.py` reproduces the notebook's training run (same CNN, 128×128 input, batch 16, 10 epochs) in Docker, on the GPU if one is available.

1. Download the dataset from Kaggle ("Deepfake and Real Images", ~1.7 GB) and extract it, so you have `Dataset\Train`, `Dataset\Validation` and `Dataset\Test`, each with `Fake` and `Real` folders.
2. Copy it into a Docker volume once (much faster to train from than a Windows folder):
   ```
   docker volume create deepfake_detection_dataset
   docker run --rm -v deepfake_detection_dataset:/data -v "C:\path\to\Dataset:/src:ro" alpine cp -r /src/. /data/
   ```
3. Train: `docker compose --profile train run --rm train`

This writes `models/deepfake_detection_retrained.h5` (point `MODEL_PATH` at it to serve it) (best epoch by validation accuracy), plus `.metrics.json` with the test accuracy and `.history.csv` with per-epoch numbers.
