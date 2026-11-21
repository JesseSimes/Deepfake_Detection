# Deepfake_Detection
This is an image Deepfake Detection project. Using Tensor Flow and CNN.

The training data was downloaded from:
https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images/data

deepfake_image.ipynb file is used to train the data.

Testing_v1.py file is the testing of the model .h5 file.

Go through the report to understand the behaviour of the code.

THANK YOU!!

---

## Web app (frontend + API)

A browser UI for the trained model: drag, drop or paste images and get a real/fake verdict, the probability, and a history of the session.

```
backend/   FastAPI server: loads the .h5 model, exposes /api/health and /api/predict
frontend/  React + Vite UI
models/    put deepfake_detection_updated_v3.h5 here (git-ignored)
```

### Run locally with the training environment (quickest)

`.venv-tf` already has TensorFlow 2.21, so it only needs the web server packages:

```
.venv-tf\Scripts\pip install fastapi==0.115.6 "uvicorn[standard]==0.32.1" python-multipart==0.0.20 pillow
cd frontend; npm install; npm run build; cd ..
$env:STATIC_DIR = "frontend/dist"
.venv-tf\Scripts\python -m uvicorn backend.app:app --port 8000
```

Then open http://localhost:8000.

### Run with Docker (no local Python needed)

1. Copy `deepfake_detection_updated_v3.h5` into `models/`.
2. `docker compose up --build`
3. Open http://localhost:8000

If no model file is found the app starts in **demo mode**: it works end to end but the scores are placeholders, and the UI says so.

### Run for development

```
# terminal 1: API (Python 3.9–3.11)
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload --port 8000

# terminal 2: UI with hot reload, proxies /api to :8000
cd frontend && npm install && npm run dev
```

Then open http://localhost:5173.

### Preprocessing note (fixes vs. Testing_v1.py)

The API preprocesses images the way the notebook **trained** the model, which differs from `Testing_v1.py` in two ways:

- **Pixel scale and colour order.** `image_dataset_from_directory` yields RGB pixels in the 0–255 range and the model has no Rescaling layer. `Testing_v1.py` fed BGR (OpenCV) pixels divided by 255.
- **Label direction.** Keras assigns class indices alphabetically (`Fake`=0, `Real`=1), so the sigmoid output is P(Real). `Testing_v1.py` treated `> 0.5` as fake, which is inverted.

Both can be overridden with env vars if a model was trained differently: `INPUT_SCALE=unit` (divide by 255) and `POSITIVE_CLASS=fake`. Other settings: `MODEL_PATH`, `MAX_UPLOAD_MB`, `CORS_ORIGINS`.

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
