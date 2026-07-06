# Serving image for the AKS/raw-Kubernetes path. The Azure ML Managed
# Online Endpoint path does NOT use this file — Azure ML builds its own
# container from a conda/pip environment spec instead.
#
# Bakes in the trained checkpoint + the processed data serve.py needs
# at startup, so the container is self-contained and runnable with no
# external volume mounts. In a real production setup these would more
# likely come from a mounted volume or be pulled from Blob storage at
# startup instead of being baked into the image (the image shouldn't
# need to be rebuilt every time the model retrains) -- simplest path
# for now, worth revisiting later.

FROM python:3.13-slim

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
COPY checkpoints/26w/two_tower.pt checkpoints/26w/two_tower.pt
COPY data/processed/customers_features.parquet data/processed/articles_features.parquet data/processed/train.parquet data/processed/encoders.pkl data/processed/

EXPOSE 8080

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
