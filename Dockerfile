# Serving image for the AKS/raw-Kubernetes path.
#
# Code-only image — no data or checkpoint baked in. At container startup,
# RecommenderService downloads the model checkpoint and processed feature
# parquets from Azure Blob Storage using the AZURE_STORAGE_CONNECTION_STRING
# and MODEL_VERSION environment variables injected by the Kubernetes Deployment.

# slim variant strips dev tools and docs — keeps the image small for faster AKS pulls.
FROM python:3.13-slim

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/

EXPOSE 8080

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
