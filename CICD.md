# CI/CD & Model Update Playbook

This document covers how code changes and new model versions flow from development to production on AKS.

---

## GitHub Actions Workflows

Two workflows live in `.github/workflows/`:

| File | Trigger | What it does |
|---|---|---|
| `ci.yml` | Every pull request to `main` | Installs deps, runs model smoke test (forward pass with synthetic data) |
| `cd.yml` | Every merge to `main` | Builds Docker image, pushes to ACR, deploys to AKS |

### One-time Setup — GitHub Secrets

Go to: GitHub repo → **Settings → Secrets and variables → Actions → New repository secret**

| Secret | How to get it |
|---|---|
| `ACR_USERNAME` | `az acr credential show --name hmrecommenderacr --query "username" -o tsv` |
| `ACR_PASSWORD` | `az acr credential show --name hmrecommenderacr --query "passwords[0].value" -o tsv` |
| `KUBE_CONFIG` | `cat ~/.kube/config \| base64 -w 0` |

---

## How to Ship a New Model Feature (e.g. Image Embeddings)

### Step 1 — Develop on a branch

```bash
git checkout -b feature/image-embeddings
# make changes to src/models/two_tower.py, src/data/preprocess.py, etc.
git push origin feature/image-embeddings
```

### Step 2 — Open a Pull Request

CI runs automatically. It must pass before merging:
- Installs all dependencies
- Instantiates the model with synthetic vocab sizes
- Runs a forward pass and checks loss > 0

### Step 3 — Merge to main

CD runs automatically:
- Builds new Docker serving image tagged with the commit SHA
- Pushes to `hmrecommenderacr.azurecr.io/hm-recommender:<sha>`
- Runs `kubectl set image` to update the AKS deployment
- Waits for rollout to complete (5 min timeout)

No manual Docker or kubectl steps needed.

### Step 4 — Retrain on Azure compute

The new model architecture needs to be retrained on cloud compute with the versioned data:

```bash
python azure/pipeline.py --data-version v3 --epochs 30
```

This submits two sequential jobs to `hm-training-cluster`:
1. `train` — reads from `hm-processed-data:3`, writes checkpoint to `checkpoints/v3/two_tower.pt` in Blob Storage
2. `evaluate` — reports Recall@K, NDCG@K, MAP@12 in Azure ML Experiments

Monitor progress: Azure ML Studio → Jobs → hm-two-tower-recommender

### Step 5 — Register the new model version

```bash
az ml model create \
  --name hm-two-tower \
  --version 2 \
  --path azureml://datastores/workspaceblobstore/paths/checkpoints/v3/two_tower.pt \
  --type custom_model \
  --resource-group resource_jhu_rec_sys \
  --workspace-name JHU_rec_sys
```

Appears under Azure ML Studio → Models → hm-two-tower → version 2.

### Step 6 — Swap the live model on AKS

No image rebuild needed — just update the env var. The running container downloads the new checkpoint from Blob Storage on restart:

```bash
kubectl set env deployment/hm-two-tower-recommender MODEL_VERSION=v3
kubectl rollout status deployment/hm-two-tower-recommender
```

### Step 7 — Test the live endpoint

```bash
curl -X POST http://20.161.92.233/recommend \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "<any-customer-id>"}'
```

---

## Azure Resources Quick Reference

| Resource | Name | Purpose |
|---|---|---|
| Resource group | `resource_jhu_rec_sys` | Contains everything |
| Storage account | `jhurecsys0742245294` | Blob Storage — data, features, checkpoints |
| Blob container | `azureml-blobstore-7157eb24-5268-482d-adb1-10eb89c8a1c2` | Default workspace container |
| ML workspace | `JHU_rec_sys` | Azure ML experiments, models, data assets |
| Compute cluster | `hm-training-cluster` | Runs training jobs (scales to 0 when idle) |
| Container registry | `hmrecommenderacr` | Stores Docker serving images |
| AKS cluster | `hm-recommender-aks` | Runs the live serving container |
| Public IP | `20.161.92.233` | Live recommendation endpoint |

## Key Blob Storage Paths

| Path | Contents |
|---|---|
| `raw/` | Original Kaggle CSVs |
| `processed/v1/` | Features for Jun 2020 cutoff |
| `processed/v2/` | Features for Jul 2020 cutoff |
| `processed/v3/` | Features for Aug 2020 cutoff |
| `checkpoints/v1/` | Trained model checkpoint for v1 |
