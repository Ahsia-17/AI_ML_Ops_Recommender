# H&M Two-Tower Recommender — Azure Deployment Guide

This document tracks every step to productionalize the recommender system on Azure.
The goal is a proper MLOps setup: versioned data, tracked experiments, registered models, and a live Kubernetes endpoint.

## Architecture Overview

```
Azure Blob Storage (raw CSVs + versioned processed features)
    ↓
Azure ML Data Assets (versioned pointers to processed feature folders)
    ↓
Azure ML Pipeline (sample → preprocess → train → evaluate, fully tracked)
    ↓
Azure ML Experiments (logs Recall@12, MAP@12, loss per run)
    ↓
Azure ML Model Registry (versioned model checkpoints)
    ↓
Docker Image (code only, pulls model + features from Blob at startup)
    ↓
Azure Container Registry (stores the Docker image)
    ↓
AKS Cluster (runs the image, exposes live endpoint)
```

---

## Versioning Strategy

We simulate a live production system by treating different date ranges of the H&M dataset as separate monthly data releases. The full dataset spans Sep 2018 – Sep 2020.

| Version | Cutoff date | Simulates |
|---|---|---|
| v1 | 2020-06-30 | First month live |
| v2 | 2020-07-31 | Second month — more recent data |
| v3 | 2020-08-31 | Third month — most recent |

Each version produces its own processed feature folder in Blob Storage, a registered Data Asset, a trained model, and tracked metrics in Azure ML Experiments.

**Note on Feature Store:** Azure ML's managed Feature Store requires a separate workspace and significant setup overhead. Instead we use versioned folders in Blob Storage (`processed/v1/`, `processed/v2/`, `processed/v3/`) as a lightweight feature store. Each folder contains the same set of precomputed feature parquets — same concept, less infrastructure. Both training and serving read from the same versioned folder, which prevents train/serve skew.

---

## Step 1 — Azure Blob Storage ✅

**What it is:** Azure's file storage service (equivalent to AWS S3). Acts as the central data lake that both the training pipeline and the serving container can access. Nothing gets stored only on your laptop.

**Why it matters:** Without this, data and model weights are baked into the Docker image — you lose all versioning and have to rebuild the entire image every time the model retrains.

**What we did:**
- Azure ML automatically created a storage account (`jhurecsys0742245294`) and a default blob container (`azureml-blobstore-7157eb24-5268-482d-adb1-10eb89c8a1c2`) when the workspace was set up
- No need to create a new storage account

**Blob Storage layout:**
```
azureml-blobstore-7157eb24-.../
  raw/
    articles.csv
    customers.csv
    transactions_train.csv
  processed/
    v1/   ← cutoff 2020-06-30
    v2/   ← cutoff 2020-07-31
    v3/   ← cutoff 2020-08-31
  checkpoints/
    v1/two_tower.pt
    v2/two_tower.pt
    v3/two_tower.pt
```

---

## Step 2 — Upload Raw CSVs to Blob Storage ✅

**What it is:** Pushing the three raw Kaggle CSV files up to Blob Storage so the training pipeline can read from a central location instead of your local machine.

**Why it matters:** This is the start of data versioning. Once the files are in Blob Storage, Azure ML can track which version of the data was used to train which model.

**Files uploaded to `raw/` folder:**
- `articles.csv` — product catalog (~105k articles, 35MB)
- `customers.csv` — customer profiles (198MB)
- `transactions_train.csv` — 31M purchase records (3.3GB)

**Command used:**
```bash
az storage blob upload-batch \
  --account-name jhurecsys0742245294 \
  --destination "azureml-blobstore-7157eb24-5268-482d-adb1-10eb89c8a1c2" \
  --source "data/raw/" \
  --destination-path "raw/" \
  --auth-mode key
```

---

## Step 3 — Register Raw Data as Azure ML Data Asset ✅

**What it is:** A Data Asset is Azure ML's versioned, named pointer to data in Blob Storage. Instead of hardcoding blob paths everywhere, pipeline steps reference `hm-raw-data:1` and Azure ML resolves where to find it.

**What we did:** Created `hm-raw-data` version 1 via Azure ML Studio UI pointing to `workspaceblobstore/raw/`.

---

## Step 4 — Generate and Upload Versioned Processed Features ✅ (in progress)

**What it is:** For each date version, run `sample.py` with a cutoff date, then `preprocess.py` to compute features, then upload the output folder to Blob Storage. This is our lightweight feature store — three versioned snapshots of precomputed customer and article features.

**Why it matters:** Each training run reads from a specific version folder. If you need to reproduce model v2, you use `processed/v2/` — the exact features that produced it. This is the same guarantee a managed feature store provides, implemented with versioned folders.

**Commands for each version:**
```bash
# v1 — cutoff June 2020
python -m src.data.sample --cutoff 2020-06-30 --weeks 26
python -m src.data.preprocess --version v1
az storage blob upload-batch \
  --account-name jhurecsys0742245294 \
  --destination "azureml-blobstore-7157eb24-5268-482d-adb1-10eb89c8a1c2" \
  --source "data/processed/v1/" \
  --destination-path "processed/v1/" \
  --auth-mode key

# v2 — cutoff July 2020
python -m src.data.sample --cutoff 2020-07-31 --weeks 26
python -m src.data.preprocess --version v2
az storage blob upload-batch \
  --account-name jhurecsys0742245294 \
  --destination "azureml-blobstore-7157eb24-5268-482d-adb1-10eb89c8a1c2" \
  --source "data/processed/v2/" \
  --destination-path "processed/v2/" \
  --auth-mode key

# v3 — cutoff August 2020
python -m src.data.sample --cutoff 2020-08-31 --weeks 26
python -m src.data.preprocess --version v3
az storage blob upload-batch \
  --account-name jhurecsys0742245294 \
  --destination "azureml-blobstore-7157eb24-5268-482d-adb1-10eb89c8a1c2" \
  --source "data/processed/v3/" \
  --destination-path "processed/v3/" \
  --auth-mode key
```

**Output per version:** `customers_features.parquet`, `articles_features.parquet`, `train.parquet`, `val.parquet`, `test.parquet`, `encoders.pkl`

---

## Step 5 — Register Processed Versions as Azure ML Data Assets ⬜

**What it is:** Register each processed version folder as a named Data Asset in Azure ML so pipeline steps and training runs can reference them by name and version instead of raw blob paths.

**Why it matters:** Links each training run to the exact feature version it used. Visible in Azure ML Studio — every experiment shows which data version produced it.

**Commands:**
```bash
# v1
az ml data create --name hm-processed-data --version 1 \
  --type uri_folder \
  --path azureml://datastores/workspaceblobstore/paths/processed/v1/ \
  --resource-group resource_jhu_rec_sys --workspace-name JHU_rec_sys

# v2
az ml data create --name hm-processed-data --version 2 \
  --type uri_folder \
  --path azureml://datastores/workspaceblobstore/paths/processed/v2/ \
  --resource-group resource_jhu_rec_sys --workspace-name JHU_rec_sys

# v3
az ml data create --name hm-processed-data --version 3 \
  --type uri_folder \
  --path azureml://datastores/workspaceblobstore/paths/processed/v3/ \
  --resource-group resource_jhu_rec_sys --workspace-name JHU_rec_sys
```

---

## Step 6 — Modify train.py to Log Metrics to Azure ML Experiments ⬜

**What it is:** Update `src/train.py` to log metrics to Azure ML's experiment tracking system after each epoch using the `azure-ai-ml` SDK.

**Why it matters:** Right now results are saved to local JSON files in `experiments/`. Azure ML Experiments gives you a dashboard in ML Studio showing every run side-by-side — loss curves, metric comparisons, hyperparameters — without manually managing JSON files. Run v1, v2, v3 and compare them all in one view.

**What gets logged per run:**
- Hyperparameters: learning rate, batch size, epochs, cutoff date, embedding dim
- Per-epoch: train loss, val loss
- Final: Recall@12, NDCG@12, MAP@12
- Data version used (v1 / v2 / v3)

---

## Step 7 — Define Azure ML Pipeline ⬜

**What it is:** Chain the four training scripts (`sample.py → preprocess.py → train.py → evaluate.py`) as a formal Azure ML Pipeline — a tracked, versioned, rerunnable workflow that runs on Azure compute instead of your laptop.

**Why it matters:** Right now you run four scripts manually in sequence on your local machine. An Azure ML Pipeline runs them as one job in the cloud, tracks every step's inputs/outputs, and lets you rerun any step in isolation. This is how production ML teams trigger retraining automatically.

**Pipeline steps:**
1. `sample.py --cutoff {date}` — reads from Blob `raw/`, writes transactions sample
2. `preprocess.py --version {v}` — computes features, writes to Blob `processed/{v}/`
3. `train.py` — reads features from Blob, trains model, logs to Azure ML Experiments
4. `evaluate.py` — evaluates model, logs final Recall@12 / MAP@12

---

## Step 8 — Register Trained Models in Azure ML Model Registry ⬜

**What it is:** After each training run, upload `two_tower.pt` to Blob Storage and register it in Azure ML's Model Registry with a version number and tags linking it to the data version and metrics.

**Why it matters:** Without a model registry, you have no formal way to promote a model from "trained" to "production". The registry lets you tag versions (e.g. `staging`, `production`) and roll back if a new version performs worse.

**Commands:**
```bash
# Upload checkpoint to blob
az storage blob upload \
  --account-name jhurecsys0742245294 \
  --container-name "azureml-blobstore-7157eb24-5268-482d-adb1-10eb89c8a1c2" \
  --name "checkpoints/v1/two_tower.pt" \
  --file "checkpoints/26w/two_tower.pt" \
  --auth-mode key

# Register in model registry
az ml model create --name hm-two-tower --version 1 \
  --path azureml://datastores/workspaceblobstore/paths/checkpoints/v1/two_tower.pt \
  --resource-group resource_jhu_rec_sys --workspace-name JHU_rec_sys
```

---

## Step 9 — Modify serve.py to Pull from Blob Storage at Startup ⬜

**What it is:** Update `src/serve.py` so `RecommenderService.__init__` downloads the model checkpoint and processed feature parquets from Blob Storage at container startup, controlled by environment variables.

**Why it matters:** Once data is pulled from Blob at startup, the Docker image contains only code. Deploying a new model version means changing `MODEL_VERSION=v2` in the Kubernetes Deployment YAML — no image rebuild needed.

**Environment variables the container will read:**
```
AZURE_STORAGE_ACCOUNT=jhurecsys0742245294
AZURE_STORAGE_CONTAINER=azureml-blobstore-7157eb24-...
MODEL_VERSION=v1   # change to v2 or v3 to swap model
```

---

## Step 10 — Rebuild Docker Image (Code Only) ⬜

**What it is:** Rebuild the Docker image after removing the `COPY` lines for the checkpoint and processed data files from the Dockerfile. The image will contain only `src/` code and `requirements-serve.txt`.

**Why it matters:** A code-only image is smaller, faster to push/pull, and doesn't need to be rebuilt when the model retrains — only when the code changes.

---

## Step 11 — Create Azure Container Registry and Push Image ⬜

**What it is:** Azure Container Registry (ACR) is Azure's private Docker image registry. You push your Docker image here so AKS can pull it.

**Why it matters:** AKS cannot pull from your local machine. It needs the image in a registry it can reach. ACR integrates directly with AKS.

**Commands:**
```bash
az acr create \
  --name hmrecommenderacr \
  --resource-group resource_jhu_rec_sys \
  --sku Basic

az acr build \
  --registry hmrecommenderacr \
  --image hm-two-tower:v1 .
```

---

## Step 12 — Create AKS Cluster ⬜

**What it is:** Azure Kubernetes Service (AKS) is a managed Kubernetes cluster — a group of VMs that run your Docker containers and handle load balancing, restarts, and scaling automatically.

**Why it matters:** This is the compute that runs your serving container. Without a cluster, there's nowhere to deploy the image.

**Command:**
```bash
az aks create \
  --name hm-recommender-aks \
  --resource-group resource_jhu_rec_sys \
  --node-count 2 \
  --node-vm-size Standard_B2s \
  --generate-ssh-keys
```

---

## Step 13 — Grant AKS Permission to Pull from ACR ⬜

**What it is:** Links AKS and ACR so Kubernetes can authenticate and pull your Docker image at deployment time.

**Why it matters:** Without this, pods fail with `ImagePullBackOff` — Kubernetes can't authenticate to your private registry.

**Command:**
```bash
az aks update \
  --name hm-recommender-aks \
  --resource-group resource_jhu_rec_sys \
  --attach-acr hmrecommenderacr
```

---

## Step 14 — Write Kubernetes Service YAML ⬜

**What it is:** The Deployment YAML tells Kubernetes what to run. The Service YAML exposes those pods to the internet via a public IP and load balancer.

**Why it matters:** Without a Service, pods are running but unreachable from outside the cluster.

**File to create:** `experiments/k8s_service.yml`

---

## Step 15 — Update Deployment YAML with Real ACR Image Path ⬜

**What it is:** Replace the placeholder `image: <your-registry>.azurecr.io/hm-two-tower:v1` in the Deployment YAML with the real ACR path, and add the Blob Storage environment variables so the container knows which model version to pull.

**Real image path:** `hmrecommenderacr.azurecr.io/hm-two-tower:v1`

---

## Step 16 — kubectl apply and Test Live Endpoint ⬜

**What it is:** Connect `kubectl` to the AKS cluster, apply both YAMLs, and send a real recommendation request to the live public IP.

**Commands:**
```bash
az aks get-credentials \
  --name hm-recommender-aks \
  --resource-group resource_jhu_rec_sys

kubectl apply -f experiments/k8s_deployment_example.yml
kubectl apply -f experiments/k8s_service.yml

kubectl get service hm-two-tower-recommender

curl -X POST http://<EXTERNAL-IP>:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "00000dbacae5abe5e23885899a1fa44253a17956c6d1c3d25f88aa139fdfc657"}'
```

---

## Key Azure Resources

| Resource | Name | Purpose |
|---|---|---|
| Resource Group | `resource_jhu_rec_sys` | Container for all Azure resources |
| ML Workspace | `JHU_rec_sys` | Azure ML hub — experiments, models, pipelines |
| Storage Account | `jhurecsys0742245294` | Blob Storage for raw data, features, checkpoints |
| Blob Container | `azureml-blobstore-7157eb24-...` | Default datastore, versioned folder layout |
| Container Registry | `hmrecommenderacr` | Stores Docker image |
| AKS Cluster | `hm-recommender-aks` | Runs serving container, exposes public endpoint |
