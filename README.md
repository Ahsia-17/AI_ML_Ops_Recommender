# H&M Two-Tower Recommender

Built for the AI/ML Ops class at Johns Hopkins University on the H&M Personalized Fashion Recommendations dataset from Kaggle. The system learns to recommend clothing items to customers based on purchase history and profile features, using a two-tower neural network trained with in-batch negative sampling.

The pipeline is designed end-to-end with productionization in mind: versioned data in Azure Blob Storage, experiment tracking in Azure ML, a containerized serving layer, and a Kubernetes deployment on AKS.

## Data

Raw Kaggle data lives in `data/raw/` (gitignored, fetched via `download_data.py`):
- `transactions_train.csv` — ~31.8M purchase rows, Sep 2018 to Sep 2020
- `customers.csv` — ~1.4M customers, sparse profile fields (age, club status, postal code, etc.)
- `articles.csv` — ~105K articles with categorical metadata (product type, colour, department, etc.)

Raw data is also stored in **Azure Blob Storage** (`workspaceblobstore/raw/`) and registered as an Azure ML Data Asset (`hm-raw-data:1`) for lineage tracking.

## Versioning Strategy

To simulate a production system with live data arrivals, we treat three date-range slices of the dataset as separate monthly data releases:

| Version | Cutoff date | Simulates |
|---|---|---|
| v1 | 2020-06-30 | First month live |
| v2 | 2020-07-31 | Second month |
| v3 | 2020-08-31 | Third month — most recent |

Each version has its own processed feature folder in Blob Storage and a registered Azure ML Data Asset (`hm-processed-data:1/2/3`). The pipeline is designed to train and evaluate a separate checkpoint per version, demonstrating the full MLOps data lineage chain: raw data → versioned features → model → metrics. Currently only v1 has a trained checkpoint deployed to AKS; v2 and v3 are ready to train via `python azure/pipeline.py --data-version v2`.

## Pipeline

```
src/data/sample.py              chunked read of full transactions file, keeps trailing N weeks
                                accepts --cutoff YYYY-MM-DD to simulate point-in-time data snapshots
src/data/preprocess.py          categorical encoding, chronological train/val/test split
                                accepts --version v1/v2/v3 to write versioned output folders
src/data/temporal_features.py   point-in-time + snapshot feature helpers — built, NOT currently active
src/data/dataset.py             PyTorch Dataset + vectorized batch iterator (3x faster than DataLoader)
src/models/two_tower.py         UserTower / ItemTower + in-batch sampled-softmax loss
src/train.py                    training loop, Azure ML metric logging, checkpointing
src/evaluate.py                 Recall@K / NDCG@K / MAP@12 vs. popularity baseline + cold-start fallback
src/serve.py                    RecommenderService — framework-agnostic serving logic (warm/cold/unknown)
src/api.py                      FastAPI REST API wrapping RecommenderService (AKS deployment path)
experiments/plot_results.py     plots Recall@12 / MAP@12 vs. epoch across saved experiment JSONs
```

**Local run order (one version):**
```bash
python -m src.data.sample --cutoff 2020-06-30 --weeks 26
python -m src.data.preprocess --version v1
python -m src.train --data-version v1 --run-name v1-26w --epochs 30
python -m src.evaluate --split test --run-name v1-26w
```

Config (sample window, model dims, training hyperparameters) is centralized in `src/config.py`.

## Model Architecture

Two-tower retrieval: a `UserTower` and an `ItemTower`, each independently embedding their categorical features (`nn.Embedding` per feature, 64-dim), concatenating, and passing through a small MLP (`→128→64→64`) to a single L2-normalized embedding. Scoring is the dot product of user and item embeddings — this decomposability is what makes retrieval cheap at serving time (precompute all item embeddings once, then nearest-neighbor search against a user embedding).

**Features (current, active):**

| Tower | Inputs |
|---|---|
| User | customer_id, age bucket, FN/Active flags, postal-code hash, club status, fashion-news frequency |
| Item | article_id, product type, colour, department, index, garment group |

**Loss:** in-batch sampled softmax — for a batch of B positive `(user, item)` pairs, score every user against every item in the batch (B×B similarity matrix), true pairs on the diagonal, cross-entropy per row. Negatives are free byproducts of other positives in the same batch (same recipe as CLIP).

## Serving Layer

**AKS / Kubernetes (Path B — non-managed, fully implemented):**
- `src/serve.py` — `RecommenderService` downloads the model checkpoint and feature parquets from Azure Blob Storage at container startup using `AZURE_STORAGE_CONNECTION_STRING` and `MODEL_VERSION` env vars. Precomputes item embeddings once. Handles three customer cases: warm (model prediction), cold (no purchase history → popularity fallback), unknown (not in dataset → popularity fallback)
- `src/api.py` — FastAPI REST API exposing `/health`, `/recommend`, `/recommend/batch` on port 8080
- `Dockerfile` — Code-only CPU serving container. No data or checkpoint baked in — everything is pulled from Blob Storage at startup so the image never needs rebuilding when the model retrains
- `experiments/k8s_deployment.yml` — Kubernetes Deployment manifest (1 replica, 1 CPU / 2GB RAM, Blob Storage credentials injected via K8s Secret)
- `experiments/k8s_service.yml` — Kubernetes Service manifest exposing a public LoadBalancer

**Live endpoint:** `http://20.161.92.233`
```bash
curl http://20.161.92.233/health
curl -X POST http://20.161.92.233/recommend \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "<customer-id>"}'
```

## Azure MLOps Setup

All infrastructure is in Azure, connected to the `JHU_rec_sys` ML workspace:

| Resource | Name | Purpose |
|---|---|---|
| Storage Account | `jhurecsys0742245294` | Blob Storage — raw data, features, checkpoints |
| Data Asset | `hm-raw-data:1` | Versioned pointer to raw CSVs |
| Data Asset | `hm-processed-data:1/2/3` | Versioned pointers to processed feature folders |
| Model | `hm-two-tower:1` | Registered model in Azure ML Model Registry |
| Compute Cluster | `hm-training-cluster` | Auto-scaling VMs for training jobs (min 0 nodes) |
| Environment | `hm-recommender-training:1` | Conda environment for training jobs |
| Container Registry | `hmrecommenderacr` | Stores Docker serving images |
| AKS Cluster | `hm-recommender-aks` | Runs serving container |

Training runs log metrics (train loss, val loss, hyperparameters, data version) to **Azure ML Experiments** via `azureml-core` — visible in Azure ML Studio under Jobs. When run locally, logging degrades gracefully to console output.

**Submitting a training pipeline run:**
```bash
python azure/pipeline.py --data-version v1  # or v2, v3
```

## CI/CD

Two GitHub Actions workflows automate the development loop:

| Workflow | Trigger | Action |
|---|---|---|
| `ci.yml` | Pull request to `main` | Installs deps, runs model smoke test (forward pass with synthetic data) |
| `cd.yml` | Merge to `main` | Builds Docker image, pushes to ACR, deploys to AKS |

**Required GitHub Secrets:** `ACR_USERNAME`, `ACR_PASSWORD`, `KUBE_CONFIG`

See [CICD.md](CICD.md) for the full playbook: secret setup, how to ship a new model feature branch, how to swap a new trained model version on the live endpoint without rebuilding the image.

## `temporal_features.py`: Built, Tested, Currently Unused

Computes point-in-time behavioral features (recency, cadence, windowed frequency, running price average). Kept deliberately — the design is sound but empirically did not improve model performance (see Key Findings #5). Two entry points share identical feature definitions so train and serve can never drift apart:
- `add_point_in_time_features()` — per-transaction, leakage-safe (training path)
- `snapshot_features()` / `materialize_snapshot_buckets()` — single cutoff snapshot (serving path)

## Evaluation Methodology

Encodes the entire item catalog once, scores every eval-split user against it via dot product, and reports Recall@K, NDCG@K, and MAP@12 (matching Kaggle's leaderboard metric) for K ∈ {12, 20, 50}. Three results reported side-by-side:
1. Popularity baseline (most-purchased articles in train, no ML)
2. Model alone
3. **Model + cold-start fallback** — customers with no train history get the popularity ranking

## Key Findings

1. **Cold start was the dominant failure mode.** 34% of test customers had zero train-period history, so their embeddings never received a gradient update. Recall@12 for cold customers (0.0007) was 11× worse than warm customers (0.0079). Fixed with a popularity fallback.
2. **Most cold start was a sampling artifact.** 75% of "cold" customers had purchase history outside the 12-week original window. Widening to 26 weeks converted most of them to warm.
3. **Combined fix beat the popularity baseline.** Model + cold-start fallback: Recall@12 0.0089, MAP@12 0.0031 at 30 epochs. Best result so far, no sign of plateauing.
4. **Kaggle leaderboard gap is expected.** Top solutions blend five candidate sources feeding a LightGBM reranker. We have one of five ingredients, no reranking stage.
5. **Temporal features hurt, not helped.** Added 4 features, ran 30-epoch sweep — temporal version peaked at epoch 5 (Recall@12 0.0078) and flatlined; no-temporal version climbed to 0.0089. Reverted; results in `experiments/26week_30epoch_with_temporal.json`.
6. **More historical data isn't simply better.** 52-week window plateaued at 0.0078 vs. 26-week's 0.0089. Older data covers a different fashion season whose patterns don't transfer well to the September 2020 test week. Reverted; results in `experiments/52week_30epoch_no_temporal.json`.
7. **Training throughput.** Original DataLoader was CPU-bound on per-row Python overhead. Replaced with a vectorized batch sampler — 3× speedup (67 it/s vs. ~22 it/s).
8. **Checkpoint overwrite bug, caught and fixed.** Early runs silently overwrote each other. Fixed with `--run-name` flag routing artifacts to `checkpoints/<run-name>/` subfolders.

## Possible Next Steps

- Re-enable temporal features with a fixed recency bucket design (original collapsed to 3 near-useless values)
- CLIP-based image embeddings for articles — useful for item cold start since new articles have photos but no sales history
- A lightweight reranking stage on top of retrieval candidates
- Automated pipeline trigger on a schedule or data arrival event (currently triggered manually)
