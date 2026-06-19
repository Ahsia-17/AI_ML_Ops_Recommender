# AI_ML_Ops_Recommender

This is our project for the AI/ML Ops class, built on the H&M Personalized Fashion Recommendations dataset from Kaggle. We implemented a two-tower neural network that learns to recommend clothing items to customers based on their purchase history and profile info.

## Data

Raw Kaggle data lives in `data/raw/` (gitignored, fetched via `download_data.py`):
- `transactions_train.csv` — ~31.8M purchase rows, 2018-09-20 to 2020-09-22
- `customers.csv` — ~1.4M customers, sparse profile fields (age, club status, etc.)
- `articles.csv` — ~105K articles with categorical metadata (product type, colour, department, ...)

For fast dev iteration the pipeline uses a **chronological windowed sample** (currently the trailing 26 weeks — see `CONFIG.sample_weeks` in `src/config.py`) rather than the full 2-year history. We tested widening this to 52 weeks and it performed *worse*, not better (see Key Findings #6) — 26 weeks is the current best-known setting, not just a placeholder.

## Pipeline

```
src/data/sample.py          chunked read of the full transactions file, keeps the trailing N weeks
src/data/preprocess.py      categorical encoding, chronological train/val/test split
src/data/temporal_features.py   point-in-time + snapshot feature helpers — built and tested, NOT
                                 currently wired into the model (see Key Findings #5)
src/data/dataset.py         PyTorch Dataset + vectorized batch iterator
src/models/two_tower.py     UserTower / ItemTower + in-batch sampled-softmax loss
src/train.py                 training loop, checkpointing (under checkpoints/<run-name>/), loss history
src/evaluate.py              Recall@K / NDCG@K / MAP@12 vs. popularity baseline + cold-start fallback
notebooks/01_eda.ipynb      popularity skew, cold-start prevalence, interaction distribution
experiments/                 saved per-run metrics (JSON) + plot_results.py for comparing runs
```

Run order:
```
python -m src.data.sample
python -m src.data.preprocess
python -m src.train --epochs 30 --checkpoint-every 5 --run-name 26w
python -m src.evaluate --split test --run-name 26w
```

`--run-name` (default: `<sample_weeks>w`, e.g. `26w`) controls which subfolder under `checkpoints/` a run reads/writes — this exists specifically so that re-running with a different config (more epochs, a different window, a feature change) doesn't silently overwrite a previous run's artifacts. `evaluate.py`'s default mirrors `train.py`'s, so the two stay in sync without needing to pass `--checkpoint` by hand (though you still can, to point at e.g. a specific epoch checkpoint).

Config (sample window, model dims, training hyperparameters) is centralized in `src/config.py` — no hardcoded paths or magic numbers scattered through the pipeline.

## Model architecture

Two-tower retrieval: a `UserTower` and an `ItemTower`, each independently embedding their categorical features (`nn.Embedding` per feature, 64-dim), concatenating, and passing through a small MLP (`→128→64→64`) to a single L2-normalized embedding. Scoring is the dot product between user and item embeddings — this decomposability is what makes retrieval cheap at serving time (precompute all item embeddings once, then nearest-neighbor search against a user embedding).

**Features (current, active):**
| Tower | Inputs |
|---|---|
| User | customer_id, age bucket, FN/Active flags, postal-code hash, club status, fashion-news frequency |
| Item | article_id, product type, colour, department, index, garment group |

All static — no temporal/behavioral features are active right now (see Key Findings #5 for why).

**Training:** in-batch sampled softmax — for a batch of B positive `(user, item)` pairs, score every user against every item in the batch (B×B similarity matrix), true pairs on the diagonal, cross-entropy per row. No negative examples are stored; they're free byproducts of other positives in the same batch (the same recipe CLIP uses for image/text pairs).

## `temporal_features.py`: built, tested, currently unused

This module computes point-in-time-correct behavioral features (recency, purchase cadence, windowed frequency, recency-weighted popularity) and is kept in the codebase deliberately even though the model doesn't use it right now — the design is sound and validated, just not currently paying off (see Key Findings #5):

- `add_point_in_time_features()` — per-transaction features using only that row's *prior* history (leakage-checked).
- `snapshot_features()` / `materialize_snapshot_buckets()` — the same feature definitions evaluated once as of a single cutoff — what a production system would call at serving time or in a scheduled batch-materialization job.
- `attach_temporal_snapshot()` — left-joins a snapshot onto entities being scored, with bucket 0 for anyone with no history.
- Verified empirically: `snapshot_features()` evaluated at a timestamp reproduces `add_point_in_time_features()`'s value for a real transaction at that exact timestamp — i.e., train-time and serve-time feature definitions provably match.

## Evaluation methodology

`src/evaluate.py` encodes the entire item catalog once, then scores every eval-split user against it via dot product (brute-force — fine at our ~51K-article scale; would move to an ANN index like FAISS or Azure AI Search at larger scale). Reports Recall@K, NDCG@K, and MAP@12 (matching Kaggle's actual competition metric), against:
- A popularity baseline (most-purchased articles in train)
- The model alone
- **Model + cold-start fallback**: customers with no purchase history in train get the popularity ranking instead of a prediction from an untrained `customer_id` embedding

`experiments/` holds saved per-epoch results (JSON) for completed runs, and `experiments/plot_results.py` regenerates a comparison chart (`experiments/comparison.png`) from whatever's in that folder — useful since checkpoints/logs themselves get overwritten by later runs, but the JSON snapshots persist.

## Key findings

1. **Cold start was the dominant failure mode.** 34% of test customers had zero train-period purchase history, so their `customer_id` embedding never received a gradient update. Measured directly: Recall@12 for cold customers (0.0007) was 11x worse than warm customers (0.0079). Fixed with a popularity fallback for any customer with no train history.
2. **Most of that cold start was a sampling artifact, not genuine.** Checked against the full 2-year history: 75% of "cold" customers actually had purchase history outside the 12-week sample window originally used. Widening the window to 26 weeks converted most of them to warm.
3. **Combined fix beat the popularity baseline:** model + cold-start fallback reached Recall@12 0.0073 / MAP@12 0.0025 vs. the popularity baseline's 0.0070 / 0.0020 at 10 epochs — confirming the diagnosis, not just a hopeful guess. With more training (30 epochs), this climbs further to Recall@12 0.0089 / MAP@12 0.0031, the best result in the project so far, with no sign of plateauing — see `experiments/26week_30epoch_no_temporal.json`.
4. **Benchmarked against the real Kaggle leaderboard** (verified via the actual silver-medal solution writeup, not assumed): top solutions blend five candidate sources (repurchase, co-purchase, price similarity, segment popularity, and two-tower retrieval — the same architecture used here) feeding a LightGBM/CatBoost reranker. A bare retrieval model was never going to match the leaderboard's MAP@12 (~0.0379) — we have one of five ingredients, with no reranking stage.
5. **Temporal features (recency, cadence, frequency, trending-popularity) did not improve results, and more training doesn't fix it.** Added 4 features, re-ran the full 30-epoch sweep: the temporal-features version peaked at epoch 5 (Recall@12 0.0078) and never improved again through epoch 30, while the no-temporal-features version climbed the whole way to 0.0089 with no plateau. This rules out "just needed more epochs to warm up" and points at the feature design itself: the recency-bucket feature collapsed to only 3 distinct values (most transactions are same-day multi-item baskets), and/or a real distributional mismatch between training (lots of "early life, no history yet" rows) and evaluation (each customer's single most-mature snapshot). Reverted; results preserved in `experiments/26week_30epoch_with_temporal.json` for comparison.
6. **More historical data isn't simply better for this domain.** Widening the sample window from 26 to 52 weeks was expected to help (the same lever that fixed cold start earlier), but it didn't: the 52-week run plateaus by epoch 20-25 around Recall@12 0.0078, below the 26-week run's 0.0089 at epoch 30. Likely explanation: the earlier 12→26 week win had a verified mechanism (fixing a real cold-start sampling artifact — see #2); 52 weeks has no equivalent story, and the additional 6 months reach into a different fashion season whose patterns may not transfer well to predicting the September 2020 test week. Reverted to 26 weeks; results preserved in `experiments/52week_30epoch_no_temporal.json`.
7. **Training throughput**: profiling showed the original `DataLoader`-based pipeline was CPU-bound on per-row Python overhead, not GPU-bound. Replacing it with a vectorized batch sampler (index a precomputed tensor directly, no `DataLoader`/`__getitem__`/collate) gave a ~3x speedup (67 it/s vs. ~20-25 it/s) — this is what made running the multiple 30-epoch sweeps in findings #5 and #6 cheap enough to actually do.
8. **Checkpoint/log overwrite bug, caught and fixed.** Early runs all wrote to the same fixed `checkpoints/two_tower.pt` / `training_history.json` paths, so each new experiment silently destroyed the previous one's artifacts — we lost the raw files for two early runs this way (numbers were recoverable from the chat transcript and re-saved to `experiments/`, but the lesson stuck). Fixed by adding `--run-name` to `train.py` so different configs write to separate `checkpoints/<run-name>/` subfolders, and keeping `experiments/*.json` as a permanent, never-overwritten record of each run's results independent of the checkpoint files themselves.

## Productionization notes (Azure)

Decisions made with an eventual Azure ML deployment in mind, even though deployment itself is future work:
- All paths are relative/config-driven (`src/config.py`), no hardcoded absolute paths.
- Training/eval are standalone scripts callable as Azure ML jobs, not notebook-only logic — though they currently read/write fixed local paths and would need light rework to accept Azure ML's mounted input/output paths instead.
- Checkpoints are self-describing — they store the vocab sizes and architecture config alongside weights, so a scoring script can reconstruct the model without depending on the training run's state.
- `temporal_features.py`'s functions take plain DataFrames and a cutoff timestamp — no file I/O, no global state — so the same code could run in a batch feature-materialization pipeline step or be called directly from an online scoring script, *if* the temporal features get revisited and fixed (see Key Findings #5).
- The likely shape of the eventual deployment: Azure ML Managed Online Endpoint (handles the container build and REST API), a scoring script that embeds the incoming user and does nearest-neighbor lookup against a precomputed item-embedding index, and a scheduled batch job that periodically refreshes the item embeddings.
- Worth adopting before deployment: MLflow for experiment tracking (Azure ML's native mechanism) instead of the manual `--run-name` + `experiments/*.json` approach — would track each run with a unique ID automatically and works locally too, without needing Azure set up first.

## Possible next steps

- Add repeat-purchase as an explicit signal (consistently one of the strongest individual predictors for this dataset) — note this doesn't fit as a tower *feature* (it's pairwise: depends on both customer and item together), so it would need to be blended in as a separate candidate source at serving time, similar to the cold-start fallback
- Redesign the recency-bucket feature (collapsed to 3 near-useless buckets) and re-test the other three temporal features in isolation, now that "just train longer" and "just add more data" have both been ruled out as fixes on their own
- CLIP-based image embeddings for articles (article images are already available via the Kaggle competition download) — particularly useful for item cold start, since a brand-new article still has a photo even with zero sales history
- A lightweight reranking stage on top of retrieval candidates, mirroring the two-stage pattern used by competitive solutions
