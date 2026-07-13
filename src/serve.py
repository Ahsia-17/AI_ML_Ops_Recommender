"""Core recommendation-serving logic, independent of any web framework.

This is the one piece both deployment paths (a hand-rolled FastAPI app on
Kubernetes/AKS, or an Azure ML Managed Online Endpoint scoring script)
call into — neither path should reimplement the recommendation logic
itself, just wrap this in whatever request-handling shell that platform
needs (raw HTTP routes for AKS, init()/run() for Azure ML).

Loading the model and the full item catalog is expensive (a few hundred
ms to a couple seconds) and only needs to happen ONCE per process, not
per request — that's what RecommenderService.__init__ does. Answering
an actual request (embed one user, dot product against the precomputed
catalog, take top-K) is cheap and is what .recommend()/.recommend_batch()
do, meant to be called many times against the same loaded instance.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import CHECKPOINTS_DIR, CONFIG, PROCESSED_DIR
from src.evaluate import (
    apply_cold_start_fallback,
    encode_catalog,
    encode_users,
    load_model,
    popularity_baseline,
    score_topk,
)

MAX_K = max(CONFIG.top_k)


def _blob_download(container_client, blob_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(container_client.download_blob(blob_path).readall())


def _fetch_artifacts_from_blob(model_version: str, tmp_dir: Path) -> tuple[Path, Path]:
    """Download checkpoint and processed feature files from Azure Blob Storage.

    Reads three env vars set by the Kubernetes Deployment:
      AZURE_STORAGE_CONNECTION_STRING  — full connection string (from a K8s Secret)
      BLOB_CONTAINER                   — blob container name (workspaceblobstore container)
      MODEL_VERSION                    — e.g. 'v1' (selects processed/v1/ and checkpoints/v1/)

    Returns (checkpoint_path, processed_dir) pointing at the downloaded files.
    """
    from azure.storage.blob import BlobServiceClient

    conn_str   = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    container  = os.environ.get("BLOB_CONTAINER", "azureml")
    version    = model_version

    client = BlobServiceClient.from_connection_string(conn_str).get_container_client(container)

    checkpoint_path = tmp_dir / "two_tower.pt"
    _blob_download(client, f"checkpoints/{version}/two_tower.pt", checkpoint_path)

    processed_dir = tmp_dir / "processed"
    for fname in ["articles_features.parquet", "customers_features.parquet",
                  "train.parquet", "encoders.pkl"]:
        _blob_download(client, f"processed/{version}/{fname}", processed_dir / fname)

    return checkpoint_path, processed_dir


class RecommenderService:
    def __init__(self, checkpoint_path=None, run_name: str = None, device: str = None):
        self.device = torch.device(device or (CONFIG.device if torch.cuda.is_available() else "cpu"))

        # If AZURE_STORAGE_CONNECTION_STRING is set, pull artifacts from Blob Storage.
        # Otherwise fall back to local paths (local dev / unit tests).
        if os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
            model_version = os.environ.get("MODEL_VERSION", "v1")
            print(f"Downloading model artifacts from Blob Storage (version={model_version})...")
            tmp_dir = Path(tempfile.mkdtemp(prefix="hm_serve_"))
            checkpoint_path, processed_dir = _fetch_artifacts_from_blob(model_version, tmp_dir)
            print("Download complete.")
        else:
            run_name = run_name or f"{CONFIG.sample_weeks}w"
            checkpoint_path = checkpoint_path or (CHECKPOINTS_DIR / run_name / "two_tower.pt")
            processed_dir = PROCESSED_DIR

        self.model = load_model(self.device, checkpoint_path=checkpoint_path)

        article_features = pd.read_parquet(processed_dir / "articles_features.parquet")
        customer_features = pd.read_parquet(processed_dir / "customers_features.parquet")
        train_df = pd.read_parquet(processed_dir / "train.parquet", columns=["customer_idx", "article_idx"])

        # Encode the whole catalog ONCE — this is the expensive part this
        # class exists to amortize across many requests.
        self.item_emb, self.catalog_article_idx = encode_catalog(self.model, article_features, self.device)
        self.article_id_by_idx = article_features.set_index("article_idx")["article_id"]

        # Customers with no purchase history in train get the popularity
        # fallback instead of a prediction from an untrained embedding —
        # same logic as evaluate.py, just precomputed once here.
        self.warm_customer_idx = set(train_df["customer_idx"])
        self.popularity_article_idx = np.array(popularity_baseline(train_df, MAX_K))

        self.customer_features = customer_features.set_index("customer_id")

    def _popularity_fallback_ids(self, k: int) -> list:
        return self.article_id_by_idx.loc[self.popularity_article_idx[:k]].tolist()

    def recommend_batch(self, customer_ids: list, k: int = 12) -> list:
        """Returns one result dict per input customer_id, in the same
        order (duplicates in the input are handled correctly via
        positional indexing, not value lookup). Unknown customer_ids (no
        profile at all) and cold customer_ids (a profile exists, but no
        purchase history) both get the popularity fallback — the only
        difference is *why*, reported in "reason"."""
        if k > MAX_K:
            raise ValueError(f"k={k} exceeds the precomputed fallback size MAX_K={MAX_K}")

        rows = self.customer_features.reindex(customer_ids)
        known_mask = rows["customer_idx"].notna().to_numpy()
        fallback_ids = self._popularity_fallback_ids(k)

        results = [None] * len(customer_ids)
        for i, is_known in enumerate(known_mask):
            if not is_known:
                results[i] = {
                    "customer_id": customer_ids[i],
                    "recommendations": fallback_ids,
                    "source": "popularity_fallback",
                    "reason": "unknown_customer",
                }

        known_positions = [i for i, is_known in enumerate(known_mask) if is_known]
        if known_positions:
            known_rows = rows.iloc[known_positions]
            user_emb, encoded_customer_idx = encode_users(self.model, known_rows, self.device)
            topk_article_idx = score_topk(user_emb, self.item_emb, self.catalog_article_idx, k)
            topk_article_idx = apply_cold_start_fallback(
                topk_article_idx, encoded_customer_idx, self.warm_customer_idx, self.popularity_article_idx[:k]
            )
            for pos, customer_idx, rec_row in zip(known_positions, encoded_customer_idx, topk_article_idx):
                is_warm = customer_idx in self.warm_customer_idx
                results[pos] = {
                    "customer_id": customer_ids[pos],
                    "recommendations": self.article_id_by_idx.loc[rec_row].tolist(),
                    "source": "model" if is_warm else "popularity_fallback",
                    "reason": None if is_warm else "no_purchase_history",
                }
        return results

    def recommend(self, customer_id: str, k: int = 12) -> dict:
        return self.recommend_batch([customer_id], k=k)[0]
