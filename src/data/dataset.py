"""PyTorch Dataset over positive (purchase) interactions.

Each item is a (customer_features, article_features) pair for one
transaction. Negatives are not drawn here — the training loop uses the
other items in a batch as in-batch negatives (standard two-tower
retrieval training), so this dataset only needs to emit positives.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocess import ARTICLE_CAT_COLS, CUSTOMER_CAT_COLS

CUSTOMER_NUMERIC_COLS = ["age_bucket", "fn_flag", "active_flag", "postal_code_bucket"]

CUSTOMER_FEATURE_COLS = CUSTOMER_NUMERIC_COLS + [f"{c}_idx" for c in CUSTOMER_CAT_COLS]
ARTICLE_FEATURE_COLS = [f"{c}_idx" for c in ARTICLE_CAT_COLS]


def _clip_column_to_tensor(series: pd.Series, device=None) -> torch.Tensor:
    arr = np.array(
        [np.array(e, dtype=np.float32) if isinstance(e, (list, np.ndarray)) else np.zeros(512, dtype=np.float32)
         for e in series.tolist()],
        dtype=np.float32,
    )
    t = torch.as_tensor(arr)
    return t.to(device) if device is not None else t


class TwoTowerDataset(Dataset):
    def __init__(
        self,
        transactions: pd.DataFrame,
        customer_features: pd.DataFrame,
        article_features: pd.DataFrame,
    ):
        # Join on integer codes, not string IDs — avoids re-encoding and is faster.
        merged = transactions.merge(customer_features, on="customer_idx", how="inner", suffixes=("", "_cust"))
        merged = merged.merge(article_features, on="article_idx", how="inner", suffixes=("", "_art"))

        self.customer_idx = torch.as_tensor(merged["customer_idx"].to_numpy(), dtype=torch.long)
        self.article_idx = torch.as_tensor(merged["article_idx"].to_numpy(), dtype=torch.long)
        self.customer_feats = {
            col: torch.as_tensor(merged[col].to_numpy(), dtype=torch.long) for col in CUSTOMER_NUMERIC_COLS
        }
        self.customer_feats.update(
            {f"{col}_idx": torch.as_tensor(merged[f"{col}_idx"].to_numpy(), dtype=torch.long) for col in CUSTOMER_CAT_COLS}
        )
        self.article_feats = {
            f"{col}_idx": torch.as_tensor(merged[f"{col}_idx"].to_numpy(), dtype=torch.long) for col in ARTICLE_CAT_COLS
        }
        self.clip_embedding: torch.Tensor | None = None
        if "clip_embedding" in merged.columns:
            self.clip_embedding = _clip_column_to_tensor(merged["clip_embedding"])

    def __len__(self):
        return len(self.customer_idx)

    def __getitem__(self, i):
        customer = {"customer_idx": self.customer_idx[i]}
        customer.update({k: v[i] for k, v in self.customer_feats.items()})
        article = {"article_idx": self.article_idx[i]}
        article.update({k: v[i] for k, v in self.article_feats.items()})
        if self.clip_embedding is not None:
            article["clip_embedding"] = self.clip_embedding[i]
        return customer, article

    def iter_batches(self, batch_size: int, shuffle: bool = True, drop_last: bool = True, device=None):
        """Vectorized batch iterator. Slices precomputed tensors directly instead of
        calling __getitem__ per sample — avoids DataLoader/collate overhead (~3x faster)."""
        n = len(self)
        indices = torch.randperm(n) if shuffle else torch.arange(n)
        last_start = n - batch_size + 1 if drop_last else n
        for start in range(0, last_start, batch_size):
            idx = indices[start : start + batch_size]
            customer = {"customer_idx": self.customer_idx[idx]}
            customer.update({k: v[idx] for k, v in self.customer_feats.items()})
            article = {"article_idx": self.article_idx[idx]}
            article.update({k: v[idx] for k, v in self.article_feats.items()})
            if self.clip_embedding is not None:
                article["clip_embedding"] = self.clip_embedding[idx]
            if device is not None:
                customer = {k: v.to(device, non_blocking=True) for k, v in customer.items()}
                article = {k: v.to(device, non_blocking=True) for k, v in article.items()}
            yield customer, article

    def num_batches(self, batch_size: int, drop_last: bool = True) -> int:
        n = len(self)
        last_start = n - batch_size + 1 if drop_last else n
        return len(range(0, last_start, batch_size))
