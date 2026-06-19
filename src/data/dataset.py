"""PyTorch Dataset over positive (purchase) interactions.

Each item is a (customer_features, article_features) pair for one
transaction. Negatives are not drawn here — the training loop uses the
other items in a batch as in-batch negatives (standard two-tower
retrieval training), so this dataset only needs to emit positives.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocess import ARTICLE_CAT_COLS, CUSTOMER_CAT_COLS

CUSTOMER_NUMERIC_COLS = ["age_bucket", "fn_flag", "active_flag", "postal_code_bucket"]

CUSTOMER_FEATURE_COLS = CUSTOMER_NUMERIC_COLS + [f"{c}_idx" for c in CUSTOMER_CAT_COLS]
ARTICLE_FEATURE_COLS = [f"{c}_idx" for c in ARTICLE_CAT_COLS]


class TwoTowerDataset(Dataset):
    def __init__(
        self,
        transactions: pd.DataFrame,
        customer_features: pd.DataFrame,
        article_features: pd.DataFrame,
    ):
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

    def __len__(self):
        return len(self.customer_idx)

    def __getitem__(self, i):
        customer = {"customer_idx": self.customer_idx[i]}
        customer.update({k: v[i] for k, v in self.customer_feats.items()})
        article = {"article_idx": self.article_idx[i]}
        article.update({k: v[i] for k, v in self.article_feats.items()})
        return customer, article

    def iter_batches(self, batch_size: int, shuffle: bool = True, drop_last: bool = True, device=None):
        """Vectorized alternative to wrapping this dataset in a
        DataLoader. The whole dataset already lives in memory as
        precomputed tensors (built once in __init__), so going through
        DataLoader's per-row __getitem__ + collate for every single
        sample is pure unnecessary Python-loop overhead — 1024 individual
        calls and a dict-merge per batch, every batch, for an operation
        that's just slicing a handful of tensors by a batch of indices.
        This samples one index tensor per batch and slices every feature
        tensor with it directly — a handful of vectorized gathers instead
        of a Python loop, with no DataLoader/collate/worker overhead.
        """
        n = len(self)
        indices = torch.randperm(n) if shuffle else torch.arange(n)
        last_start = n - batch_size + 1 if drop_last else n
        for start in range(0, last_start, batch_size):
            idx = indices[start : start + batch_size]
            customer = {"customer_idx": self.customer_idx[idx]}
            customer.update({k: v[idx] for k, v in self.customer_feats.items()})
            article = {"article_idx": self.article_idx[idx]}
            article.update({k: v[idx] for k, v in self.article_feats.items()})
            if device is not None:
                customer = {k: v.to(device, non_blocking=True) for k, v in customer.items()}
                article = {k: v.to(device, non_blocking=True) for k, v in article.items()}
            yield customer, article

    def num_batches(self, batch_size: int, drop_last: bool = True) -> int:
        n = len(self)
        last_start = n - batch_size + 1 if drop_last else n
        return len(range(0, last_start, batch_size))
