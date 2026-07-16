"""Turn the raw H&M tables + the windowed transactions sample into
model-ready, integer-encoded feature tables, split chronologically into
train / val / test.

Outputs (all under data/processed/):
  customers_features.parquet   - one row per customer in the sample, integer-coded features
  articles_features.parquet    - one row per article in the sample, integer-coded features
  train.parquet / val.parquet / test.parquet - transaction rows with customer/article integer ids
  encoders.pkl                 - dict of {column: {category: code}} + vocab sizes, needed at
                                  inference time to encode new requests consistently
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG, PROCESSED_DIR, RAW_DIR

CUSTOMER_CAT_COLS = ["club_member_status", "fashion_news_frequency"]
ARTICLE_CAT_COLS = [
    "product_type_name",
    "colour_group_name",
    "department_name",
    "index_name",
    "garment_group_name",
]


def encode_column(series: pd.Series, unknown_token: str = "__UNK__") -> tuple[pd.Series, dict]:
    """Map a categorical column to dense integer codes. Reserves code 0 for
    unknown/missing values so the same mapping can be reused at inference
    time on categories never seen during training."""
    filled = series.fillna(unknown_token).astype(str)
    categories = sorted(filled.unique())
    if unknown_token in categories:
        categories.remove(unknown_token)
    mapping = {unknown_token: 0}
    mapping.update({cat: i + 1 for i, cat in enumerate(categories)})
    codes = filled.map(mapping).astype("int64")
    return codes, mapping


def bucket_age(age: pd.Series, bins) -> pd.Series:
    is_missing = age.isna()
    bucket = pd.cut(age, bins=bins, labels=False)
    # Shift by 1 so code 0 is always "missing", matching the convention in encode_column.
    bucket = bucket.fillna(-1).astype("int64") + 1
    bucket[is_missing] = 0
    return bucket


def hash_postal_code(postal_code: pd.Series, n_buckets: int) -> pd.Series:
    # Hashing instead of LabelEncoding: postal codes are high-cardinality and new
    # values appear at inference time, so a fixed-size hash bucket is more robust
    # than a closed vocabulary that would OOV on unseen codes.
    return postal_code.apply(lambda x: hash(x) % n_buckets).astype("int64")


def build_customer_features(customer_ids: set, raw_dir=None) -> tuple[pd.DataFrame, dict]:
    _raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
    customers = pd.read_csv(_raw_dir / "customers.csv")
    customers = customers[customers["customer_id"].isin(customer_ids)].reset_index(drop=True)

    encoders = {}
    customer_id_codes, customer_id_map = encode_column(customers["customer_id"])
    customers["customer_idx"] = customer_id_codes
    encoders["customer_id"] = customer_id_map

    for col in CUSTOMER_CAT_COLS:
        codes, mapping = encode_column(customers[col])
        customers[f"{col}_idx"] = codes
        encoders[col] = mapping

    customers["age_bucket"] = bucket_age(customers["age"], CONFIG.age_bins)
    customers["fn_flag"] = customers["FN"].fillna(0).astype("int64")
    customers["active_flag"] = customers["Active"].fillna(0).astype("int64")
    customers["postal_code_bucket"] = hash_postal_code(customers["postal_code"], CONFIG.postal_code_buckets)

    keep_cols = [
        "customer_id",
        "customer_idx",
        "age_bucket",
        "fn_flag",
        "active_flag",
        "postal_code_bucket",
    ] + [f"{col}_idx" for col in CUSTOMER_CAT_COLS]
    return customers[keep_cols], encoders


def build_article_features(article_ids: set, raw_dir=None) -> tuple[pd.DataFrame, dict]:
    _raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
    articles = pd.read_csv(_raw_dir / "articles.csv", dtype={"article_id": str})
    articles = articles[articles["article_id"].isin(article_ids)].reset_index(drop=True)

    encoders = {}
    article_id_codes, article_id_map = encode_column(articles["article_id"])
    articles["article_idx"] = article_id_codes
    encoders["article_id"] = article_id_map

    for col in ARTICLE_CAT_COLS:
        codes, mapping = encode_column(articles[col])
        articles[f"{col}_idx"] = codes
        encoders[col] = mapping

    keep_cols = ["article_id", "article_idx"] + [f"{col}_idx" for col in ARTICLE_CAT_COLS]
    return articles[keep_cols], encoders


def chronological_split(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 1-week test window matches the Kaggle leaderboard eval period (last week of data).
    # Val is the week immediately before test so the model never sees future purchases during training.
    max_date = transactions["t_dat"].max()
    test_start = max_date - pd.Timedelta(weeks=1) + pd.Timedelta(days=1)
    val_start = test_start - pd.Timedelta(weeks=1)

    train = transactions[transactions["t_dat"] < val_start]
    val = transactions[(transactions["t_dat"] >= val_start) & (transactions["t_dat"] < test_start)]
    test = transactions[transactions["t_dat"] >= test_start]
    return train, val, test


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", type=str, default=None,
        help="Feature store version tag (e.g. 'v1', 'v2', 'v3'). "
             "Outputs go to data/processed/{version}/ locally. "
             "If omitted, writes to data/processed/ (legacy flat layout)."
    )
    parser.add_argument(
        "--raw-dir", type=str, default=None,
        help="Directory containing articles.csv and customers.csv. "
             "Defaults to data/raw/. Set by Azure ML Pipeline to the hm-raw-data mount.",
    )
    parser.add_argument(
        "--transactions-path", type=str, default=None,
        help="Path to transactions_sample.parquet. "
             "Defaults to data/processed/transactions_sample.parquet. "
             "Set by Azure ML Pipeline to the sample step output.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for all processed files. "
             "Overrides --version. Set by Azure ML Pipeline to the step output folder.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else RAW_DIR
    transactions_path = (
        Path(args.transactions_path) if args.transactions_path
        else PROCESSED_DIR / "transactions_sample.parquet"
    )
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.version:
        out_dir = PROCESSED_DIR / args.version
    else:
        out_dir = PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    transactions = pd.read_parquet(transactions_path)

    customer_features, customer_encoders = build_customer_features(set(transactions["customer_id"]), raw_dir=raw_dir)
    article_features, article_encoders = build_article_features(set(transactions["article_id"]), raw_dir=raw_dir)

    transactions = transactions.merge(
        customer_features[["customer_id", "customer_idx"]], on="customer_id", how="inner"
    )
    transactions = transactions.merge(
        article_features[["article_id", "article_idx"]], on="article_id", how="inner"
    )

    train, val, test = chronological_split(transactions)
    # .copy() materializes each split as an independent DataFrame so that
    # downstream writes (e.g. adding columns) don't trigger SettingWithCopyWarning.
    train, val, test = train.copy(), val.copy(), test.copy()
    print(f"train={len(train):,} val={len(val):,} test={len(test):,}")
    assert train["t_dat"].max() < val["t_dat"].min(), "train/val leakage"
    assert val["t_dat"].max() < test["t_dat"].min(), "val/test leakage"

    customer_features.to_parquet(out_dir / "customers_features.parquet", index=False)
    article_features.to_parquet(out_dir / "articles_features.parquet", index=False)
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)

    encoders = {
        "customer": customer_encoders,
        "article": article_encoders,
        "vocab_sizes": {
            # +1 on encoder length accounts for code 0 (the UNK/missing sentinel),
            # which is not stored as a key in the mapping dict but does occupy an
            # embedding row — so the nn.Embedding vocab size must include it.
            "customer_id": len(customer_encoders["customer_id"]) + 1,
            "article_id": len(article_encoders["article_id"]) + 1,
            "age_bucket": len(CONFIG.age_bins),  # one bucket per bin interval
            "postal_code_bucket": CONFIG.postal_code_buckets,
            **{col: len(customer_encoders[col]) + 1 for col in CUSTOMER_CAT_COLS},
            **{col: len(article_encoders[col]) + 1 for col in ARTICLE_CAT_COLS},
        },
    }
    with open(out_dir / "encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    print(f"customers={len(customer_features):,} articles={len(article_features):,}")
    print(f"Wrote processed tables + encoders to {out_dir}")


if __name__ == "__main__":
    main()
