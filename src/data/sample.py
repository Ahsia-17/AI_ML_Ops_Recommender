"""Carve a recent, chronologically-bounded slice out of the full
transactions_train.csv so the dev loop doesn't have to pay the cost of
a 31M-row / 3.3GB file on every iteration.

The full file is read in chunks (it does not fit comfortably in memory
alongside a training process) and only rows within the trailing
``sample_weeks`` window are kept.
"""

import argparse

import pandas as pd

from src.config import CONFIG, PROCESSED_DIR, RAW_DIR

TRANSACTIONS_PATH = RAW_DIR / "transactions_train.csv"
SAMPLE_PATH = PROCESSED_DIR / "transactions_sample.parquet"

CHUNK_SIZE = 1_000_000
DTYPES = {
    "customer_id": "string",
    "article_id": "string",
    "price": "float32",
    "sales_channel_id": "int8",
}


def find_max_date(path) -> pd.Timestamp:
    max_date = None
    for chunk in pd.read_csv(path, usecols=["t_dat"], parse_dates=["t_dat"], chunksize=CHUNK_SIZE):
        chunk_max = chunk["t_dat"].max()
        if max_date is None or chunk_max > max_date:
            max_date = chunk_max
    return max_date


def build_sample(weeks: int) -> pd.DataFrame:
    max_date = find_max_date(TRANSACTIONS_PATH)
    cutoff = max_date - pd.Timedelta(weeks=weeks)
    print(f"Full data range ends {max_date.date()}; keeping rows on/after {cutoff.date()}")

    keep_chunks = []
    for chunk in pd.read_csv(
        TRANSACTIONS_PATH,
        parse_dates=["t_dat"],
        dtype=DTYPES,
        chunksize=CHUNK_SIZE,
    ):
        filtered = chunk[chunk["t_dat"] >= cutoff]
        if not filtered.empty:
            keep_chunks.append(filtered)

    sample = pd.concat(keep_chunks, ignore_index=True)
    print(f"Sample: {len(sample):,} rows, {sample['customer_id'].nunique():,} customers, "
          f"{sample['article_id'].nunique():,} articles")
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weeks", type=int, default=CONFIG.sample_weeks)
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sample = build_sample(args.weeks)
    sample.to_parquet(SAMPLE_PATH, index=False)
    print(f"Wrote {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
