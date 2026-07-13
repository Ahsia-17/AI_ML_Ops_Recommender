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


def find_max_date(path, hard_cutoff: pd.Timestamp = None) -> pd.Timestamp:
    max_date = None
    for chunk in pd.read_csv(path, usecols=["t_dat"], parse_dates=["t_dat"], chunksize=CHUNK_SIZE):
        chunk_max = chunk["t_dat"].max()
        if max_date is None or chunk_max > max_date:
            max_date = chunk_max
    # If a hard cutoff is provided, cap the max date there — simulates
    # "we only have data up to this point in time" for versioned pipeline runs.
    if hard_cutoff is not None and max_date > hard_cutoff:
        max_date = hard_cutoff
    return max_date


def build_sample(weeks: int, cutoff: str = None) -> pd.DataFrame:
    hard_cutoff = pd.Timestamp(cutoff) if cutoff else None
    max_date = find_max_date(TRANSACTIONS_PATH, hard_cutoff)
    window_start = max_date - pd.Timedelta(weeks=weeks)
    print(f"Snapshot cutoff: {max_date.date()}; keeping rows from {window_start.date()} to {max_date.date()}")

    keep_chunks = []
    for chunk in pd.read_csv(
        TRANSACTIONS_PATH,
        parse_dates=["t_dat"],
        dtype=DTYPES,
        chunksize=CHUNK_SIZE,
    ):
        filtered = chunk[(chunk["t_dat"] >= window_start) & (chunk["t_dat"] <= max_date)]
        if not filtered.empty:
            keep_chunks.append(filtered)

    sample = pd.concat(keep_chunks, ignore_index=True)
    print(f"Sample: {len(sample):,} rows, {sample['customer_id'].nunique():,} customers, "
          f"{sample['article_id'].nunique():,} articles")
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weeks", type=int, default=CONFIG.sample_weeks)
    parser.add_argument(
        "--cutoff", type=str, default=None,
        help="Hard cutoff date (YYYY-MM-DD). Keep all rows ON OR BEFORE this date, "
             "then apply the trailing --weeks window from there. "
             "Simulates a point-in-time data snapshot for versioned pipeline runs "
             "(e.g. --cutoff 2020-06-30 for v1, 2020-07-31 for v2, 2020-08-31 for v3). "
             "If omitted, uses the actual max date in the file."
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sample = build_sample(args.weeks, cutoff=args.cutoff)
    sample.to_parquet(SAMPLE_PATH, index=False)
    print(f"Wrote {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
