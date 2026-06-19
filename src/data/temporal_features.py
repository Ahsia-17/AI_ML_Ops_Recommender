"""Point-in-time-correct behavioral features computed from the raw
transaction log: recency, purchase cadence (time between purchases),
windowed frequency, and running-average price. Works identically for
customers and articles (article-side frequency over time is a
recency-weighted popularity / "trending" signal).

Two entry points share the same feature definitions so training and
serving can never silently drift apart:

  add_point_in_time_features(...)  - one row per historical transaction,
                                      using only that transaction's prior
                                      history. This is what feeds model
                                      training (no leakage from the future).

  snapshot_features(...)           - one row per entity, using all history
                                      up to a single `as_of` cutoff. This is
                                      what a production feature pipeline
                                      would materialize on a schedule (e.g.
                                      a nightly Azure ML batch job writing to
                                      a feature table) for an online scoring
                                      script to look up by id at request time
                                      instead of scanning the transaction log.

Both are plain functions over DataFrames — no file I/O, no global state —
so the same module can be imported by an offline preprocessing script, a
batch feature-materialization job, or an online scoring script.
"""

import numpy as np
import pandas as pd


def _expanding_prior_mean(values: pd.Series, group_keys: pd.Series) -> pd.Series:
    """Mean of `values` over each group's PRIOR rows only (the current
    row's own value is excluded). Assumes rows are already sorted by
    group and time. NaN until a group has at least one prior value."""
    is_valid = values.notna().astype(float)
    cum_sum = values.fillna(0).groupby(group_keys).cumsum()
    cum_count = is_valid.groupby(group_keys).cumsum()
    prior_sum = cum_sum.groupby(group_keys).shift(1)
    prior_count = cum_count.groupby(group_keys).shift(1)
    return prior_sum / prior_count.replace(0, np.nan)


def _rolling_count(df: pd.DataFrame, entity_col: str, time_col: str, window_days: int) -> pd.Series:
    """Count of this entity's events strictly before the current row's
    timestamp, within the trailing `window_days` window. `df` must
    already be sorted by [entity_col, time_col]."""
    indexed = df[[entity_col, time_col]].set_index(time_col)
    indexed["_one"] = 1
    counts = indexed.groupby(entity_col)["_one"].rolling(f"{window_days}D", closed="left").count()
    return counts.reset_index(drop=True).fillna(0).astype("int64")


def add_point_in_time_features(
    transactions: pd.DataFrame,
    entity_col: str,
    time_col: str = "t_dat",
    windows_days: tuple = (7, 30, 90),
    value_col: str | None = "price",
) -> pd.DataFrame:
    """Returns a copy of `transactions` with leakage-safe behavioral
    features appended for `entity_col` (e.g. "customer_idx" or
    "article_idx"). Every feature at row i uses only events strictly
    before row i's timestamp — safe to use as model input for predicting
    row i's outcome.

    Adds:
      {entity_col}_days_since_prev   - recency: gap since this entity's last event
      {entity_col}_avg_days_between  - cadence: mean gap over prior events
      {entity_col}_prior_count       - lifetime frequency so far
      {entity_col}_count_last_{w}d   - windowed frequency, for each w in windows_days
      {entity_col}_avg_{value_col}_prior - running mean of value_col (e.g. price), if given
    """
    df = transactions.sort_values([entity_col, time_col]).reset_index(drop=True)
    grouped_time = df.groupby(entity_col, sort=True)[time_col]

    gap_days = (df[time_col] - grouped_time.shift(1)).dt.total_seconds() / 86400
    df[f"{entity_col}_days_since_prev"] = gap_days
    df[f"{entity_col}_avg_days_between"] = _expanding_prior_mean(gap_days, df[entity_col])
    df[f"{entity_col}_prior_count"] = df.groupby(entity_col, sort=True).cumcount()

    for w in windows_days:
        df[f"{entity_col}_count_last_{w}d"] = _rolling_count(df, entity_col, time_col, w)

    if value_col is not None:
        df[f"{entity_col}_avg_{value_col}_prior"] = _expanding_prior_mean(df[value_col], df[entity_col])

    return df


def snapshot_features(
    transactions: pd.DataFrame,
    entity_col: str,
    as_of: pd.Timestamp,
    time_col: str = "t_dat",
    windows_days: tuple = (7, 30, 90),
    value_col: str | None = "price",
) -> pd.DataFrame:
    """One row per entity: the same features as add_point_in_time_features,
    evaluated once as of `as_of` using all history strictly before it.
    Intended for materializing a feature lookup table for online serving.
    """
    history = transactions[transactions[time_col] < as_of]
    if history.empty:
        return pd.DataFrame(columns=[entity_col])

    history = history.sort_values([entity_col, time_col])
    g = history.groupby(entity_col, sort=True)

    out = g[time_col].max().to_frame("_last_event")
    out[f"{entity_col}_days_since_prev"] = (as_of - out["_last_event"]).dt.total_seconds() / 86400
    out[f"{entity_col}_prior_count"] = g.size()

    gap_days = (history[time_col] - g[time_col].shift(1)).dt.total_seconds() / 86400
    out[f"{entity_col}_avg_days_between"] = gap_days.groupby(history[entity_col]).mean().reindex(out.index)

    for w in windows_days:
        cutoff = as_of - pd.Timedelta(days=w)
        windowed = history[history[time_col] >= cutoff]
        out[f"{entity_col}_count_last_{w}d"] = windowed.groupby(entity_col).size().reindex(out.index, fill_value=0)

    if value_col is not None:
        out[f"{entity_col}_avg_{value_col}_prior"] = g[value_col].mean()

    return out.drop(columns="_last_event").reset_index()


def fit_quantile_buckets(values: pd.Series, n_bins: int) -> np.ndarray:
    """Learn bin edges from training data only — these must be persisted
    (alongside the other encoders) and reused as-is at serving time, the
    same way preprocess.py persists categorical vocabularies. Recomputing
    quantiles on live serving data would silently change the feature
    definition out from under the trained model."""
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = values.dropna().quantile(quantiles).to_numpy()
    return np.unique(edges)


def apply_quantile_buckets(values: pd.Series, edges: np.ndarray) -> pd.Series:
    """Bucket `values` using previously-fit edges. Missing values (e.g. a
    customer with no prior purchases, so no recency/cadence yet) map to
    bucket 0, matching the missing/unknown convention used everywhere
    else in preprocess.py."""
    bucketed = pd.cut(values, bins=edges, labels=False, include_lowest=True)
    return bucketed.fillna(-1).astype("int64") + 1


def materialize_snapshot_buckets(
    history: pd.DataFrame,
    entity_col: str,
    as_of: pd.Timestamp,
    feature_cols: list,
    bucket_edges: dict,
    windows_days: tuple = (7, 30, 90),
    value_col: str | None = "price",
) -> pd.DataFrame:
    """The one function an online scoring script (or a nightly batch
    materialization job) calls in production: given recent transaction
    history and "now" (`as_of`), produce the exact same bucketed features
    the model was trained on, using the bucket edges fit during training.

    This is the single reusable piece that makes the feature definitions
    portable to Azure — a batch job (Azure ML pipeline step) can call this
    on a schedule to refresh a feature table, and an online endpoint's
    scoring script can call it directly on a smaller recent-history slice
    for a single incoming customer/article. Same function, same bucket
    edges, either context — that's what prevents train/serve skew.
    """
    snap = snapshot_features(history, entity_col, as_of, windows_days=windows_days, value_col=value_col)
    for col in feature_cols:
        snap[f"{col}_bucket"] = apply_quantile_buckets(snap[col], bucket_edges[col])
    return snap


def attach_temporal_snapshot(
    entities: pd.DataFrame,
    history: pd.DataFrame,
    entity_col: str,
    as_of: pd.Timestamp,
    feature_cols: list,
    bucket_edges: dict,
    windows_days: tuple = (7, 30, 90),
    value_col: str | None = "price",
) -> pd.DataFrame:
    """Left-join bucketed snapshot features onto `entities` (the
    customers or articles being scored right now). Entities with no
    history at all (genuinely cold) get bucket 0 — the same
    missing-value convention used everywhere else — instead of being
    dropped. This is the exact call an online scoring script makes for
    one incoming request, and what a batch evaluation script makes for
    many at once; only `entities` and `history` differ between the two.
    """
    snap = materialize_snapshot_buckets(history, entity_col, as_of, feature_cols, bucket_edges, windows_days, value_col)
    bucket_cols = [f"{c}_bucket" for c in feature_cols]
    merged = entities.merge(snap[[entity_col] + bucket_cols], on=entity_col, how="left")
    for col in bucket_cols:
        merged[col] = merged[col].fillna(0).astype("int64")
    return merged
