"""Evaluate the trained two-tower model on held-out test interactions.

Retrieval metrics (Recall@K, NDCG@K) are computed by encoding the full
item catalog once, scoring every test user against the whole catalog
via a dot product, and comparing the top-K retrieved articles against
the user's actual test-period purchases. A most-popular-items baseline
is reported alongside the model so the numbers have a reference point.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import CHECKPOINTS_DIR, CONFIG, PROCESSED_DIR
from src.data.dataset import ARTICLE_FEATURE_COLS, CUSTOMER_FEATURE_COLS, _clip_column_to_tensor
from src.models.two_tower import TwoTowerModel

# Score users in chunks to avoid OOM when the full user×item similarity matrix
# would be too large to materialize at once (e.g. 50k users × 50k items on CPU).
USER_SCORE_CHUNK = 1024


def load_model(device, checkpoint_path) -> TwoTowerModel:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TwoTowerModel(
        ckpt["vocab_sizes"], ckpt["embedding_dim"], ckpt["tower_hidden_dims"], ckpt["dropout"],
        use_clip=ckpt.get("use_clip", False),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def encode_catalog(model, article_features: pd.DataFrame, device) -> tuple[torch.Tensor, np.ndarray]:
    # Encode the full item catalog once upfront — every user is then scored against
    # the same fixed item embeddings via a single matrix multiply per chunk.
    feats = {
        "article_idx": torch.as_tensor(article_features["article_idx"].to_numpy(), dtype=torch.long, device=device),
    }
    feats.update(
        {col: torch.as_tensor(article_features[col].to_numpy(), dtype=torch.long, device=device) for col in ARTICLE_FEATURE_COLS}
    )
    if "clip_embedding" in article_features.columns:
        feats["clip_embedding"] = _clip_column_to_tensor(article_features["clip_embedding"], device=device)
    with torch.no_grad():
        item_emb = model.encode_items(feats)
    return item_emb, article_features["article_idx"].to_numpy()


def encode_users(model, customer_features: pd.DataFrame, device) -> tuple[torch.Tensor, np.ndarray]:
    feats = {
        "customer_idx": torch.as_tensor(customer_features["customer_idx"].to_numpy(), dtype=torch.long, device=device),
    }
    feats.update(
        {col: torch.as_tensor(customer_features[col].to_numpy(), dtype=torch.long, device=device) for col in CUSTOMER_FEATURE_COLS}
    )
    with torch.no_grad():
        user_emb = model.encode_users(feats)
    return user_emb, customer_features["customer_idx"].to_numpy()


def ndcg_at_k(hit_ranks: list, num_relevant: int, k: int) -> float:
    # rank is 0-indexed, so position 1 in the NDCG formula is rank+2 (log2(2)=1 for the top hit).
    dcg = sum(1.0 / np.log2(rank + 2) for rank in hit_ranks if rank < k)
    ideal_hits = min(num_relevant, k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def ap_at_k(hit_ranks: list, num_relevant: int, k: int) -> float:
    """Average precision @ k — this is what Kaggle's MAP@12 leaderboard
    metric for this competition averages over users. hit_ranks must be
    sorted ascending (true here since they come from enumerate())."""
    hits, score = 0, 0.0
    for rank in hit_ranks:
        if rank < k:
            hits += 1
            score += hits / (rank + 1)
    denom = min(num_relevant, k)
    return score / denom if denom > 0 else 0.0


def evaluate_rankings(topk_article_idx: np.ndarray, ground_truth: dict, user_ids: np.ndarray, ks: tuple) -> dict:
    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    aps = {k: [] for k in ks}

    for row, user_id in zip(topk_article_idx, user_ids):
        relevant = ground_truth.get(user_id)
        if not relevant:
            continue
        ranked = row.tolist()
        hit_ranks = [pos for pos, item in enumerate(ranked) if item in relevant]
        for k in ks:
            hits_in_k = sum(1 for r in hit_ranks if r < k)
            recalls[k].append(hits_in_k / len(relevant))
            ndcgs[k].append(ndcg_at_k(hit_ranks, len(relevant), k))
            aps[k].append(ap_at_k(hit_ranks, len(relevant), k))

    return {
        "recall": {k: float(np.mean(v)) if v else 0.0 for k, v in recalls.items()},
        "ndcg": {k: float(np.mean(v)) if v else 0.0 for k, v in ndcgs.items()},
        "map": {k: float(np.mean(v)) if v else 0.0 for k, v in aps.items()},
        "num_users_evaluated": len(next(iter(recalls.values()))) if recalls else 0,
    }


def score_topk(user_emb: torch.Tensor, item_emb: torch.Tensor, catalog_article_idx: np.ndarray, k: int) -> np.ndarray:
    results = []
    for start in range(0, user_emb.size(0), USER_SCORE_CHUNK):
        chunk = user_emb[start : start + USER_SCORE_CHUNK]
        scores = chunk @ item_emb.T
        topk = torch.topk(scores, k=k, dim=-1).indices.cpu().numpy()
        results.append(catalog_article_idx[topk])
    return np.concatenate(results, axis=0)


def popularity_baseline(train_df: pd.DataFrame, k: int) -> list:
    return train_df["article_idx"].value_counts().head(k).index.tolist()


def apply_cold_start_fallback(
    topk_article_idx: np.ndarray, user_ids: np.ndarray, warm_customers: set, fallback_topk_row: np.ndarray
) -> np.ndarray:
    """Replace the model's ranking with the popularity ranking for any
    customer with no purchase history in train — their customer_id
    embedding never received a real gradient update, so the model's
    prediction for them is closer to noise than to a useful ranking."""
    result = topk_article_idx.copy()
    cold_mask = np.array([uid not in warm_customers for uid in user_ids])
    result[cold_mask] = fallback_topk_row
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--run-name", type=str, default=f"{CONFIG.sample_weeks}w",
        help="Matches train.py's --run-name: looks for checkpoints/<run-name>/two_tower.pt by default.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Explicit checkpoint path, overrides --run-name (e.g. checkpoints/26w/two_tower_epoch15.pt).",
    )
    parser.add_argument(
        "--processed-dir", type=str, default=None,
        help="Override processed data directory. Used by Azure ML Pipeline to pass the versioned feature folder.",
    )
    parser.add_argument(
        "--clip-dir", type=str, default=None,
        help="Directory containing articles_clip_embeddings.parquet. "
             "Defaults to --processed-dir. Set by Azure ML Pipeline to the hm-clip-embeddings mount.",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir) if args.processed_dir else PROCESSED_DIR

    try:
        from azureml.core.run import Run
        aml_run = Run.get_context()
        if not hasattr(aml_run, "log"):
            aml_run = None
    except ImportError:
        aml_run = None

    checkpoint_path = args.checkpoint or (CHECKPOINTS_DIR / args.run_name / "two_tower.pt")
    device = torch.device(CONFIG.device if torch.cuda.is_available() else "cpu")
    model = load_model(device, checkpoint_path=checkpoint_path)

    customer_features = pd.read_parquet(processed_dir / "customers_features.parquet")
    article_features = pd.read_parquet(processed_dir / "articles_features.parquet")

    if model.item_tower.clip_projection is not None:
        clip_dir = Path(args.clip_dir) if args.clip_dir else processed_dir
        clip_path = clip_dir / "articles_clip_embeddings.parquet"
        article_features = article_features.drop(columns=["clip_embedding"], errors="ignore")
        clip_df = pd.read_parquet(clip_path)[["article_id", "clip_embedding"]]
        article_features = article_features.merge(clip_df, on="article_id", how="left")
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    eval_df = pd.read_parquet(processed_dir / f"{args.split}.parquet")

    eval_customers = customer_features[customer_features["customer_idx"].isin(eval_df["customer_idx"])]
    ground_truth = eval_df.groupby("customer_idx")["article_idx"].apply(set).to_dict()

    item_emb, catalog_article_idx = encode_catalog(model, article_features, device)
    user_emb, user_ids = encode_users(model, eval_customers, device)

    max_k = max(CONFIG.top_k)
    topk_article_idx = score_topk(user_emb, item_emb, catalog_article_idx, max_k)
    metrics = evaluate_rankings(topk_article_idx, ground_truth, user_ids, CONFIG.top_k)

    print(f"Evaluated {metrics['num_users_evaluated']:,} users on '{args.split}' split")
    for k in CONFIG.top_k:
        print(f"  Recall@{k}: {metrics['recall'][k]:.4f}   NDCG@{k}: {metrics['ndcg'][k]:.4f}   MAP@{k}: {metrics['map'][k]:.4f}")
        if aml_run is not None:
            aml_run.log(f"recall@{k}", metrics["recall"][k])
            aml_run.log(f"ndcg@{k}", metrics["ndcg"][k])
            aml_run.log(f"map@{k}", metrics["map"][k])

    pop_items = popularity_baseline(train_df, max_k)
    pop_items_arr = np.array(pop_items)
    pop_topk = np.tile(pop_items_arr, (len(user_ids), 1))
    pop_metrics = evaluate_rankings(pop_topk, ground_truth, user_ids, CONFIG.top_k)
    print("Most-popular-items baseline:")
    for k in CONFIG.top_k:
        print(f"  Recall@{k}: {pop_metrics['recall'][k]:.4f}   NDCG@{k}: {pop_metrics['ndcg'][k]:.4f}   MAP@{k}: {pop_metrics['map'][k]:.4f}")
        if aml_run is not None:
            aml_run.log(f"baseline_recall@{k}", pop_metrics["recall"][k])
            aml_run.log(f"baseline_map@{k}", pop_metrics["map"][k])

    warm_customers = set(train_df["customer_idx"])
    hybrid_topk = apply_cold_start_fallback(topk_article_idx, user_ids, warm_customers, pop_items_arr)
    hybrid_metrics = evaluate_rankings(hybrid_topk, ground_truth, user_ids, CONFIG.top_k)
    print("Model + popularity fallback for cold-start customers:")
    for k in CONFIG.top_k:
        print(f"  Recall@{k}: {hybrid_metrics['recall'][k]:.4f}   NDCG@{k}: {hybrid_metrics['ndcg'][k]:.4f}   MAP@{k}: {hybrid_metrics['map'][k]:.4f}")
        if aml_run is not None:
            aml_run.log(f"hybrid_recall@{k}", hybrid_metrics["recall"][k])
            aml_run.log(f"hybrid_map@{k}", hybrid_metrics["map"][k])

    if aml_run is not None:
        aml_run.log("num_users_evaluated", metrics["num_users_evaluated"])
        aml_run.complete()


if __name__ == "__main__":
    main()
