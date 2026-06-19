"""Train the two-tower model on the preprocessed dev sample and
checkpoint the result, alongside the vocab sizes / config needed to
reconstruct the model and a per-epoch loss history for tracking.
"""

import argparse
import json
import pickle
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import CHECKPOINTS_DIR, CONFIG, PROCESSED_DIR
from src.data.dataset import TwoTowerDataset
from src.models.two_tower import TwoTowerModel, in_batch_sampled_softmax_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, dataset: TwoTowerDataset, batch_size, device, optimizer=None) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, total_examples = 0.0, 0
    batches = dataset.iter_batches(batch_size, shuffle=train_mode, drop_last=True, device=device)
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for customer_batch, article_batch in tqdm(batches, total=dataset.num_batches(batch_size), leave=False):
            user_emb, item_emb = model(customer_batch, article_batch)
            loss = in_batch_sampled_softmax_loss(user_emb, item_emb)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size_actual = user_emb.size(0)
            total_loss += loss.item() * batch_size_actual
            total_examples += batch_size_actual
    return total_loss / total_examples


def save_checkpoint(model, vocab_sizes, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_sizes": vocab_sizes,
            "embedding_dim": CONFIG.embedding_dim,
            "tower_hidden_dims": CONFIG.tower_hidden_dims,
            "dropout": CONFIG.dropout,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    parser.add_argument(
        "--checkpoint-every", type=int, default=0,
        help="Also save a numbered checkpoint every N epochs (e.g. two_tower_epoch10.pt), for comparing how metrics evolve with more training without retraining from scratch each time. 0 disables.",
    )
    parser.add_argument(
        "--run-name", type=str, default=f"{CONFIG.sample_weeks}w",
        help="Subdirectory under checkpoints/ for this run's artifacts. Defaults to the sample window "
        "(e.g. '52w'), but different runs at the SAME window (e.g. with/without a feature change) will "
        "still clobber each other unless you pass an explicit, distinct --run-name.",
    )
    args = parser.parse_args()

    run_dir = CHECKPOINTS_DIR / args.run_name
    checkpoint_path = run_dir / "two_tower.pt"
    history_path = run_dir / "training_history.json"

    set_seed(CONFIG.seed)
    device = torch.device(CONFIG.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    customer_features = pd.read_parquet(PROCESSED_DIR / "customers_features.parquet")
    article_features = pd.read_parquet(PROCESSED_DIR / "articles_features.parquet")
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    with open(PROCESSED_DIR / "encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    vocab_sizes = encoders["vocab_sizes"]

    # drop_last=True (handled inside iter_batches) keeps batch size constant
    # since in-batch negatives need a full batch of distinct items.
    train_dataset = TwoTowerDataset(train_df, customer_features, article_features)
    val_dataset = TwoTowerDataset(val_df, customer_features, article_features)

    model = TwoTowerModel(vocab_sizes, CONFIG.embedding_dim, CONFIG.tower_hidden_dims, CONFIG.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_dataset, args.batch_size, device, optimizer)
        val_loss = run_epoch(model, val_dataset, args.batch_size, device, optimizer=None)
        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            epoch_path = run_dir / f"two_tower_epoch{epoch}.pt"
            save_checkpoint(model, vocab_sizes, epoch_path)
            print(f"  saved {epoch_path}")

    save_checkpoint(model, vocab_sizes, checkpoint_path)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved checkpoint to {checkpoint_path} and history to {history_path}")


if __name__ == "__main__":
    main()
