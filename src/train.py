"""Train the two-tower model on the preprocessed dev sample and
checkpoint the result, alongside the vocab sizes / config needed to
reconstruct the model and a per-epoch loss history for tracking.

When running inside an Azure ML job, metrics are logged to Azure ML
Experiments via the Run context (azureml-core). When running locally,
logging is skipped gracefully — same code works in both contexts.
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
    # Doubles as train and eval pass — optimizer=None signals eval mode.
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, total_examples = 0.0, 0
    batches = dataset.iter_batches(batch_size, shuffle=train_mode, drop_last=True, device=device)
    # torch.no_grad() skips building the computation graph during eval,
    # saving memory and time. torch.enable_grad() is the explicit counterpart
    # needed here because no_grad can be inherited from an outer scope.
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
            # Weighted accumulation so the epoch average is correct even if
            # the last batch is smaller than batch_size (drop_last=False case).
            total_loss += loss.item() * batch_size_actual
            total_examples += batch_size_actual
    return total_loss / total_examples


def save_checkpoint(model, vocab_sizes, path):
    # vocab_sizes and architecture config are saved alongside weights so the model
    # can be reconstructed at inference time without needing the original config file.
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
    parser.add_argument(
        "--data-version", type=str, default=None,
        help="Data version tag (e.g. 'v1', 'v2', 'v3'). Logged to Azure ML Experiments "
             "so every run is traceable back to the exact feature set that produced it. "
             "Also controls which versioned subdirectory under data/processed/ to read from "
             "when --processed-dir is not set explicitly.",
    )
    parser.add_argument(
        "--processed-dir", type=str, default=None,
        help="Override the processed data directory. If omitted, uses data/processed/{data-version}/ "
             "when --data-version is set, or data/processed/ otherwise.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Directory to write the checkpoint and training history. "
             "Overrides the default checkpoints/<run-name>/ path. "
             "Set by Azure ML Pipeline to a mounted output folder in Blob Storage.",
    )
    args = parser.parse_args()

    if args.processed_dir:
        from pathlib import Path
        processed_dir = Path(args.processed_dir)
    elif args.data_version:
        processed_dir = PROCESSED_DIR / args.data_version
    else:
        processed_dir = PROCESSED_DIR

    from pathlib import Path
    run_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (CHECKPOINTS_DIR / args.run_name)
    checkpoint_path = run_dir / "two_tower.pt"
    history_path = run_dir / "training_history.json"

    set_seed(CONFIG.seed)
    device = torch.device(CONFIG.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Reading features from: {processed_dir}")

    customer_features = pd.read_parquet(processed_dir / "customers_features.parquet")
    article_features = pd.read_parquet(processed_dir / "articles_features.parquet")
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    with open(processed_dir / "encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    vocab_sizes = encoders["vocab_sizes"]

    # drop_last=True (handled inside iter_batches) keeps batch size constant
    # since in-batch negatives need a full batch of distinct items.
    train_dataset = TwoTowerDataset(train_df, customer_features, article_features)
    val_dataset = TwoTowerDataset(val_df, customer_features, article_features)

    model = TwoTowerModel(vocab_sizes, CONFIG.embedding_dim, CONFIG.tower_hidden_dims, CONFIG.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)

    # Get Azure ML run context if running inside an Azure ML job.
    # Returns a dummy no-op object when running locally so the same
    # code works in both contexts without any changes.
    try:
        from azureml.core.run import Run
        aml_run = Run.get_context()
        is_aml = hasattr(aml_run, "log")
        if is_aml:
            print("Azure ML run context detected — logging metrics to Azure ML Experiments")
    except ImportError:
        aml_run = None
        is_aml = False

    def log_metric(name, value, step=None):
        if is_aml:
            aml_run.log(name, value)
        # Always print so local runs still show progress
        print(f"  {name}={value:.4f}" + (f" (step {step})" if step else ""))

    # Log hyperparameters once at run start
    if is_aml:
        aml_run.log("epochs", args.epochs)
        aml_run.log("batch_size", args.batch_size)
        aml_run.log("embedding_dim", CONFIG.embedding_dim)
        aml_run.log("lr", CONFIG.lr)
        aml_run.log("weight_decay", CONFIG.weight_decay)
        aml_run.log("dropout", CONFIG.dropout)
        aml_run.log("sample_weeks", CONFIG.sample_weeks)
        aml_run.log("data_version", args.data_version or "default")

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_dataset, args.batch_size, device, optimizer)
        val_loss = run_epoch(model, val_dataset, args.batch_size, device, optimizer=None)
        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        log_metric("train_loss", train_loss, step=epoch)
        log_metric("val_loss", val_loss, step=epoch)

        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            epoch_path = run_dir / f"two_tower_epoch{epoch}.pt"
            save_checkpoint(model, vocab_sizes, epoch_path)
            print(f"  saved {epoch_path}")

    save_checkpoint(model, vocab_sizes, checkpoint_path)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved checkpoint to {checkpoint_path} and history to {history_path}")

    if is_aml:
        aml_run.complete()


if __name__ == "__main__":
    main()
