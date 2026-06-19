"""Plot Recall@12 / MAP@12 vs. epoch for every saved run in this folder.

Each run is a small JSON file (see e.g. 26week_30epoch_no_temporal.json)
with a "eval_metrics_model_plus_cold_start_fallback.by_epoch" list. Run
this after adding a new experiment file to regenerate the comparison
plots.

Usage: python experiments/plot_results.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

EXPERIMENTS_DIR = Path(__file__).resolve().parent


def load_runs():
    runs = {}
    for path in sorted(EXPERIMENTS_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        by_epoch = data.get("eval_metrics_model_plus_cold_start_fallback", {}).get("by_epoch")
        if by_epoch:
            runs[path.stem] = by_epoch
    return runs


def plot_metric(runs: dict, metric: str, ax):
    for name, by_epoch in runs.items():
        epochs = [e["epoch"] for e in by_epoch]
        values = [e[metric] for e in by_epoch]
        ax.plot(epochs, values, marker="o", label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    runs = load_runs()
    if not runs:
        print("No experiment JSON files with eval metrics found in", EXPERIMENTS_DIR)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_metric(runs, "recall_at_12", axes[0])
    plot_metric(runs, "map_at_12", axes[1])
    fig.tight_layout()

    out_path = EXPERIMENTS_DIR / "comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
