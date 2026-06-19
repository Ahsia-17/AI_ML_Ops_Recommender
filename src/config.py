from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"


@dataclass
class Config:
    # Sampling
    sample_weeks: int = 26

    # Preprocessing
    age_bins: tuple = (-1, 18, 25, 35, 45, 55, 65, 200)
    postal_code_buckets: int = 5000  # hash bucket size for high-cardinality postal_code

    # Model
    embedding_dim: int = 64
    tower_hidden_dims: tuple = (128, 64)
    dropout: float = 0.1

    # Training
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-6
    epochs: int = 5
    seed: int = 42

    # Evaluation
    top_k: tuple = (12, 20, 50)

    device: str = "cuda"


CONFIG = Config()
