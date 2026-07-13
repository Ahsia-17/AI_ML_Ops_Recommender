"""Two-tower retrieval model.

Each tower independently embeds its categorical features (an
nn.Embedding per feature), concatenates them, and runs the result
through an MLP to produce a single L2-normalized embedding vector.
Scoring between a user and an item is just the dot product of their
two embeddings — this is what makes the towers usable for retrieval
(precompute all item embeddings once, then do nearest-neighbor search
against a user embedding, no need to re-run the model per candidate).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import ARTICLE_FEATURE_COLS, CUSTOMER_FEATURE_COLS


def _mlp(input_dim: int, hidden_dims: tuple, output_dim: int, dropout: float) -> nn.Sequential:
    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
        prev_dim = h
    # Final projection has no activation — the Tower applies L2 normalization after this.
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class FeatureEmbedder(nn.Module):
    """Holds one nn.Embedding per categorical feature and concatenates
    their lookups into a single vector per example.

    Features whose code 0 means "unknown/missing" (everything produced by
    preprocess.encode_column, plus the missing-flag features) use
    padding_idx=0 so that absence of information contributes a zero
    vector rather than a learned embedding. Hash-bucketed features (e.g.
    postal_code_bucket) have no such "missing" sentinel — bucket 0 is a
    regular value there — so they keep a normal, trainable embedding.
    """

    NON_MISSING_FEATURES = {"postal_code_bucket"}

    def __init__(self, feature_vocab_sizes: dict, embedding_dim: int):
        super().__init__()
        self.feature_names = list(feature_vocab_sizes.keys())
        # nn.ModuleDict (not a plain dict) so PyTorch tracks these embeddings
        # as registered parameters and includes them in model.parameters().
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(
                    vocab_size,
                    embedding_dim,
                    padding_idx=None if name in self.NON_MISSING_FEATURES else 0,
                )
                for name, vocab_size in feature_vocab_sizes.items()
            }
        )
        self.output_dim = embedding_dim * len(self.feature_names)

    def forward(self, features: dict) -> torch.Tensor:
        vecs = [self.embeddings[name](features[name]) for name in self.feature_names]
        return torch.cat(vecs, dim=-1)


class Tower(nn.Module):
    def __init__(self, feature_vocab_sizes: dict, embedding_dim: int, hidden_dims: tuple, dropout: float):
        super().__init__()
        self.embedder = FeatureEmbedder(feature_vocab_sizes, embedding_dim)
        self.mlp = _mlp(self.embedder.output_dim, hidden_dims, embedding_dim, dropout)

    def forward(self, features: dict) -> torch.Tensor:
        x = self.embedder(features)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, vocab_sizes: dict, embedding_dim: int, hidden_dims: tuple, dropout: float):
        super().__init__()

        customer_vocab_sizes = {
            "customer_idx": vocab_sizes["customer_id"],
            "age_bucket": vocab_sizes["age_bucket"],
            "fn_flag": 2,      # binary flag: 0 or 1
            "active_flag": 2,  # binary flag: 0 or 1
            "postal_code_bucket": vocab_sizes["postal_code_bucket"],
            "club_member_status_idx": vocab_sizes["club_member_status"],
            "fashion_news_frequency_idx": vocab_sizes["fashion_news_frequency"],
        }
        article_vocab_sizes = {
            "article_idx": vocab_sizes["article_id"],
            "product_type_name_idx": vocab_sizes["product_type_name"],
            "colour_group_name_idx": vocab_sizes["colour_group_name"],
            "department_name_idx": vocab_sizes["department_name"],
            "index_name_idx": vocab_sizes["index_name"],
            "garment_group_name_idx": vocab_sizes["garment_group_name"],
        }
        # Guard against feature drift: if dataset.py and this file ever get out of sync
        # on which columns exist, this will catch it at model construction time.
        assert set(customer_vocab_sizes) == {"customer_idx", *CUSTOMER_FEATURE_COLS}
        assert set(article_vocab_sizes) == {"article_idx", *ARTICLE_FEATURE_COLS}

        self.user_tower = Tower(customer_vocab_sizes, embedding_dim, hidden_dims, dropout)
        self.item_tower = Tower(article_vocab_sizes, embedding_dim, hidden_dims, dropout)

    def forward(self, customer_features: dict, article_features: dict) -> tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.user_tower(customer_features)
        item_emb = self.item_tower(article_features)
        return user_emb, item_emb

    def encode_users(self, customer_features: dict) -> torch.Tensor:
        return self.user_tower(customer_features)

    def encode_items(self, article_features: dict) -> torch.Tensor:
        return self.item_tower(article_features)


def in_batch_sampled_softmax_loss(user_emb: torch.Tensor, item_emb: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """Cross-entropy over the in-batch similarity matrix: for each user,
    the matching item (same row) is the positive class and every other
    item in the batch is a sampled negative."""
    # @ is Python's matrix multiply operator. Dividing by a small temperature
    # sharpens the distribution — without it the softmax saturates near uniform
    # because L2-normalized dot products stay close to zero.
    logits = user_emb @ item_emb.T / temperature
    # Positive pairs are on the diagonal: user[i] should match item[i].
    # torch.arange gives [0, 1, 2, ..., B-1] as the target class per row.
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)
