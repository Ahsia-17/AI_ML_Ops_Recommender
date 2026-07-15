"""Generate CLIP embeddings for all H&M articles by fusing image + text.

For each article we produce one 512-dim embedding by:
  1. Encoding the product image with CLIP's image encoder
  2. Encoding a text description built from the article metadata with CLIP's text encoder
  3. L2-normalising both, then averaging → a single 512-dim vector

Using both modalities captures complementary signals:
  - Image: colour, silhouette, texture, pattern, visual style
  - Text: fit language ("slim"), material, department, category hierarchy

CLIP was trained to align image and text in the same embedding space, so
averaging them is semantically meaningful — not just concatenating arbitrary vectors.

Output: data/processed/articles_clip_embeddings.parquet
  article_id      str   — original H&M 10-digit article ID string
  clip_embedding  list  — 512-dim float32 vector

The output is NOT versioned because CLIP embeddings depend only on the static
article catalog (images + articles.csv), not on the transaction sample window.
Run this once; preprocess.py picks it up automatically when building any version.

Usage:
    pip install transformers Pillow
    python -m src.data.embed_images
    python -m src.data.embed_images --batch-size 512
    python -m src.data.embed_images --model openai/clip-vit-large-patch14
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from src.config import PROCESSED_DIR, RAW_DIR

IMAGES_DIR = RAW_DIR / "images"
DEFAULT_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_BATCH_SIZE = 256


def _image_path(article_id: str) -> Path:
    # H&M organises images as images/{first_3_digits}/{article_id}.jpg
    return IMAGES_DIR / article_id[:3] / f"{article_id}.jpg"


def _build_text(row: pd.Series) -> str:
    """Construct a natural-language description from structured metadata columns.

    Puts the most semantically useful fields first so they survive CLIP's
    77-token limit even when detail_desc is long.
    """
    structured = ", ".join(
        str(row[col]).strip()
        for col in ["colour_group_name", "product_type_name", "department_name", "garment_group_name"]
        if pd.notna(row.get(col)) and str(row.get(col)).strip()
    )
    desc = row.get("detail_desc")
    if pd.notna(desc) and str(desc).strip():
        return f"{structured}. {str(desc).strip()}"
    return structured


def _encode_batch(
    model,
    processor,
    images: list,
    texts: list[str],
    device: torch.device,
) -> np.ndarray:
    """Encode one batch and return averaged, re-normalised embeddings (B, 512)."""
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,  # CLIP's hard token limit
    ).to(device)

    with torch.no_grad():
        # model(**inputs) returns CLIPOutput with image_embeds + text_embeds
        # already projected to 512-dim — more reliable across transformers versions
        # than get_image_features() / get_text_features() whose return types changed.
        outputs   = model(**inputs)
        img_feat  = outputs.image_embeds
        text_feat = outputs.text_embeds

    # L2-normalise so both modalities sit on the unit sphere before averaging
    img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    # Average — valid because CLIP aligns both spaces during pre-training
    combined = (img_feat + text_feat) / 2.0
    # Re-normalise after averaging so downstream dot products stay in [-1, 1]
    combined = combined / combined.norm(dim=-1, keepdim=True)

    return combined.cpu().float().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace CLIP model ID")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Model  : {args.model}")
    print(f"Batch  : {args.batch_size}")

    print("Loading CLIP model and processor...")
    model     = CLIPModel.from_pretrained(args.model).to(device)
    processor = CLIPProcessor.from_pretrained(args.model)
    model.eval()

    articles = pd.read_csv(RAW_DIR / "articles.csv", dtype={"article_id": str})
    print(f"Articles in catalog: {len(articles):,}")

    out_ids: list[str]   = []
    out_embs: list[list] = []
    skipped = 0

    # Accumulate into batches, flush when full
    batch_ids:    list[str] = []
    batch_images: list      = []
    batch_texts:  list[str] = []

    def flush():
        if not batch_ids:
            return
        embs = _encode_batch(model, processor, batch_images, batch_texts, device)
        out_ids.extend(batch_ids)
        out_embs.extend(embs.tolist())
        batch_ids.clear()
        batch_images.clear()
        batch_texts.clear()

    for _, row in tqdm(articles.iterrows(), total=len(articles), desc="Encoding articles"):
        aid      = row["article_id"]
        img_path = _image_path(aid)

        if not img_path.exists():
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        batch_ids.append(aid)
        batch_images.append(img)
        batch_texts.append(_build_text(row))

        if len(batch_ids) >= args.batch_size:
            flush()

    flush()  # final partial batch

    print(f"\nEncoded : {len(out_ids):,}")
    print(f"Skipped : {skipped:,}  (missing or corrupt images)")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "articles_clip_embeddings.parquet"
    pd.DataFrame({"article_id": out_ids, "clip_embedding": out_embs}).to_parquet(
        out_path, index=False
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved   : {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
