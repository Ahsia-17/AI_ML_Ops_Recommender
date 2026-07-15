"""Streamlit demo app — H&M Two-Tower Recommender.

Picks a random customer from the training data, hits the live API,
and displays recommended products as a grid with images and metadata.

Usage:
    pip install streamlit
    streamlit run app.py
"""

import os
import random
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("API_URL", "http://20.161.92.233/recommend")
PROCESSED_DIR = Path("data/processed")
IMAGES_DIR    = Path("data/raw/images")
RAW_DIR       = Path("data/raw")

st.set_page_config(page_title="H&M Recommender Demo", layout="wide")


@st.cache_data
def load_customers() -> list[str]:
    df = pd.read_parquet(PROCESSED_DIR / "customers_features.parquet")
    return df["customer_id"].tolist()


@st.cache_data
def load_articles() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "articles.csv", dtype={"article_id": str})
    return df.set_index("article_id")


def image_path(article_id: str) -> Path:
    return IMAGES_DIR / article_id[:3] / f"{article_id}.jpg"


def get_recommendations(customer_id: str, k: int = 12) -> dict:
    resp = requests.post(API_URL, json={"customer_id": customer_id, "k": k}, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("H&M Recommender")
st.sidebar.markdown("Two-tower retrieval model with CLIP multi-modal embeddings.")

customers   = load_customers()
articles_df = load_articles()

if "customer_id" not in st.session_state:
    st.session_state.customer_id = random.choice(customers)

if st.sidebar.button("Random Customer", use_container_width=True):
    st.session_state.customer_id = random.choice(customers)

customer_id = st.session_state.customer_id
st.sidebar.markdown(f"**Customer ID**")
st.sidebar.code(customer_id)

k = st.sidebar.slider("Recommendations", min_value=4, max_value=24, value=12, step=4)

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("Picked for you")

with st.spinner("Fetching recommendations..."):
    try:
        result = get_recommendations(customer_id, k=k)
    except Exception as e:
        st.error(f"API error: {e}")
        st.stop()

source = result.get("source", "")
reason = result.get("reason", "")

if source == "model":
    st.success("Model recommendation")
else:
    st.info(f"Popularity fallback — {reason}")

rec_ids = result.get("recommendations", [])

cols = st.columns(4)
for i, article_id in enumerate(rec_ids):
    with cols[i % 4]:
        img_path = image_path(article_id)
        if img_path.exists():
            st.image(Image.open(img_path), use_container_width=True)
        else:
            st.markdown("🖼️ *No image*")

        if article_id in articles_df.index:
            row = articles_df.loc[article_id]
            st.markdown(f"**{row.get('prod_name', article_id)}**")
            st.caption(f"{row.get('product_type_name', '')} · {row.get('colour_group_name', '')}")
        else:
            st.markdown(f"`{article_id}`")
