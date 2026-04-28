"""
Download and prepare the SQID (Shopping Queries Image Dataset) from HuggingFace.

Source: crossingminds/shopping-queries-image-dataset
  - 164,900 products with CLIP image + text embeddings (768-dim, float32)
  - 8,956 queries with CLIP text embeddings (768-dim, float32)

Embeddings are NOT pre-normalized. We save both raw and L2-normalized versions.
Cosine similarity = inner product on normalized vectors.
"""

import os
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "crossingminds/shopping-queries-image-dataset"
DATA_DIR = os.environ.get("SQID_DATA_DIR", "./data/sqid")


def _download_parquet(filename: str) -> pd.DataFrame:
    print(f"Fetching {filename} ...")
    path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")
    return pd.read_parquet(path)


def _stack_embeddings(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.stack(df[col].apply(np.array).values).astype("float32")


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-8, None)


def download_and_prepare(data_dir: str = DATA_DIR) -> dict[str, str]:
    """
    Download SQID and save embeddings as .npy files.

    Returns a dict with keys:
      product_image     — (164900, 768) float32, L2-normalized
      product_text      — (164900, 768) float32, L2-normalized
      product_ids       — (164900,) str array of product IDs
      query_text        — (8956, 768) float32, L2-normalized
      query_ids         — (8956,) int64 array of query IDs
    """
    os.makedirs(data_dir, exist_ok=True)

    out = {
        "product_image": os.path.join(data_dir, "product_image_embeddings.npy"),
        "product_text":  os.path.join(data_dir, "product_text_embeddings.npy"),
        "product_ids":   os.path.join(data_dir, "product_ids.npy"),
        "query_text":    os.path.join(data_dir, "query_text_embeddings.npy"),
        "query_ids":     os.path.join(data_dir, "query_ids.npy"),
    }

    if all(os.path.exists(p) for p in out.values()):
        print("All files already exist, skipping download.")
        return out

    # --- Products ---
    prod_df = _download_parquet("data/product_features.parquet")
    print(f"  products total: {prod_df.shape[0]:,} rows")

    # 8,358 products have no image — drop them for cross-modal retrieval
    has_image = prod_df["clip_image_features"].notna()
    print(f"  products with image embeddings: {has_image.sum():,} "
          f"(dropped {(~has_image).sum():,} with null image)")
    prod_df = prod_df[has_image].reset_index(drop=True)

    prod_img = _normalize(_stack_embeddings(prod_df, "clip_image_features"))
    prod_txt = _normalize(_stack_embeddings(prod_df, "clip_text_features"))
    prod_ids = prod_df["product_id"].values

    np.save(out["product_image"], prod_img)
    np.save(out["product_text"],  prod_txt)
    np.save(out["product_ids"],   prod_ids)
    print(f"  saved product image embeddings: {prod_img.shape}")
    print(f"  saved product text  embeddings: {prod_txt.shape}")

    # --- Queries ---
    qry_df = _download_parquet("data/query_features.parquet")
    print(f"  queries: {qry_df.shape[0]:,} rows")

    qry_txt = _normalize(_stack_embeddings(qry_df, "clip_text_features"))
    qry_ids = qry_df["query_id"].values

    np.save(out["query_text"], qry_txt)
    np.save(out["query_ids"],  qry_ids)
    print(f"  saved query text embeddings: {qry_txt.shape}")

    # --- Sanity checks ---
    assert prod_img.shape == (156542, 768), f"Unexpected shape: {prod_img.shape}"
    assert qry_txt.shape[1] == 768
    assert np.allclose(np.linalg.norm(prod_img[:5], axis=1), 1.0, atol=1e-5), \
        "Product image embeddings not normalized"
    assert np.allclose(np.linalg.norm(qry_txt[:5], axis=1), 1.0, atol=1e-5), \
        "Query embeddings not normalized"
    print("  sanity checks passed.")

    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()
    paths = download_and_prepare(args.data_dir)
    print("\nReady:")
    for k, v in paths.items():
        arr = np.load(v, allow_pickle=True)
        print(f"  {k:20s} {str(arr.shape):20s} {arr.dtype}")
