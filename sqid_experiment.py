"""SQID experiment: IVF/PQ parameter sweep for cross-modal product retrieval.

Sweeps nprobe and k at fixed config (nlist=256, M=16, OPQ).
Ground truth computed once with IndexFlatIP and cached.

Results written to indexes/sqid_results.json.

Usage:
  python sqid_experiment.py

Expected (from prior runs):
  nprobe=16  recall@10 ~0.278  recall@100 ~0.421  QPS ~121K
  Recall ceiling ~0.431 at nprobe=64 — PQ quantization bottleneck, not IVF coverage.
"""

import json
import time
from pathlib import Path

import faiss
import numpy as np

from build_index import SQID_CONFIG, build_index, load_index
from evaluate import benchmark_search, cluster_size_stats, opq_needed, recall_at_k
from search import search_batch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR   = Path("data/sqid")
INDEX_DIR  = Path("indexes")
PROD_FILE  = DATA_DIR / "product_image_embeddings.npy"
QUERY_FILE = DATA_DIR / "query_text_embeddings.npy"
INDEX_FILE = INDEX_DIR / "sqid.index"
GT_FILE    = INDEX_DIR / "sqid_gt.npy"

faiss.omp_set_num_threads(10)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-8, None)


def load_data() -> tuple[np.ndarray, np.ndarray]:
    products = l2_normalize(np.load(PROD_FILE))
    queries  = l2_normalize(np.load(QUERY_FILE))
    print(f"products: {products.shape}  queries: {queries.shape}")
    assert np.allclose(np.linalg.norm(products[:10], axis=1), 1.0, atol=1e-5)
    return products, queries


def get_ground_truth(products: np.ndarray, queries: np.ndarray, k: int = 100) -> np.ndarray:
    if GT_FILE.exists():
        gt = np.load(str(GT_FILE))
        print(f"[gt] loaded cached GT  {gt.shape}")
        return gt
    print("[gt] computing exact GT with IndexFlatIP ...")
    t0   = time.perf_counter()
    flat = faiss.IndexFlatIP(products.shape[1])
    flat.add(products)
    _, gt = flat.search(queries, k)
    print(f"[gt] done in {time.perf_counter() - t0:.1f}s")
    np.save(str(GT_FILE), gt)
    return gt


def build_or_load(products: np.ndarray) -> faiss.Index:
    if INDEX_FILE.exists():
        index = load_index(str(INDEX_FILE))
        print(f"[build] loaded existing index: ntotal={index.ntotal:,}")
        return index
    INDEX_DIR.mkdir(exist_ok=True)
    print("[build] OPQ diagnostic ...")
    needed, ratio = opq_needed(products[:10_000], SQID_CONFIG.M)
    print(f"[build] training OPQ16_768,IVF256,PQ16 on {len(products):,} vectors ...")
    t0    = time.perf_counter()
    index = build_index(products, SQID_CONFIG, name="sqid", out_dir=str(INDEX_DIR))
    print(f"[build] total: {time.perf_counter() - t0:.1f}s")
    return index

# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def nprobe_sweep(index: faiss.Index, queries: np.ndarray, gt: np.ndarray) -> list[dict]:
    print("\n=== nprobe sweep  (k=100) ===")
    print(f"{'nprobe':>8}  {'recall@10':>10}  {'recall@100':>11}  {'QPS':>8}")
    rows = []
    for nprobe in [1, 2, 4, 8, 16, 32, 64]:
        indices, _, _ = search_batch(index, queries, k=100, nprobe=nprobe,
                                     batch_size=len(queries), timeout_s=60)
        r10  = recall_at_k(indices, gt, k=10)
        r100 = recall_at_k(indices, gt, k=100)
        bench = benchmark_search(index, queries, k=100, nprobe=nprobe, runs=3)
        rows.append({"nprobe": nprobe, "recall@10": round(r10, 4),
                     "recall@100": round(r100, 4), **bench})
        print(f"  {nprobe:>6}  {r10:>10.3f}  {r100:>11.3f}  {bench['qps_mean']:>8.0f}")
    return rows


def k_sweep(index: faiss.Index, queries: np.ndarray, gt: np.ndarray) -> list[dict]:
    print("\n=== k sweep  (nprobe=16) ===")
    print(f"{'k':>6}  {'recall@k':>9}  {'QPS':>8}")
    rows = []
    for k in [10, 50, 100]:
        indices, _, _ = search_batch(index, queries, k=k, nprobe=16,
                                     batch_size=len(queries), timeout_s=60)
        rk    = recall_at_k(indices, gt[:, :k], k=k)
        bench = benchmark_search(index, queries, k=k, nprobe=16, runs=3)
        rows.append({"k": k, f"recall@{k}": round(rk, 4), **bench})
        print(f"  {k:>4}  {rk:>9.3f}  {bench['qps_mean']:>8.0f}")
    return rows

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== SQID: IVF/PQ parameter sweep ===\n")
    products, queries = load_data()

    index = build_or_load(products)

    print("\n=== cluster stats ===")
    stats = cluster_size_stats(index)
    for key, val in stats.items():
        print(f"  {key}: {val}")

    gt = get_ground_truth(products, queries, k=100)

    results = {
        "nprobe_sweep": nprobe_sweep(index, queries, gt),
        "k_sweep":      k_sweep(index, queries, gt),
    }

    out = INDEX_DIR / "sqid_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nresults → {out}")


if __name__ == "__main__":
    main()
