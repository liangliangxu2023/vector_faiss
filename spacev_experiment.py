"""SPACEV-10M experiment: IVF/PQ parameter sweep for same-modal L2 retrieval.

Sweeps nprobe and k at fixed config (nlist=4096, M=25, no OPQ).
Ground truth computed once with IndexFlatL2 over the 10M subset and cached.

Results written to indexes/spacev_results.json.

Usage:
  python spacev_experiment.py

Expected (from prior runs):
  nprobe=64  recall@10 ~0.702  recall@100 ~0.693  QPS ~9.5K
  Recall ceiling ~0.725 at nprobe=256 — PQ quantization bottleneck.
  recall@10 ≈ recall@100 across all nprobe (sign of good geometric clustering).
"""

import json
import time
from pathlib import Path

import faiss
import numpy as np

from build_index import SPACEV_10M_CONFIG, build_index, load_index
from evaluate import benchmark_search, cluster_size_stats, recall_at_k
from search import search_batch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR   = Path("data")
INDEX_DIR  = Path("indexes")
BASE_FILE  = DATA_DIR / "base_10M.npy"
QUERY_FILE = DATA_DIR / "query.npy"
INDEX_FILE = INDEX_DIR / "spacev_10m.index"
GT_FILE    = INDEX_DIR / "spacev_10m_gt.npy"

faiss.omp_set_num_threads(10)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data() -> tuple[np.ndarray, np.ndarray]:
    base    = np.load(BASE_FILE)                        # (10M, 100) int8
    queries = np.load(QUERY_FILE).astype(np.float32)   # (29316, 100)
    print(f"base:    {base.shape}  {base.dtype}")
    print(f"queries: {queries.shape}  {queries.dtype}")
    return base, queries


def get_ground_truth(base: np.ndarray, queries: np.ndarray, k: int = 100) -> np.ndarray:
    if GT_FILE.exists():
        gt = np.load(str(GT_FILE))
        print(f"[gt] loaded cached GT  {gt.shape}")
        return gt
    print("[gt] computing exact GT with IndexFlatL2 ...")
    t0   = time.perf_counter()
    flat = faiss.IndexFlatL2(base.shape[1])
    flat.add(base.astype(np.float32))
    _, gt = flat.search(queries, k)
    print(f"[gt] done in {time.perf_counter() - t0:.1f}s")
    np.save(str(GT_FILE), gt)
    return gt


def build_or_load(base: np.ndarray) -> faiss.Index:
    if INDEX_FILE.exists():
        index = load_index(str(INDEX_FILE))
        print(f"[build] loaded existing index: ntotal={index.ntotal:,}")
        return index
    INDEX_DIR.mkdir(exist_ok=True)
    print(f"[build] training IVF4096,PQ25 on {len(base):,} vectors ...")
    t0    = time.perf_counter()
    index = build_index(base.astype(np.float32), SPACEV_10M_CONFIG,
                        name="spacev_10m", out_dir=str(INDEX_DIR))
    print(f"[build] total: {time.perf_counter() - t0:.1f}s")
    return index

# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def nprobe_sweep(index: faiss.Index, queries: np.ndarray, gt: np.ndarray) -> list[dict]:
    print("\n=== nprobe sweep  (k=100) ===")
    print(f"{'nprobe':>8}  {'recall@10':>10}  {'recall@100':>11}  {'QPS':>8}")
    rows = []
    for nprobe in [8, 16, 32, 64, 128, 256]:
        indices, _, _ = search_batch(index, queries, k=100, nprobe=nprobe,
                                     batch_size=len(queries), timeout_s=120)
        r10  = recall_at_k(indices, gt, k=10)
        r100 = recall_at_k(indices, gt, k=100)
        bench = benchmark_search(index, queries, k=100, nprobe=nprobe, runs=3)
        rows.append({"nprobe": nprobe, "recall@10": round(r10, 4),
                     "recall@100": round(r100, 4), **bench})
        print(f"  {nprobe:>6}  {r10:>10.3f}  {r100:>11.3f}  {bench['qps_mean']:>8.0f}")
    return rows


def k_sweep(index: faiss.Index, queries: np.ndarray, gt: np.ndarray) -> list[dict]:
    print("\n=== k sweep  (nprobe=64) ===")
    print(f"{'k':>6}  {'recall@k':>9}  {'QPS':>8}")
    rows = []
    for k in [10, 50, 100]:
        indices, _, _ = search_batch(index, queries, k=k, nprobe=64,
                                     batch_size=len(queries), timeout_s=120)
        rk    = recall_at_k(indices, gt[:, :k], k=k)
        bench = benchmark_search(index, queries, k=k, nprobe=64, runs=3)
        rows.append({"k": k, f"recall@{k}": round(rk, 4), **bench})
        print(f"  {k:>4}  {rk:>9.3f}  {bench['qps_mean']:>8.0f}")
    return rows

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== SPACEV-10M: IVF/PQ parameter sweep ===\n")
    base, queries = load_data()

    index = build_or_load(base)

    print("\n=== cluster stats ===")
    stats = cluster_size_stats(index)
    for key, val in stats.items():
        print(f"  {key}: {val}")

    gt = get_ground_truth(base, queries, k=100)

    results = {
        "nprobe_sweep": nprobe_sweep(index, queries, gt),
        "k_sweep":      k_sweep(index, queries, gt),
    }

    out = INDEX_DIR / "spacev_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nresults → {out}")


if __name__ == "__main__":
    main()
