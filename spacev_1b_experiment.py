"""SPACEV-1B experiment: IVF262144_HNSW32,PQ25 at 1B scale.

Three phases:
  train — IVF262144_HNSW32 trained on 10M sample; saves trained checkpoint
  add   — stream-add from raw base.1B.i8bin (falls back to 10M if file absent)
  search — nprobe sweep (k=100) + QPS benchmark; writes spacev_1b_results.json

Artifacts written to indexes/:
  spacev_1b_trained.index       trained checkpoint (empty; used by stream_add)
  spacev_1b.index               trained + 10M vectors (side effect of train phase)
  spacev_1b_1B.index            trained + 1B vectors (stream_add output)
  spacev_1b_gt_<N>M.npy         cached GT for subsets; pre-downloaded gt100.npy used for full 1B
  spacev_1b_results.json        per-nprobe recall + QPS results

Usage:
  python spacev_1b_experiment.py
"""

import json
import time
from pathlib import Path

import faiss
import numpy as np

from build_index import SPACEV_1B_CONFIG, build_index, load_index, stream_add
from evaluate import benchmark_search, cluster_size_stats, recall_at_k
from search import search_batch

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------

DATA_DIR    = Path("data")
INDEX_DIR   = Path("indexes")
BASE_10M    = DATA_DIR / "base_10M.npy"
BASE_1B_RAW = DATA_DIR / "raw" / "base.1B.i8bin"
QUERY_FILE  = DATA_DIR / "query.npy"
GT_1B_FILE  = DATA_DIR / "gt100.npy"

TRAINED_IDX = INDEX_DIR / "spacev_1b_trained.index"
IDX_10M     = INDEX_DIR / "spacev_1b.index"
IDX_1B      = INDEX_DIR / "spacev_1b_1B.index"

N_THREADS   = 10  # M3 Max: 10 P-cores

faiss.omp_set_num_threads(N_THREADS)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[np.ndarray, np.ndarray]:
    base    = np.load(BASE_10M)                        # (10M, 100) int8
    queries = np.load(QUERY_FILE).astype(np.float32)  # (29316, 100)
    print(f"base:    {base.shape}  {base.dtype}")
    print(f"queries: {queries.shape}  {queries.dtype}")
    return base, queries

# ---------------------------------------------------------------------------
# Phase 1 — train
# ---------------------------------------------------------------------------

def phase_train(base: np.ndarray) -> None:
    """Train IVF262144_HNSW32 on 10M sample. Writes trained checkpoint + 10M index."""
    if TRAINED_IDX.exists():
        print(f"[train] checkpoint exists: {TRAINED_IDX}")
        return
    INDEX_DIR.mkdir(exist_ok=True)
    print("[train] training IVF262144_HNSW32 on 10M vectors ...")
    t0 = time.perf_counter()
    build_index(
        base.astype(np.float32),
        SPACEV_1B_CONFIG,
        name="spacev_1b",
        out_dir=str(INDEX_DIR),
        checkpoint=True,
    )
    print(f"[train] total: {time.perf_counter() - t0:.1f}s")

# ---------------------------------------------------------------------------
# Phase 2 — add
# ---------------------------------------------------------------------------

def phase_add(base: np.ndarray) -> tuple[faiss.Index, int]:
    """Return (index, n_vectors_in_index).

    Priority:
      1. Load spacev_1b_1B.index if it exists (resumed 1B run).
      2. Stream-add from base.1B.i8bin if present.
      3. Fall back to the 10M index from the train phase.
    """
    if IDX_1B.exists():
        index = load_index(str(IDX_1B))
        print(f"[add] loaded 1B index: ntotal={index.ntotal:,}")
        return index, index.ntotal

    if not BASE_1B_RAW.exists():
        print(f"[add] {BASE_1B_RAW} not found — using 10M index for evaluation")
        index = load_index(str(IDX_10M))
        return index, index.ntotal

    print(f"[add] stream-adding from {BASE_1B_RAW} ...")
    t0 = time.perf_counter()
    index = load_index(str(TRAINED_IDX))  # trained, empty
    n_added = stream_add(index, str(BASE_1B_RAW), chunk_size=500_000)
    faiss.write_index(index, str(IDX_1B))
    print(f"[add] saved → {IDX_1B}  ({IDX_1B.stat().st_size / 1e9:.1f} GB)")
    print(f"[add] total: {time.perf_counter() - t0:.1f}s")
    return index, n_added

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def get_ground_truth(
    index: faiss.Index,
    base: np.ndarray,
    queries: np.ndarray,
    k: int = 100,
) -> np.ndarray:
    """Return GT index array (nq, k).

    For full 1B: use pre-downloaded gt100.npy (computed by benchmark authors).
    For any subset: compute with IndexFlatL2 and cache to disk.
    """
    n = index.ntotal

    if n == 1_000_000_000 and GT_1B_FILE.exists():
        gt = np.load(GT_1B_FILE)[:, :k]
        print(f"[gt] pre-computed 1B GT  {gt.shape}")
        return gt

    gt_path = INDEX_DIR / f"spacev_1b_gt_{n // 1_000_000}M.npy"
    if gt_path.exists():
        gt = np.load(str(gt_path))
        print(f"[gt] cached GT: {gt_path}  {gt.shape}")
        return gt

    print(f"[gt] computing exact GT with IndexFlatL2 over {n:,} vectors ...")
    flat = faiss.IndexFlatL2(base.shape[1])
    flat.add(base[:n].astype(np.float32))
    t0 = time.perf_counter()
    _, gt = flat.search(queries, k)
    print(f"[gt] done in {time.perf_counter() - t0:.1f}s")
    np.save(str(gt_path), gt)
    print(f"[gt] cached → {gt_path}")
    return gt

# ---------------------------------------------------------------------------
# Phase 3 — search
# ---------------------------------------------------------------------------

def phase_search(index: faiss.Index, queries: np.ndarray, gt: np.ndarray) -> None:
    n = index.ntotal
    timeout = 300 if n >= 1_000_000_000 else 60

    print("\n=== cluster stats ===")
    stats = cluster_size_stats(index)
    for key, val in stats.items():
        print(f"  {key}: {val}")

    print(f"\n=== nprobe sweep  (k=100, ntotal={n:,}) ===")
    print(f"{'nprobe':>8}  {'recall@10':>10}  {'recall@100':>11}  {'QPS':>8}  threads")
    results = []

    for nprobe in [16, 32, 64, 128, 256, 512, 1024]:
        indices, _, n_searched = search_batch(
            index, queries, k=100, nprobe=nprobe,
            batch_size=2000, timeout_s=timeout,
        )
        if n_searched < len(queries):
            print(f"  nprobe={nprobe}: timed out after {n_searched}/{len(queries)} queries; stopping sweep")
            break

        r10  = recall_at_k(indices, gt, k=10)
        r100 = recall_at_k(indices, gt, k=100)
        bench = benchmark_search(index, queries[:200], k=100, nprobe=nprobe, runs=3)
        results.append({
            "nprobe": nprobe, "recall@10": round(r10, 4), "recall@100": round(r100, 4),
            **bench,
        })
        print(f"  {nprobe:>6}  {r10:>10.3f}  {r100:>11.3f}  "
              f"{bench['qps_mean']:>8.0f}  {bench['omp_threads']}")

    out = INDEX_DIR / "spacev_1b_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nresults → {out}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== SPACEV-1B: HNSW experiment ===\n")
    base, queries = load_data()

    phase_train(base)
    index, n_added = phase_add(base)
    print(f"\nindex.ntotal = {index.ntotal:,}")

    gt = get_ground_truth(index, base, queries, k=100)
    phase_search(index, queries, gt)


if __name__ == "__main__":
    main()
