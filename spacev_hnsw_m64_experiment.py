"""SPACEV-100M: HNSW M=32 vs M=64 recall comparison.

Hypothesis: HNSW M=32 recall plateaus at nprobe=512 because the graph exhausts
reachable candidates. Increasing M to 64 gives the graph more connections,
allowing it to reach more true nearest centroids and close the ~7pp recall gap
vs flat routing.

Reuses existing artifacts:
  - indexes/spacev_100m_hnsw_1B.index     (M=32, already built)
  - indexes/spacev_100m_gt.npy            (100M GT, already cached)
  - data/raw/spacev_base_100M.i8bin       (raw vectors for stream-add)

Builds fresh:
  - indexes/spacev_100m_hnsw64_trained.index
  - indexes/spacev_100m_hnsw64_1B.index

Results written to indexes/spacev_hnsw_m64_results.json.

Usage:
  python spacev_hnsw_m64_experiment.py
"""

import json
import time
from pathlib import Path

import faiss
import numpy as np

from build_index import SPACEV_100M_HNSW64_CONFIG, build_index, load_index, stream_add
from evaluate import benchmark_search, cluster_size_stats, recall_at_k
from search import search_batch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR        = Path("data")
INDEX_DIR       = Path("indexes")
BASE_100M_RAW   = DATA_DIR / "raw" / "spacev_base_100M.i8bin"
QUERY_FILE      = DATA_DIR / "query.npy"
GT_FILE         = INDEX_DIR / "spacev_100m_gt.npy"

IDX_M32         = INDEX_DIR / "spacev_100m_hnsw_1B.index"
TRAINED_M64     = INDEX_DIR / "spacev_100m_hnsw64_trained.index"
IDX_M64         = INDEX_DIR / "spacev_100m_hnsw64_1B.index"
PRIOR_RESULTS   = INDEX_DIR / "spacev_100m_results.json"

TRAIN_SIZE      = 3_000_000
N_THREADS       = 10

faiss.omp_set_num_threads(N_THREADS)

# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _read_raw_sample(path: Path, n: int, dim: int = 100) -> np.ndarray:
    with open(path, "rb") as f:
        f.seek(8)
        raw = np.frombuffer(f.read(n * dim), dtype=np.int8).reshape(n, dim)
    return raw.astype(np.float32)


def build_m64(train_f32: np.ndarray) -> faiss.Index:
    """Train + stream-add M=64 index; returns loaded index."""
    if IDX_M64.exists():
        index = load_index(str(IDX_M64))
        print(f"[m64] loaded existing index: ntotal={index.ntotal:,}")
        return index

    INDEX_DIR.mkdir(exist_ok=True)

    if not TRAINED_M64.exists():
        print(f"[m64] training IVF65536_HNSW64 on {TRAIN_SIZE:,} vectors ...")
        t0 = time.perf_counter()
        build_index(train_f32, SPACEV_100M_HNSW64_CONFIG,
                    name="spacev_100m_hnsw64", out_dir=str(INDEX_DIR), checkpoint=True)
        print(f"[m64] train total: {time.perf_counter() - t0:.1f}s")
    else:
        print(f"[m64] train checkpoint exists: {TRAINED_M64}")

    print(f"[m64] stream-adding 100M vectors ...")
    t0    = time.perf_counter()
    index = load_index(str(TRAINED_M64))
    stream_add(index, str(BASE_100M_RAW), chunk_size=500_000)
    faiss.write_index(index, str(IDX_M64))
    print(f"[m64] saved → {IDX_M64}  ({IDX_M64.stat().st_size / 1e9:.2f} GB)")
    print(f"[m64] add total: {time.perf_counter() - t0:.1f}s")
    return index

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def run_sweep(index: faiss.Index, queries: np.ndarray, gt: np.ndarray,
              label: str) -> list[dict]:
    rows = []
    for nprobe in [32, 64, 128, 256, 512, 1024]:
        indices, _, n_done = search_batch(
            index, queries, k=100, nprobe=nprobe,
            batch_size=2000, timeout_s=180,
        )
        if n_done < len(queries):
            print(f"  [{label}] nprobe={nprobe}: timed out ({n_done}/{len(queries)}); stopping")
            break
        r10   = recall_at_k(indices, gt, k=10)
        r100  = recall_at_k(indices, gt, k=100)
        bench = benchmark_search(index, queries[:500], k=100, nprobe=nprobe, runs=3)
        rows.append({"nprobe": nprobe, "recall@10": round(r10, 4),
                     "recall@100": round(r100, 4), **bench})
        print(f"  {nprobe:>6}  [{label}]  r@10={r10:.3f}  r@100={r100:.3f}  "
              f"QPS={bench['qps_mean']:,.0f}  threads={bench['omp_threads']}")
    return rows

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not BASE_100M_RAW.exists():
        print(f"ERROR: {BASE_100M_RAW} not found.")
        print("Download with:  python download_spacev.py --subset-size 100000000")
        return

    print("=== SPACEV-100M: HNSW M=32 vs M=64 recall comparison ===\n")
    queries   = np.load(QUERY_FILE).astype(np.float32)
    gt        = np.load(str(GT_FILE))
    print(f"queries: {queries.shape}  gt: {gt.shape}")

    # build M=64 (train + stream-add)
    train_f32 = _read_raw_sample(BASE_100M_RAW, TRAIN_SIZE)
    idx_m64   = build_m64(train_f32)
    del train_f32

    # load M=32
    idx_m32 = load_index(str(IDX_M32))
    print(f"[m32] loaded: ntotal={idx_m32.ntotal:,}")

    # cluster stats
    print("\n=== cluster stats ===")
    for label, idx in [("M=32", idx_m32), ("M=64", idx_m64)]:
        s = cluster_size_stats(idx)
        print(f"  {label}: balance_ratio={s['balance_ratio']:.3f}  "
              f"mean={s['mean']:.0f}  empty={s['empty_lists']}")

    # side-by-side sweep
    print(f"\n=== nprobe sweep  ntotal={idx_m32.ntotal:,}  k=100 ===")
    print(f"  {'nprobe':>6}  {'config':>6}  {'r@10':>7}  {'r@100':>7}  {'QPS':>8}")

    print("\n--- M=32 ---")
    rows_m32 = run_sweep(idx_m32, queries, gt, "M=32")

    print("\n--- M=64 ---")
    rows_m64 = run_sweep(idx_m64, queries, gt, "M=64")

    # side-by-side delta summary
    print("\n=== recall delta: M=64 − M=32 ===")
    print(f"  {'nprobe':>6}  {'r@10 M32':>9}  {'r@10 M64':>9}  {'Δr@10':>7}  {'r@100 M32':>10}  {'r@100 M64':>10}  {'Δr@100':>8}")
    m32_by_np = {r["nprobe"]: r for r in rows_m32}
    for r64 in rows_m64:
        np_ = r64["nprobe"]
        if np_ in m32_by_np:
            r32 = m32_by_np[np_]
            d10  = r64["recall@10"]  - r32["recall@10"]
            d100 = r64["recall@100"] - r32["recall@100"]
            print(f"  {np_:>6}  {r32['recall@10']:>9.3f}  {r64['recall@10']:>9.3f}  {d10:>+7.3f}"
                  f"  {r32['recall@100']:>10.3f}  {r64['recall@100']:>10.3f}  {d100:>+8.3f}")

    results = {"hnsw_m32": rows_m32, "hnsw_m64": rows_m64}
    out = INDEX_DIR / "spacev_hnsw_m64_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nresults → {out}")


if __name__ == "__main__":
    main()
