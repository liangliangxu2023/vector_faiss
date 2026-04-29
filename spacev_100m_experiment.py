"""SPACEV-100M: flat (Exp B) vs HNSW (Exp C) quantizer comparison.

Both experiments use identical data and nlist=65536. The only difference is
the coarse quantizer — flat O(nlist) vs HNSW O(log nlist). QPS difference
isolates routing overhead; recall should be nearly identical.

Prereq: download 100M data first:
  python download_spacev.py --subset-size 100000000

Phases (each resumable via checkpoints):
  1. train B and C on 3M sample
  2. stream-add 100M vectors into each
  3. compute ground truth (sharded flat L2, cached)
  4. nprobe sweep side-by-side, QPS benchmark
  5. write indexes/spacev_100m_results.json
"""

import json
import time
from pathlib import Path

import faiss
import numpy as np

from build_index import (
    SPACEV_100M_CONFIG, SPACEV_100M_HNSW_CONFIG,
    build_index, load_index, stream_add,
)
from evaluate import benchmark_search, cluster_size_stats, recall_at_k
from search import search_batch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR        = Path("data")
INDEX_DIR       = Path("indexes")
BASE_100M_RAW   = DATA_DIR / "raw" / "spacev_base_100M.i8bin"
QUERY_FILE      = DATA_DIR / "query.npy"

TRAINED_FLAT    = INDEX_DIR / "spacev_100m_flat_trained.index"
TRAINED_HNSW    = INDEX_DIR / "spacev_100m_hnsw_trained.index"
IDX_FLAT        = INDEX_DIR / "spacev_100m_flat_1B.index"   # _1B suffix avoids collision
IDX_HNSW        = INDEX_DIR / "spacev_100m_hnsw_1B.index"   # with 3M train-phase index
GT_FILE         = INDEX_DIR / "spacev_100m_gt.npy"

N_THREADS       = 10   # M3 Max: 10 P-cores
TRAIN_SIZE      = 3_000_000

faiss.omp_set_num_threads(N_THREADS)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_raw_sample(path: Path, n: int, dim: int = 100) -> np.ndarray:
    """Read first n vectors from a .i8bin file as float32."""
    with open(path, "rb") as f:
        f.seek(8)
        raw = np.frombuffer(f.read(n * dim), dtype=np.int8).reshape(n, dim)
    return raw.astype(np.float32)


def compute_gt_sharded(
    path: Path,
    queries: np.ndarray,
    k: int = 100,
    chunk_size: int = 10_000_000,
) -> np.ndarray:
    """Exact GT via sharded flat L2 — avoids loading 100M × 100 × 4B = 40GB at once.

    Streams the raw .i8bin in chunks, searches each chunk, merges top-k.
    Peak RAM per chunk: ~4 GB float32 + FAISS index overhead.
    """
    if GT_FILE.exists():
        gt = np.load(str(GT_FILE))
        print(f"[gt] loaded cached GT  {gt.shape}")
        return gt

    with open(path, "rb") as f:
        dim = int(np.frombuffer(f.read(4) + f.read(4), dtype=np.int32)[1])
    n_vecs = (path.stat().st_size - 8) // dim
    nq     = len(queries)

    best_d = np.full((nq, k), np.inf,  dtype=np.float32)
    best_i = np.full((nq, k), -1,      dtype=np.int64)

    offset = 0
    t0     = time.perf_counter()
    with open(path, "rb") as f:
        f.seek(8)
        while offset < n_vecs:
            batch = min(chunk_size, n_vecs - offset)
            raw   = np.frombuffer(f.read(batch * dim), dtype=np.int8).reshape(batch, dim)
            flat  = faiss.IndexFlatL2(dim)
            flat.add(raw.astype(np.float32))
            D, I  = flat.search(queries, k)
            del flat, raw

            I = I.astype(np.int64) + offset
            merged_d = np.concatenate([best_d, D],  axis=1)
            merged_i = np.concatenate([best_i, I],  axis=1)
            order    = np.argsort(merged_d, axis=1)[:, :k]
            best_d   = np.take_along_axis(merged_d, order, axis=1)
            best_i   = np.take_along_axis(merged_i, order, axis=1)

            offset += batch
            elapsed = time.perf_counter() - t0
            print(f"  [gt] shard {offset // chunk_size}/{n_vecs // chunk_size}"
                  f"  ({offset:,}/{n_vecs:,})  {elapsed:.0f}s")

    np.save(str(GT_FILE), best_i)
    print(f"[gt] cached → {GT_FILE}")
    return best_i

# ---------------------------------------------------------------------------
# Phase 1 — train
# ---------------------------------------------------------------------------

def phase_train(train_f32: np.ndarray) -> None:
    INDEX_DIR.mkdir(exist_ok=True)
    for trained_path, config, label in [
        (TRAINED_FLAT, SPACEV_100M_CONFIG,      "flat "),
        (TRAINED_HNSW, SPACEV_100M_HNSW_CONFIG, "hnsw "),
    ]:
        if trained_path.exists():
            print(f"[train {label}] checkpoint exists: {trained_path}")
            continue
        name = "spacev_100m_flat" if "flat" in label else "spacev_100m_hnsw"
        print(f"[train {label}] training {config.nlist} centroids on {TRAIN_SIZE:,} vectors ...")
        t0 = time.perf_counter()
        build_index(train_f32, config, name=name, out_dir=str(INDEX_DIR), checkpoint=True)
        print(f"[train {label}] total: {time.perf_counter() - t0:.1f}s")

# ---------------------------------------------------------------------------
# Phase 2 — add
# ---------------------------------------------------------------------------

def phase_add(trained_path: Path, final_path: Path, label: str) -> faiss.Index:
    if final_path.exists():
        index = load_index(str(final_path))
        print(f"[add {label}] loaded: ntotal={index.ntotal:,}")
        return index
    print(f"[add {label}] stream-adding 100M vectors ...")
    t0      = time.perf_counter()
    index   = load_index(str(trained_path))
    n_added = stream_add(index, str(BASE_100M_RAW), chunk_size=500_000)
    faiss.write_index(index, str(final_path))
    print(f"[add {label}] saved → {final_path}  ({final_path.stat().st_size / 1e9:.2f} GB)")
    print(f"[add {label}] total: {time.perf_counter() - t0:.1f}s")
    return index

# ---------------------------------------------------------------------------
# Phase 3 — search
# ---------------------------------------------------------------------------

def phase_search(
    idx_b: faiss.Index,
    idx_c: faiss.Index,
    queries: np.ndarray,
    gt: np.ndarray,
) -> None:
    print("\n=== cluster stats ===")
    for label, idx in [("flat (B)", idx_b), ("hnsw (C)", idx_c)]:
        s = cluster_size_stats(idx)
        print(f"  {label}: balance_ratio={s['balance_ratio']:.3f}  "
              f"mean={s['mean']:.0f}  empty={s['empty_lists']}")

    results = {"flat": [], "hnsw": []}
    print(f"\n=== nprobe sweep  ntotal={idx_b.ntotal:,}  k=100 ===")
    print(f"{'nprobe':>8}  {'exp':>6}  {'recall@10':>10}  {'recall@100':>11}  {'QPS':>8}  threads")

    for nprobe in [32, 64, 128, 256, 512, 1024]:
        for label, idx, bucket in [("flat", idx_b, "flat"), ("hnsw", idx_c, "hnsw")]:
            indices, _, n_done = search_batch(
                idx, queries, k=100, nprobe=nprobe,
                batch_size=2000, timeout_s=180,
            )
            if n_done < len(queries):
                print(f"  nprobe={nprobe} {label}: timed out ({n_done}/{len(queries)}); stopping")
                results[bucket] = None
                break

            r10   = recall_at_k(indices, gt, k=10)
            r100  = recall_at_k(indices, gt, k=100)
            bench = benchmark_search(idx, queries[:500], k=100, nprobe=nprobe, runs=3)
            row   = {"nprobe": nprobe, "recall@10": round(r10, 4),
                     "recall@100": round(r100, 4), **bench}
            results[bucket].append(row)
            print(f"  {nprobe:>6}  {label:>6}  {r10:>10.3f}  {r100:>11.3f}  "
                  f"{bench['qps_mean']:>8.0f}  {bench['omp_threads']}")

        if results["flat"] is None or results["hnsw"] is None:
            break

    out = INDEX_DIR / "spacev_100m_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nresults → {out}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not BASE_100M_RAW.exists():
        print(f"ERROR: {BASE_100M_RAW} not found.")
        print("Download with:  python download_spacev.py --subset-size 100000000")
        return

    print("=== SPACEV-100M: flat vs HNSW routing comparison ===\n")
    queries   = np.load(QUERY_FILE).astype(np.float32)
    train_f32 = _read_raw_sample(BASE_100M_RAW, TRAIN_SIZE)
    print(f"queries: {queries.shape}  train sample: {train_f32.shape}")

    phase_train(train_f32)
    del train_f32  # free before adding 100M

    idx_b = phase_add(TRAINED_FLAT, IDX_FLAT, "flat")
    idx_c = phase_add(TRAINED_HNSW, IDX_HNSW, "hnsw")

    gt = compute_gt_sharded(BASE_100M_RAW, queries, k=100)
    phase_search(idx_b, idx_c, queries, gt)


if __name__ == "__main__":
    main()
