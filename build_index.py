import dataclasses
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from config import IndexConfig
from evaluate import cluster_size_stats

# ---------------------------------------------------------------------------
# Dataset presets
# ---------------------------------------------------------------------------

SQID_CONFIG = IndexConfig(
    d=768, nlist=256, M=16, metric="IP", train_size=None, opq=True
)

SPACEV_10M_CONFIG = IndexConfig(
    d=100, nlist=4096, M=25, metric="L2", train_size=500_000
)

SPACEV_100M_CONFIG = IndexConfig(
    d=100, nlist=65536, M=25, metric="L2", train_size=3_000_000
)

SPACEV_100M_HNSW_CONFIG = IndexConfig(
    d=100, nlist=65536, M=25, metric="L2", train_size=3_000_000, hnsw_m=32
)

SPACEV_100M_HNSW64_CONFIG = IndexConfig(
    d=100, nlist=65536, M=25, metric="L2", train_size=3_000_000, hnsw_m=64
)

SPACEV_1B_CONFIG = IndexConfig(
    d=100, nlist=262_144, M=25, metric="L2", train_size=10_000_000, hnsw_m=32
)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _factory_string(config: IndexConfig) -> str:
    if config.hnsw_m > 0:
        ivf = f"IVF{config.nlist}_HNSW{config.hnsw_m}"
    else:
        ivf = f"IVF{config.nlist}"
    pq = f"PQ{config.M}"
    if config.opq:
        return f"OPQ{config.M}_{config.d},{ivf},{pq}"
    return f"{ivf},{pq}"



def _print_speedup_tips(config: IndexConfig) -> None:
    print("\n--- speed-up suggestions ---")
    print(f"1. faiss.omp_set_num_threads(N)     — current default may be under-utilised")
    print(f"2. increase chunk_size              (current: {config.chunk_size:,})")
    print(f"3. reduce nlist                     (current: {config.nlist}) — trades recall for speed")
    print(f"4. reduce M                         (current: {config.M}) — trades recall for speed")
    print(f"5. index.by_residual = False        — skip residual subtraction (~5–10% recall cost)")
    print(f"6. pre-save float32 vectors to disk — avoid repeated int8 cast (SPACEV only)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index(
    vectors: np.ndarray,
    config: IndexConfig,
    name: str,
    out_dir: str = ".",
    checkpoint: bool = False,
    timeout_s: float | None = None,
) -> faiss.Index:
    """Train an IVF/PQ index, add vectors, save to disk, return the live index.

    Callers are responsible for float32 dtype, normalization (SQID),
    and int8→float32 casting (SPACEV) before passing vectors in.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- create index ---
    metric_flag = faiss.METRIC_INNER_PRODUCT if config.metric == "IP" else faiss.METRIC_L2
    index = faiss.index_factory(config.d, _factory_string(config), metric_flag)
    if config.hnsw_m == 0:
        # .cp is on the flat quantizer only; HNSW quantizer uses its own graph params
        faiss.extract_index_ivf(index).cp.niter = config.niter

    # --- train ---
    trained_path = out / f"{name}_trained.index"
    n_train = config.train_size or len(vectors)

    if checkpoint and trained_path.exists():
        print(f"resuming from {trained_path}, skipping train")
        index = faiss.read_index(str(trained_path))
        train_s = 0.0
    else:
        t0 = time.perf_counter()
        index.train(vectors[:n_train])
        train_s = time.perf_counter() - t0
        assert index.is_trained
        print(f"trained on {n_train:,} vectors in {train_s:.1f}s")
        if checkpoint:
            faiss.write_index(index, str(trained_path))
            print(f"checkpoint saved → {trained_path}")

    # --- add ---
    early_stop = False
    t0 = time.perf_counter()
    total = len(vectors)

    for start in range(0, total, config.chunk_size):
        index.add(vectors[start : start + config.chunk_size])
        elapsed = time.perf_counter() - t0
        print(f"  added {index.ntotal:,}/{total:,}  ({elapsed:.1f}s)")
        if timeout_s is not None and elapsed >= timeout_s:
            print(f"  early stop: {elapsed:.1f}s >= {timeout_s}s")
            early_stop = True
            break

    add_s = time.perf_counter() - t0
    if not early_stop:
        assert index.ntotal == total

    # --- save ---
    stem       = f"{name}_partial" if early_stop else name
    index_path = out / f"{stem}.index"
    meta_path  = out / f"{stem}_meta.json"

    faiss.write_index(index, str(index_path))

    meta = {
        "name":                    stem,
        "partial":                 early_stop,
        "n_vectors_total":         total,
        "config":                  {**dataclasses.asdict(config), "train_size": n_train},
        "ntotal":                  index.ntotal,
        "opq":                     config.opq,
        "subspace_variance_ratio": None,  # patched by caller via evaluate.opq_needed()
        "cluster_size_stats":      cluster_size_stats(index),
        "timing": {
            "train_s": round(train_s, 2),
            "add_s":   round(add_s, 2),
            "total_s": round(train_s + add_s, 2),
        },
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "omp_threads": faiss.omp_get_max_threads(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"saved → {index_path}  ({index_path.stat().st_size / 1e6:.0f} MB)")

    if early_stop:
        _print_speedup_tips(config)

    return index


def load_index(path: str) -> faiss.Index:
    return faiss.read_index(path)


def stream_add(
    index: faiss.Index,
    path: str,
    chunk_size: int = 500_000,
    timeout_s: float | None = None,
) -> int:
    """Add vectors from a raw .i8bin file to a trained index without loading all into RAM.

    Reads the file in chunks, casts int8→float32 per chunk, and calls index.add().
    Supports early stop via timeout_s.

    Args:
        index:      Trained FAISS index (index.is_trained must be True).
        path:       Path to a raw .i8bin file (4-byte n_vecs header, 4-byte dim header, then int8 data).
        chunk_size: Vectors per chunk. Default 500K ≈ 200 MB float32.
        timeout_s:  Stop early if elapsed >= this value. None = no limit.

    Returns:
        Number of vectors successfully added.
    """
    assert index.is_trained, "index must be trained before stream_add"

    with open(path, "rb") as f:
        n_vecs_header = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        dim           = int(np.frombuffer(f.read(4), dtype=np.int32)[0])

    # derive true vector count from file size — handles partial downloads where
    # the header still reflects the full dataset (e.g. 1B header on a 100M file)
    n_vecs_file = (Path(path).stat().st_size - 8) // dim
    n_vecs = n_vecs_file
    if n_vecs != n_vecs_header:
        print(f"  header says {n_vecs_header:,} vectors; file has {n_vecs:,} — using file size")

    t0 = time.perf_counter()
    n_added = 0

    with open(path, "rb") as f:
        f.seek(8)  # skip header
        while n_added < n_vecs:
            batch = min(chunk_size, n_vecs - n_added)
            raw   = np.frombuffer(f.read(batch * dim), dtype=np.int8).reshape(batch, dim)
            index.add(raw.astype(np.float32))
            n_added += batch

            elapsed = time.perf_counter() - t0
            qps     = n_added / elapsed
            print(f"  added {n_added:,}/{n_vecs:,}  ({elapsed:.1f}s, {qps:.0f} vecs/s)")

            if timeout_s is not None and elapsed >= timeout_s:
                print(f"  early stop: {elapsed:.1f}s >= {timeout_s}s ({n_added:,}/{n_vecs:,} added)")
                break

    elapsed = time.perf_counter() - t0
    print(f"stream_add done: {n_added:,} vectors in {elapsed:.1f}s  ({n_added / elapsed:.0f} vecs/s avg)")
    return n_added


def dump_debug(
    index: faiss.Index,
    vectors: np.ndarray,
    out_dir: str,
    name: str,
    ids: np.ndarray | None = None,
    n_samples: int | None = None,
) -> None:
    """Write centroid assignments to CSV files. n_samples=None uses all vectors.

    ids: optional external IDs (e.g. product_ids) aligned with vectors.
    Outputs:
      {out_dir}/{name}_assignments.csv  — one row per vector
      {out_dir}/{name}_centroids.csv    — one row per centroid with assigned vec_ids
    """
    from collections import defaultdict

    ivfpq  = faiss.downcast_index(faiss.extract_index_ivf(index))
    sample = vectors if n_samples is None else vectors[:n_samples]
    n      = len(sample)
    ids    = ids[:n] if ids is not None else None

    if isinstance(index, faiss.IndexPreTransform):
        vt          = faiss.downcast_VectorTransform(index.chain.at(0))
        transformed = vt.apply_py(sample)
    else:
        transformed = sample

    dists, centroid_ids = ivfpq.quantizer.search(transformed, 1)

    id_col = ids if ids is not None else np.arange(n).astype(str)
    id_header = "product_id" if ids is not None else "vec_id"

    # --- assignments.csv ---
    a_path = Path(out_dir) / f"{name}_assignments.csv"
    rows = [f"{id_header},centroid_id,dist_to_centroid,cluster_size"]
    for i in range(n):
        cid = int(centroid_ids[i, 0])
        rows.append(f"{id_col[i]},{cid},{dists[i, 0]:.6f},{ivfpq.get_list_size(cid)}")
    a_path.write_text("\n".join(rows))
    print(f"assignments → {a_path}  ({a_path.stat().st_size / 1024:.1f} KB)")

    # --- centroids.csv ---
    centroid_to_ids = defaultdict(list)
    for i in range(n):
        centroid_to_ids[int(centroid_ids[i, 0])].append(id_col[i])

    c_path = Path(out_dir) / f"{name}_centroids.csv"
    rows = [f"centroid_id,cluster_size,{id_header}s"]
    for cid, members in sorted(centroid_to_ids.items()):
        rows.append(f"{cid},{ivfpq.get_list_size(cid)},{members}")
    c_path.write_text("\n".join(rows))
    print(f"centroids  → {c_path}  ({c_path.stat().st_size / 1024:.1f} KB)")
