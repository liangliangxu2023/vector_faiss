"""KNN search over a FAISS IVF/PQ index."""

import numpy as np
import faiss


def search(
    index: faiss.Index,
    queries: np.ndarray,
    k: int,
    nprobe: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Search the index for the k nearest neighbors of each query.

    Args:
        index:   Any FAISS index (IndexIVFPQ or IndexPreTransform wrapping one).
        queries: float32 array of shape (nq, d). Caller is responsible for
                 normalization (SQID) and dtype casting (SPACEV int8 → float32).
        k:       Number of neighbors to return.
        nprobe:  Number of IVF clusters to probe. Higher → better recall, lower QPS.

    Returns:
        indices:   int64 array of shape (nq, k) — row indices into the database.
        distances: float32 array of shape (nq, k) — L2 distances or IP scores
                   depending on the index metric. -1 in indices means no result.
    """
    _warn_k(index, k, nprobe)

    ivf = faiss.extract_index_ivf(index)
    ivf.nprobe = nprobe

    distances, indices = index.search(queries, k)
    return indices, distances


def search_batch(
    index: faiss.Index,
    queries: np.ndarray,
    k: int,
    nprobe: int = 16,
    batch_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Same as search() but processes queries in batches.

    Useful when nq is large and you want bounded peak memory or intermediate
    progress logging.
    """
    nq = len(queries)
    all_indices   = np.empty((nq, k), dtype=np.int64)
    all_distances = np.empty((nq, k), dtype=np.float32)

    ivf = faiss.extract_index_ivf(index)
    ivf.nprobe = nprobe
    _warn_k(index, k, nprobe)

    for start in range(0, nq, batch_size):
        end = min(start + batch_size, nq)
        d, i = index.search(queries[start:end], k)
        all_distances[start:end] = d
        all_indices[start:end]   = i

    return all_indices, all_distances


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _effective_search_pool(index: faiss.Index, nprobe: int) -> float:
    """Expected number of vectors examined per query (nprobe * avg cluster size)."""
    ivf = faiss.extract_index_ivf(index)
    return nprobe * (ivf.ntotal / ivf.nlist)


def _warn_k(index: faiss.Index, k: int, nprobe: int) -> None:
    """Warn if k exceeds 10% of the effective search pool.

    Beyond that threshold, PQ quantization noise dominates intra-cluster
    ordering and the ranked list becomes near-random.
    """
    pool = _effective_search_pool(index, nprobe)
    threshold = int(pool * 0.10)
    if k > threshold:
        print(
            f"WARNING: k={k} > 10% of effective search pool "
            f"({pool:.0f} vectors at nprobe={nprobe}). "
            f"Results beyond rank ~{threshold} are unreliable."
        )
