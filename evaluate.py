"""Evaluation utilities: recall, QPS, and pre-build diagnostics."""

import time

import numpy as np
import faiss


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """Mean recall@k across all queries.

    For each query: fraction of true top-k neighbors found in the retrieved top-k.
    Averaged over all queries.

    Args:
        retrieved:    int array (nq, >=k) — ANN index results (row positions).
        ground_truth: int array (nq, >=k) — exact nearest neighbors.
        k:            cutoff rank.

    Returns:
        Scalar in [0, 1]. 1.0 = perfect recall.
    """
    nq = len(retrieved)
    r  = retrieved[:, :k]
    gt = ground_truth[:, :k]
    hits = (r[:, :, None] == gt[:, None, :]).any(axis=2).sum()
    return float(hits) / (nq * k)


# ---------------------------------------------------------------------------
# QPS / latency
# ---------------------------------------------------------------------------

def benchmark_search(
    index: faiss.Index,
    queries: np.ndarray,
    k: int,
    nprobe: int,
    runs: int = 5,
) -> dict:
    """Measure search throughput and latency over multiple runs.

    Args:
        index:   Trained and populated FAISS index.
        queries: float32 array (nq, d).
        k:       Number of neighbors per query.
        nprobe:  IVF clusters to probe.
        runs:    Number of timed repetitions (first run excluded as warm-up).

    Returns:
        dict with keys: qps_mean, qps_std, latency_ms_mean, latency_ms_std, nq, k, nprobe
    """
    ivf = faiss.extract_index_ivf(index)
    ivf.nprobe = nprobe

    nq = len(queries)
    times = []

    for i in range(runs + 1):
        t0 = time.perf_counter()
        index.search(queries, k)
        elapsed = time.perf_counter() - t0
        if i > 0:  # skip warm-up
            times.append(elapsed)

    times = np.array(times)
    qps = nq / times

    return {
        "qps_mean":         round(float(qps.mean()), 1),
        "qps_std":          round(float(qps.std()), 1),
        "latency_ms_mean":  round(float(times.mean() * 1000), 2),
        "latency_ms_std":   round(float(times.std() * 1000), 2),
        "nq":               nq,
        "k":                k,
        "nprobe":           nprobe,
    }


# ---------------------------------------------------------------------------
# Cluster quality (post-build)
# ---------------------------------------------------------------------------

def cluster_size_stats(index: faiss.Index) -> dict:
    """Cluster size distribution after index.add().

    Works for both plain IndexIVFPQ and OPQ-wrapped IndexPreTransform.
    balance_ratio = std/mean; lower is better (< 0.5 is well-balanced).
    """
    ivf = faiss.extract_index_ivf(index)
    sizes = np.array([ivf.get_list_size(i) for i in range(ivf.nlist)])
    mean = float(sizes.mean())
    return {
        "min":           int(sizes.min()),
        "max":           int(sizes.max()),
        "mean":          round(mean, 2),
        "std":           round(float(sizes.std()), 2),
        "balance_ratio": round(float(sizes.std() / mean), 3),
        "empty_lists":   int((sizes == 0).sum()),
    }


# ---------------------------------------------------------------------------
# OPQ diagnostic (pre-build)
# ---------------------------------------------------------------------------

def subspace_variance(vectors: np.ndarray, M: int) -> np.ndarray:
    """Variance of each PQ subspace.

    Uniform variance across subspaces → ratio ≈ 1 → OPQ not needed.
    Uneven variance → high ratio → OPQ will help.
    """
    d = vectors.shape[1]
    sub_d = d // M
    return np.array([vectors[:, i * sub_d: (i + 1) * sub_d].var() for i in range(M)])


def opq_needed(
    vectors: np.ndarray,
    M: int,
    threshold: float = 5.0,
) -> tuple[bool, float]:
    """True if max/min subspace variance ratio exceeds threshold.

    Args:
        vectors:   Sample of database vectors (10K is sufficient).
        M:         Number of PQ sub-quantizers.
        threshold: Ratio above which OPQ is recommended (default 5.0).

    Returns:
        (needed, ratio): bool and the measured max/min ratio.
    """
    sv = subspace_variance(vectors, M)
    ratio = float(sv.max() / sv.min())
    print(f"subspace variance  min={sv.min():.4f}  max={sv.max():.4f}  ratio={ratio:.1f}x")
    needed = ratio >= threshold
    print(f"OPQ {'recommended' if needed else 'not needed'}  (threshold={threshold}x)")
    return needed, ratio
