"""Evaluation utilities: recall, QPS, and pre-build diagnostics."""

import numpy as np
import faiss


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
    r  = retrieved[:, :k]     # (nq, k)
    gt = ground_truth[:, :k]  # (nq, k)

    # For each query and each retrieved item, check if it appears anywhere in gt row.
    # r[:, :, None] == gt[:, None, :] → (nq, k, k) bool; any() over last axis → (nq, k)
    hits = (r[:, :, None] == gt[:, None, :]).any(axis=2).sum()
    return float(hits) / (nq * k)
