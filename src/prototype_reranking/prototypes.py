"""
prototypes.py
=============
Prototype construction from patch embeddings.

Provides:
  build_fixed_prototypes(patches, K)  -- plain KMeans, used by fixed reranking.
  build_adaptive_entry(patches, ...)  -- auto-selects K* by silhouette, used by
                                        adaptive reranking.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def build_fixed_prototypes(
    patches: np.ndarray,
    K: int,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster patch embeddings into K prototypes using KMeans.

    Parameters
    ----------
    patches:
        (P, d) patch embedding matrix. Rows should already be L2-normalised if
        the downstream scorer uses cosine similarity.
    K:
        Number of clusters.
    seed:
        Random state for KMeans.

    Returns
    -------
    prototypes : np.ndarray of shape (K, d), L2-normalised centroids.
    assignments : np.ndarray of shape (P,) with cluster index per patch.
    """
    patches = np.asarray(patches, dtype=np.float32)
    K = min(K, patches.shape[0])
    if K == 1:
        return l2_normalize(patches.mean(axis=0, keepdims=True)), np.zeros(patches.shape[0], dtype=np.int32)
    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    assignments = km.fit_predict(patches).astype(np.int32)
    centroids = np.stack([
        patches[assignments == k].mean(axis=0) if np.any(assignments == k) else km.cluster_centers_[k]
        for k in range(K)
    ], axis=0).astype(np.float32)
    return l2_normalize(centroids), assignments


def _coverage(patches: np.ndarray, prototypes: np.ndarray) -> float:
    """Mean max cosine similarity from each patch to its closest prototype."""
    sim = patches @ prototypes.T
    return float(np.max(sim, axis=1).mean())


def build_adaptive_entry(
    case_id: str,
    patches: np.ndarray,
    *,
    K_range: tuple[int, int] = (1, 12),
    seed: int = 42,
) -> dict[str, Any]:
    """Auto-select K* and build the adaptive prototype entry for one case.

    K* is chosen by silhouette score (or coverage when only 1 prototype is
    considered). The returned dict is compatible with
    ``AdaptivePrototypeReranker.build_bank``.

    Parameters
    ----------
    case_id : str
    patches : (P, d) L2-normalised patch embeddings.
    K_range : (min_K, max_K) inclusive.
    seed : random state.

    Returns
    -------
    dict with keys: case_id, K_star, prototypes, assignments, cluster_sizes,
                    support, q_proto_original.
    """
    patches = np.asarray(patches, dtype=np.float32)
    P = patches.shape[0]
    lo, hi = int(K_range[0]), min(int(K_range[1]), P)

    best_K, best_score, best_protos, best_assign = lo, -2.0, None, None
    for K in range(lo, hi + 1):
        protos, assign = build_fixed_prototypes(patches, K, seed=seed)
        if K == 1:
            score = _coverage(patches, protos)
        else:
            try:
                score = float(silhouette_score(patches, assign, metric="cosine", sample_size=min(P, 500)))
            except Exception:
                score = _coverage(patches, protos)
        if score > best_score:
            best_score, best_K, best_protos, best_assign = score, K, protos, assign

    cluster_sizes = np.bincount(best_assign, minlength=best_K).astype(np.int32)
    support = float(np.count_nonzero(cluster_sizes)) / max(best_K, 1)
    q_proto_original = float(np.clip(best_score if best_score >= 0 else 0.0, 0.0, 1.0))

    return {
        "case_id": case_id,
        "K_star": best_K,
        "prototypes": best_protos,
        "assignments": best_assign.astype(np.int32),
        "cluster_sizes": cluster_sizes,
        "support": support,
        "q_proto_original": q_proto_original,
        "n_patches": P,
    }
