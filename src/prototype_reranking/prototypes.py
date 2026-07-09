"""Prototype construction from patch embeddings.

K* selection (adaptive): for each candidate K in k_grid, compute
    coverage = mean max-cosine-similarity from each patch to its nearest prototype
    support  = clip(patches_per_proto / good_support, 0, 1)
    utility  = coverage * support
K* is the smallest K whose utility is within near_best_ratio of the best
utility across the grid. This favours parsimonious representations while
ensuring adequate patch coverage and cluster support.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. Accepts 1-D or 2-D input."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = float(np.linalg.norm(x))
        return x / max(n, 1e-12)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def _clip01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0)) if math.isfinite(v) else 0.0


def build_fixed_prototypes(
    patches: np.ndarray,
    K: int,
    *,
    seed: int = 42,
    n_init: int = 3,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster patch embeddings into K L2-normalised prototypes via MiniBatchKMeans.

    Parameters
    ----------
    patches : (P, d) L2-normalised patch embedding matrix.
    K : number of prototypes.

    Returns
    -------
    prototypes : (K, d) L2-normalised centroids.
    assignments : (P,) cluster index per patch.
    """
    patches = np.asarray(patches, dtype=np.float32)
    K = min(K, patches.shape[0])
    batch = max(K * 10, 256, patches.shape[0])
    km = MiniBatchKMeans(
        n_clusters=K, random_state=seed, n_init=n_init,
        max_iter=max_iter, batch_size=batch,
    )
    assignments = km.fit_predict(patches).astype(np.int32)
    return l2_normalize(np.asarray(km.cluster_centers_, dtype=np.float32)), assignments


def build_adaptive_entry(
    case_id: str,
    patches: np.ndarray,
    *,
    k_grid: tuple[int, ...] = (2, 4, 6, 8, 12),
    min_support: int = 6,
    good_support: int = 20,
    near_best_ratio: float = 0.97,
    seed: int = 42,
    n_init: int = 3,
    max_iter: int = 100,
) -> dict[str, Any]:
    """Auto-select K* for one case using the utility criterion.

    K* is the smallest K in k_grid such that utility(K) is within
    near_best_ratio of the maximum utility across valid K values.

    Parameters
    ----------
    case_id : str
    patches : (P, d) L2-normalised patch embeddings.
    k_grid : candidate K values to evaluate.
    min_support : minimum patches per prototype (filters invalid K values).
    good_support : reference patches-per-proto for normalising support.
    near_best_ratio : threshold fraction of best utility.

    Returns
    -------
    dict with keys: case_id, K_star, prototypes, assignments, cluster_sizes,
                    support, coverage, utility, q_proto_base, n_patches.
    """
    patches = np.asarray(patches, dtype=np.float32)
    n = patches.shape[0]

    if n < 2:
        proto = l2_normalize(patches.mean(axis=0) if n > 0 else np.zeros(patches.shape[1]))
        return {
            "case_id": case_id, "K_star": 1,
            "prototypes": proto.reshape(1, -1),
            "assignments": np.zeros(n, dtype=np.int32),
            "cluster_sizes": np.array([n], dtype=np.int32),
            "support": _clip01(n / max(good_support, 1)),
            "coverage": 0.0, "utility": 0.0,
            "q_proto_base": 0.0, "n_patches": n,
        }

    valid_ks = [k for k in sorted(k_grid) if 0 < k <= n and (n / k) >= min_support]
    if not valid_ks:
        valid_ks = [k for k in sorted(k_grid) if 0 < k <= n] or [1]

    results: dict[int, dict] = {}
    for k in valid_ks:
        protos, assign = build_fixed_prototypes(
            patches, k, seed=seed, n_init=n_init, max_iter=max_iter
        )
        sim = patches @ protos.T
        coverage = float(np.max(sim, axis=1).mean())
        support = _clip01((n / k) / max(good_support, 1))
        utility = coverage * support if math.isfinite(coverage) else 0.0
        cs = np.bincount(assign, minlength=k).astype(np.int32)
        results[k] = {
            "prototypes": protos, "assignments": assign, "cluster_sizes": cs,
            "coverage": coverage, "support": support, "utility": utility,
        }

    best_utility = max(r["utility"] for r in results.values())
    threshold = near_best_ratio * best_utility
    k_star = min(k for k in valid_ks if results[k]["utility"] >= threshold)
    sel = results[k_star]

    return {
        "case_id": case_id,
        "K_star": k_star,
        "prototypes": sel["prototypes"].astype(np.float32),
        "assignments": sel["assignments"].astype(np.int32),
        "cluster_sizes": sel["cluster_sizes"],
        "support": float(sel["support"]),
        "coverage": float(sel["coverage"]),
        "utility": float(sel["utility"]),
        "q_proto_base": _clip01(sel["utility"]),
        "n_patches": n,
    }
