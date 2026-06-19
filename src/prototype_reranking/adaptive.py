"""
adaptive.py
===========
Adaptive prototype reranking.

Method summary
--------------
Each image is represented by an *adaptive* set of K* weighted prototypes,
where K* is selected per-case (e.g. by silhouette) and the prototypes are
*weighted centroids* (softmax-weighted toward the cluster mean, τ=0.05).

The confidence score c_i blends a fixed baseline quality Q_FIXED with the
per-case prototype quality u(K*):

    c_i = (1 - ρ) · Q_FIXED  +  ρ · u(K*)        (ρ = 0.50 by default)

Final retrieval score:

    s_final = (1 - c_i) · s_global  +  c_i · s_proto

where s_proto is the mean top-m cosine similarity between the query and the
candidate's weighted prototypes.

Parameters
----------
Q_FIXED : float = 0.80   (fixed baseline proto weight, matching paper)
RHO     : float = 0.50   (blend factor between fixed and per-case quality)
TAU     : float = 0.05   (softmax temperature for weighted centroid)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .prototypes import build_adaptive_entry, l2_normalize

Q_FIXED: float = 0.80
RHO: float = 0.50
TAU: float = 0.05


# ---------------------------------------------------------------------------
# Weighted prototype construction
# ---------------------------------------------------------------------------

def _softmax(values: np.ndarray, tau: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    shifted = arr - float(np.max(arr))
    exp = np.exp(shifted / max(float(tau), 1e-6)).astype(np.float32)
    total = float(exp.sum())
    if not math.isfinite(total) or total <= 0.0:
        return np.full(arr.shape[0], 1.0 / max(arr.shape[0], 1), dtype=np.float32)
    return (exp / total).astype(np.float32)


def _weighted_centroid(members: np.ndarray, tau: float) -> np.ndarray:
    """Softmax-weighted centroid of cluster members (τ controls peakedness)."""
    members = np.asarray(members, dtype=np.float32)
    if members.shape[0] == 1:
        v = members[0]
    else:
        mean = l2_normalize(members.mean(axis=0, keepdims=True))[0]
        sims = members @ mean
        weights = _softmax(sims, tau)
        v = np.sum(weights[:, None] * members, axis=0)
    return l2_normalize(v.reshape(1, -1))[0]


def _compute_balance(cluster_sizes: np.ndarray) -> float:
    sizes = np.asarray(cluster_sizes, dtype=np.float32)
    sizes = sizes[sizes > 0]
    if sizes.size <= 1:
        return 1.0
    probs = sizes / float(sizes.sum())
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
    return float(np.clip(entropy / math.log(float(sizes.size)), 0.0, 1.0))


def _compute_separation(prototypes: np.ndarray) -> float:
    if prototypes.shape[0] <= 1:
        return 1.0
    norm = l2_normalize(prototypes)
    sim = norm @ norm.T
    mask = ~np.eye(sim.shape[0], dtype=bool)
    return float(np.clip(1.0 - float(np.mean(sim[mask])), 0.0, 1.0))


def build_weighted_entry(
    case_id: str,
    patches: np.ndarray,
    adaptive_entry: dict[str, Any],
    *,
    q_fixed: float = Q_FIXED,
    rho: float = RHO,
    tau: float = TAU,
) -> dict[str, Any]:
    """Build the weighted prototype entry for one case.

    Parameters
    ----------
    case_id : str
    patches : (P, d) L2-normalised patch embeddings.
    adaptive_entry : output of ``build_adaptive_entry`` (or compatible dict).
    q_fixed, rho, tau : method hyperparameters.
    """
    patches = np.asarray(patches, dtype=np.float32)
    assignments = np.asarray(adaptive_entry["assignments"], dtype=np.int32)
    K_star = int(adaptive_entry["K_star"])
    cluster_sizes = np.bincount(assignments, minlength=K_star).astype(np.int32)

    weighted_rows: list[np.ndarray] = []
    orig_protos = l2_normalize(np.asarray(adaptive_entry["prototypes"], dtype=np.float32))
    for k in range(K_star):
        members = patches[assignments == k]
        if members.shape[0] == 0:
            weighted_rows.append(orig_protos[k] if k < orig_protos.shape[0] else np.zeros(patches.shape[1]))
        else:
            weighted_rows.append(_weighted_centroid(members, tau))
    weighted_protos = l2_normalize(np.stack(weighted_rows, axis=0))

    support = float(np.count_nonzero(cluster_sizes)) / max(K_star, 1)
    balance = _compute_balance(cluster_sizes)
    separation = _compute_separation(weighted_protos)
    cov_w = float(np.max(patches @ weighted_protos.T, axis=1).mean()) if patches.size else math.nan

    q_original = float(np.clip(float(adaptive_entry.get("q_proto_original", 0.0)), 0.0, 1.0))
    q_final = float(np.clip((1.0 - rho) * q_fixed + rho * q_original, 0.0, 1.0))
    utility = float(cov_w * support * balance * separation) if math.isfinite(cov_w) else 0.0

    return {
        "case_id": case_id,
        "K_star": K_star,
        "prototypes": weighted_protos.astype(np.float32),
        "q_proto": q_final,
        "cluster_sizes": cluster_sizes,
        "support": support,
        "balance": balance,
        "separation": separation,
        "coverage": cov_w,
        "utility": utility,
    }


def build_bank(
    patch_vectors: dict[str, np.ndarray],
    case_ids: list[str],
    *,
    K_range: tuple[int, int] = (1, 12),
    q_fixed: float = Q_FIXED,
    rho: float = RHO,
    tau: float = TAU,
    seed: int = 42,
    prebuilt_entries: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the adaptive prototype bank for all cases.

    If ``prebuilt_entries`` is provided (e.g. loaded from disk), those entries
    are used directly and K-selection is skipped for the matched case IDs.

    Parameters
    ----------
    patch_vectors : case_id -> (P, d) L2-normalised patch matrix.
    case_ids : ordered list of case IDs.
    K_range : (min_K, max_K) for adaptive K selection.
    q_fixed, rho, tau : method hyperparameters (see module docstring).
    seed : KMeans random state.
    prebuilt_entries : optional pre-computed adaptive entries.

    Returns
    -------
    bank : case_id -> weighted prototype entry dict.
    """
    base_entries: dict[str, dict[str, Any]] = dict(prebuilt_entries or {})
    bank: dict[str, dict[str, Any]] = {}
    for cid in case_ids:
        if cid not in patch_vectors:
            continue
        patches = np.asarray(patch_vectors[cid], dtype=np.float32)
        if cid not in base_entries:
            base_entries[cid] = build_adaptive_entry(cid, patches, K_range=K_range, seed=seed)
        bank[cid] = build_weighted_entry(cid, patches, base_entries[cid], q_fixed=q_fixed, rho=rho, tau=tau)
    return bank


# ---------------------------------------------------------------------------
# Score matrix
# ---------------------------------------------------------------------------

def _top_m_mean(sims: np.ndarray, m: int) -> float:
    top_m = min(m, sims.shape[0])
    return float(np.sort(sims)[::-1][:top_m].mean())


def score_matrix_adaptive(
    query_matrix: np.ndarray,
    candidate_ids: list[str],
    bank: dict[str, dict[str, Any]],
    global_scores: np.ndarray,
    *,
    top_m: int = 5,
    rerank_top_n: int = 100,
) -> np.ndarray:
    """Compute adaptive-reranking score matrix (Q x C).

    Parameters
    ----------
    query_matrix : (Q, d) query embeddings.
    candidate_ids : ordered list of candidate case IDs.
    bank : adaptive prototype bank from ``build_bank``.
    global_scores : (Q, C) global cosine similarity matrix.
    top_m : number of top prototype similarities to average.
    rerank_top_n : candidates per query to rerank.

    Returns
    -------
    final_scores : (Q, C) combined score matrix.
    """
    final = global_scores.astype(np.float32).copy()
    qmat = l2_normalize(np.asarray(query_matrix, dtype=np.float32))
    for q_idx in range(global_scores.shape[0]):
        order = np.argsort(-global_scores[q_idx])[:rerank_top_n]
        for c_idx in order:
            cid = candidate_ids[c_idx]
            if cid not in bank:
                continue
            entry = bank[cid]
            protos = np.asarray(entry["prototypes"], dtype=np.float32)
            ci = float(entry["q_proto"])
            ps = _top_m_mean(protos @ qmat[q_idx], top_m)
            gs = float(global_scores[q_idx, c_idx])
            final[q_idx, c_idx] = (1.0 - ci) * gs + ci * ps
    return final
