"""
adaptive.py
===========
Adaptive prototype reranking.

Overview
--------
For each image, K* prototypes are selected per-case using a utility criterion
(see prototypes.build_adaptive_entry). The prototypes are then refined into
softmax-weighted centroids (temperature tau=0.05), making each centroid more
representative of the dominant patch direction within its cluster.

The confidence weight c_i is a regularised blend of a fixed baseline Q_FIXED
and the per-case prototype quality q_proto_base:

    c_i = (1 - rho) * Q_FIXED  +  rho * q_proto_base     [rho=0.50, Q_FIXED=0.80]

The final retrieval score combines global and prototype similarity:

    s_final = (1 - c_i) * s_global  +  c_i * s_proto

where s_proto = max_k cos(query, weighted_prototype_k).

Cases with low patch support or poor prototype structure automatically fall back
toward global similarity (c_i closer to Q_FIXED * (1-rho) = 0.40), while cases
with strong, well-separated prototype structure receive higher prototype weight.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .prototypes import build_adaptive_entry, l2_normalize, _clip01

Q_FIXED: float = 0.80
RHO: float = 0.50
TAU: float = 0.05


# ---------------------------------------------------------------------------
# Softmax-weighted centroid
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray, tau: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    shifted = (x - x.max()) / max(float(tau), 1e-6)
    exp = np.exp(shifted).astype(np.float32)
    s = float(exp.sum())
    return (exp / s).astype(np.float32) if s > 0 else np.full(len(x), 1.0 / max(len(x), 1))


def _weighted_centroid(members: np.ndarray, tau: float) -> np.ndarray:
    """Softmax-weighted centroid. Members should be L2-normalised."""
    if members.shape[0] == 1:
        return l2_normalize(members[0])
    mean = l2_normalize(members.mean(axis=0))
    weights = _softmax(members @ mean, tau)
    return l2_normalize(np.sum(weights[:, None] * members, axis=0))


def build_weighted_entry(
    case_id: str,
    patches: np.ndarray,
    base_entry: dict[str, Any],
    *,
    q_fixed: float = Q_FIXED,
    rho: float = RHO,
    tau: float = TAU,
) -> dict[str, Any]:
    """Build the weighted-centroid prototype entry for one case.

    Parameters
    ----------
    patches : (P, d) L2-normalised patch embeddings.
    base_entry : output of build_adaptive_entry.
    q_fixed, rho, tau : method hyperparameters.
    """
    patches = np.asarray(patches, dtype=np.float32)
    assign = np.asarray(base_entry["assignments"], dtype=np.int32)
    K_star = int(base_entry["K_star"])
    orig_protos = l2_normalize(np.asarray(base_entry["prototypes"], dtype=np.float32))

    rows: list[np.ndarray] = []
    for k in range(K_star):
        members = patches[assign == k]
        rows.append(
            _weighted_centroid(members, tau) if members.shape[0] > 0
            else orig_protos[k] if k < orig_protos.shape[0]
            else np.zeros(patches.shape[1], dtype=np.float32)
        )
    weighted_protos = l2_normalize(np.stack(rows, axis=0))

    q_base = float(base_entry.get("q_proto_base", base_entry.get("utility", 0.0)))
    c_i = _clip01((1.0 - rho) * q_fixed + rho * q_base)

    return {
        "case_id": case_id,
        "K_star": K_star,
        "prototypes": weighted_protos.astype(np.float32),
        "q_proto": c_i,
        "q_proto_base": q_base,
    }


def build_bank(
    patch_vectors: dict[str, np.ndarray],
    case_ids: list[str],
    *,
    k_grid: tuple[int, ...] = (2, 4, 6, 8, 12),
    min_support: int = 6,
    good_support: int = 20,
    near_best_ratio: float = 0.97,
    q_fixed: float = Q_FIXED,
    rho: float = RHO,
    tau: float = TAU,
    seed: int = 42,
    prebuilt_entries: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the adaptive prototype bank for all cases.

    If prebuilt_entries is provided (e.g. loaded from disk), those base entries
    are reused and the softmax-weighting is applied on top.

    Returns
    -------
    bank : case_id -> dict with prototypes (K*, d) and q_proto (float c_i).
    """
    base: dict[str, dict[str, Any]] = dict(prebuilt_entries or {})
    bank: dict[str, dict[str, Any]] = {}
    for cid in case_ids:
        if cid not in patch_vectors:
            continue
        patches = np.asarray(patch_vectors[cid], dtype=np.float32)
        if cid not in base:
            base[cid] = build_adaptive_entry(
                cid, patches, k_grid=k_grid, min_support=min_support,
                good_support=good_support, near_best_ratio=near_best_ratio, seed=seed,
            )
        bank[cid] = build_weighted_entry(
            cid, patches, base[cid], q_fixed=q_fixed, rho=rho, tau=tau
        )
    return bank


# ---------------------------------------------------------------------------
# Score matrix
# ---------------------------------------------------------------------------

def score_matrix_adaptive(
    query_matrix: np.ndarray,
    candidate_ids: list[str],
    bank: dict[str, dict[str, Any]],
    global_scores: np.ndarray,
    *,
    rerank_top_n: int = 100,
) -> np.ndarray:
    """Compute the adaptive-reranking score matrix (Q x C).

    Prototype score = max cosine similarity between query and the K* weighted
    prototypes of each candidate (consistent with the fixed reranking mode).

    Parameters
    ----------
    query_matrix : (Q, d) L2-normalised query embeddings.
    candidate_ids : ordered list of C candidate case IDs.
    bank : adaptive bank from build_bank.
    global_scores : (Q, C) global cosine-similarity matrix.
    rerank_top_n : candidates per query to rerank.
    """
    qmat = l2_normalize(np.asarray(query_matrix, dtype=np.float32))
    final = global_scores.astype(np.float32).copy()
    for q_idx in range(global_scores.shape[0]):
        top_n = np.argsort(-global_scores[q_idx])[:rerank_top_n]
        for c_idx in top_n:
            cid = candidate_ids[c_idx]
            if cid not in bank:
                continue
            entry = bank[cid]
            protos = np.asarray(entry["prototypes"], dtype=np.float32)
            ci = float(entry["q_proto"])
            s_proto = float(np.max(protos @ qmat[q_idx]))
            s_global = float(global_scores[q_idx, c_idx])
            final[q_idx, c_idx] = (1.0 - ci) * s_global + ci * s_proto
    return final
