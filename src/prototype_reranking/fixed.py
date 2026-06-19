"""
fixed.py
========
Fixed prototype reranking.

Score combination (from paper):
    s_final = alpha_global * s_global + (1 - alpha_global) * s_proto
    s_proto  = mean of top-m cosine similarities between the text (or image)
               query and the K fixed prototypes of each candidate.

alpha_global defaults to 0.20 (q_proto = 0.80), matching the published protocol.
K is chosen per dataset based on mean patch count (recommended: AI4SKIN=8, SICAP=2).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .prototypes import build_fixed_prototypes, l2_normalize


ALPHA_GLOBAL: float = 0.20    # global similarity weight (paper value)
Q_PROTO: float = 1.0 - ALPHA_GLOBAL  # prototype weight = 0.80


def build_fixed_bank(
    patch_vectors: dict[str, np.ndarray],
    case_ids: list[str],
    K: int,
    *,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Build a fixed prototype bank for all cases.

    Parameters
    ----------
    patch_vectors : case_id -> (P, d) patch matrix (L2-normalised rows).
    case_ids : ordered list of case IDs to include.
    K : number of prototypes per case.

    Returns
    -------
    bank : case_id -> dict with keys ``prototypes`` (K, d) and ``q_proto`` (float).
    """
    bank: dict[str, dict[str, Any]] = {}
    for cid in case_ids:
        if cid not in patch_vectors:
            continue
        patches = np.asarray(patch_vectors[cid], dtype=np.float32)
        protos, assign = build_fixed_prototypes(patches, K, seed=seed)
        bank[cid] = {
            "case_id": cid,
            "prototypes": protos,
            "q_proto": Q_PROTO,
            "K_star": int(protos.shape[0]),
            "assignments": assign,
        }
    return bank


def _top_m_proto_score(
    query_vector: np.ndarray,
    candidate_prototypes: np.ndarray,
    m: int = 5,
) -> float:
    """Mean cosine similarity of the top-m query–prototype pairs."""
    sims = candidate_prototypes @ query_vector
    top_m = min(m, sims.shape[0])
    return float(np.sort(sims)[::-1][:top_m].mean())


def score_matrix_fixed(
    query_matrix: np.ndarray,
    candidate_ids: list[str],
    bank: dict[str, dict[str, Any]],
    global_scores: np.ndarray,
    *,
    top_m: int = 5,
    q_proto: float = Q_PROTO,
    rerank_top_n: int = 100,
) -> np.ndarray:
    """Compute fixed-reranking score matrix (n_queries x n_candidates).

    Parameters
    ----------
    query_matrix : (Q, d) query embeddings (text for I2T, image for T2I).
    candidate_ids : ordered list of candidate case IDs.
    bank : fixed prototype bank from ``build_fixed_bank``.
    global_scores : (Q, C) global cosine similarity matrix.
    top_m : number of top prototypes to average for the prototype score.
    q_proto : prototype weight (1 - alpha_global).
    rerank_top_n : number of top candidates per query to rerank.

    Returns
    -------
    final_scores : (Q, C) combined score matrix.
    """
    Q, C = global_scores.shape
    final = global_scores.astype(np.float32).copy()
    for q_idx in range(Q):
        order = np.argsort(-global_scores[q_idx])[:rerank_top_n]
        for rank, c_idx in enumerate(order):
            cid = candidate_ids[c_idx]
            if cid not in bank:
                continue
            protos = np.asarray(bank[cid]["prototypes"], dtype=np.float32)
            q = l2_normalize(query_matrix[q_idx : q_idx + 1])[0]
            ps = _top_m_proto_score(q, protos, m=top_m)
            gs = float(global_scores[q_idx, c_idx])
            final[q_idx, c_idx] = (1.0 - q_proto) * gs + q_proto * ps
    return final
