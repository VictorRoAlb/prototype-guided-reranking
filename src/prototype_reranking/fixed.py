"""Fixed prototype reranking.

Prototype score (per candidate):
    s_proto = max_k  cos(query, prototype_k)          (max over K prototypes)

Final score:
    s_final = (1 - q_proto) * s_global  +  q_proto * s_proto
    q_proto = 0.80  (prototype weight, i.e. alpha_global = 0.20)

Reranking is applied to the top-rerank_top_n candidates per query; the rest
keep their global-similarity order.
"""
from __future__ import annotations
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from .prototypes import build_fixed_prototypes, l2_normalize

Q_PROTO: float = 0.80       # prototype weight  (= 1 - alpha_global)


def _build_fixed_entry(cid: str, patches: np.ndarray, K: int, seed: int) -> dict[str, Any]:
    protos, assign = build_fixed_prototypes(np.asarray(patches, dtype=np.float32), K, seed=seed)
    return {
        "case_id": cid,
        "prototypes": protos,
        "q_proto": Q_PROTO,
        "K_star": int(protos.shape[0]),
        "assignments": assign,
    }


def build_fixed_bank(
    patch_vectors: dict[str, np.ndarray],
    case_ids: list[str],
    K: int,
    *,
    seed: int = 42,
    n_jobs: int = 1,
) -> dict[str, dict[str, Any]]:
    """Build a fixed prototype bank for all cases.

    Parameters
    ----------
    patch_vectors : case_id -> (P, d) L2-normalised patch matrix.
    case_ids : ordered list of case IDs to include.
    K : number of prototypes per case.
    n_jobs : number of worker processes (joblib). Each case is clustered
        independently with the same fixed ``seed``, so results are identical
        to the sequential (``n_jobs=1``) run regardless of worker count or
        scheduling order — this only changes wall-clock time, never the
        prototypes/scores. Use -1 for all available cores.

    Returns
    -------
    bank : case_id -> dict with keys prototypes (K, d) and q_proto (float).
    """
    usable = [cid for cid in case_ids if cid in patch_vectors]
    entries = Parallel(n_jobs=n_jobs)(
        delayed(_build_fixed_entry)(cid, patch_vectors[cid], K, seed) for cid in usable
    )
    return {entry["case_id"]: entry for entry in entries}


def score_matrix_fixed(
    query_matrix: np.ndarray,
    candidate_ids: list[str],
    bank: dict[str, dict[str, Any]],
    global_scores: np.ndarray,
    *,
    rerank_top_n: int = 50,
) -> np.ndarray:
    """Compute the fixed-reranking score matrix (Q x C).

    Parameters
    ----------
    query_matrix : (Q, d) L2-normalised query embeddings.
    candidate_ids : ordered list of C candidate case IDs.
    bank : fixed prototype bank from build_fixed_bank.
    global_scores : (Q, C) global cosine-similarity matrix.
    rerank_top_n : number of top candidates per query to rerank.

    Returns
    -------
    final_scores : (Q, C) combined score matrix.
    """
    qmat = l2_normalize(np.asarray(query_matrix, dtype=np.float32))
    final = global_scores.astype(np.float32).copy()
    for q_idx in range(global_scores.shape[0]):
        top_n = np.argsort(-global_scores[q_idx])[:rerank_top_n]
        for c_idx in top_n:
            cid = candidate_ids[c_idx]
            if cid not in bank:
                continue
            protos = np.asarray(bank[cid]["prototypes"], dtype=np.float32)
            ci = float(bank[cid]["q_proto"])
            s_proto = float(np.max(protos @ qmat[q_idx]))
            s_global = float(global_scores[q_idx, c_idx])
            final[q_idx, c_idx] = (1.0 - ci) * s_global + ci * s_proto
    return final
