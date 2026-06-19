"""
evaluation.py
=============
Cross-modal retrieval evaluation loop (I2T and T2I).

Provides ``evaluate_retrieval`` which takes precomputed embedding matrices and
a score matrix and produces a per-query DataFrame with Recall@K, MRR@10, and
the top-10 label string needed for MacroMAP@10.

Exact-pair exclusion
--------------------
When a query image and a candidate image are the *same case*, the candidate is
removed before ranking. This prevents trivially perfect self-retrieval from
inflating the metrics. Enable with ``exclude_exact_pair=True`` (default).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import recall_at_k, mrr_at_k


def _exclude_self(scores: np.ndarray, q_idx: int) -> np.ndarray:
    """Return a copy of ``scores`` with the self-score set to -inf."""
    out = scores.copy()
    out[q_idx] = -np.inf
    return out


def evaluate_retrieval(
    score_matrix: np.ndarray,
    case_ids: list[str],
    labels: list[str],
    *,
    direction: str,
    method: str,
    model: str = "",
    dataset: str = "",
    exclude_exact_pair: bool = True,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    mrr_cutoff: int = 10,
) -> pd.DataFrame:
    """Evaluate a retrieval score matrix and return a per-query DataFrame.

    Parameters
    ----------
    score_matrix : (N, N) pairwise scores. Higher = more similar.
        For I2T: rows = image queries, columns = text candidates.
        For T2I: rows = text queries, columns = image candidates.
    case_ids : list of N case identifiers (same ordering as rows/columns).
    labels : list of N class labels (same ordering).
    direction : ``"I2T"`` or ``"T2I"``.
    method : method name tag (for the output DataFrame).
    model, dataset : metadata tags.
    exclude_exact_pair : remove the query's own index from the ranking.
    ks : cutoff values for Recall@K.
    mrr_cutoff : cutoff for MRR.

    Returns
    -------
    pd.DataFrame with one row per query.
    """
    N = len(case_ids)
    rows: list[dict[str, Any]] = []
    for q_idx in range(N):
        scores = score_matrix[q_idx].copy()
        if exclude_exact_pair:
            scores = _exclude_self(scores, q_idx)
        ranked_indices = np.argsort(-scores)
        ranked_labels = [labels[int(i)] for i in ranked_indices]
        ranked_ids = [case_ids[int(i)] for i in ranked_indices]
        top10_labels = "|".join(ranked_labels[:10])
        top10_ids = "|".join(ranked_ids[:10])
        row: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "method": method,
            "direction": direction,
            "query_case_id": case_ids[q_idx],
            "query_label": labels[q_idx],
            "top1_case_id": ranked_ids[0] if ranked_ids else "",
            "top1_label": ranked_labels[0] if ranked_labels else "",
            "top10_ids": top10_ids,
            "top10_labels": top10_labels,
        }
        for k in ks:
            row[f"Recall@{k}"] = recall_at_k(ranked_labels, labels[q_idx], k)
        row["MRR@10"] = mrr_at_k(ranked_labels, labels[q_idx], mrr_cutoff)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_both_directions(
    text_matrix: np.ndarray,
    image_matrix: np.ndarray,
    case_ids: list[str],
    labels: list[str],
    *,
    method: str,
    model: str = "",
    dataset: str = "",
    score_matrix_i2t: np.ndarray | None = None,
    score_matrix_t2i: np.ndarray | None = None,
    exclude_exact_pair: bool = True,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """Evaluate I2T and T2I directions and concatenate results.

    If ``score_matrix_i2t`` / ``score_matrix_t2i`` are provided they are used
    directly (for custom reranking scores). Otherwise the baseline cosine
    similarity is computed from ``image_matrix @ text_matrix.T``.
    """
    if score_matrix_i2t is None:
        score_matrix_i2t = (image_matrix @ text_matrix.T).astype(np.float32)
    if score_matrix_t2i is None:
        score_matrix_t2i = score_matrix_i2t.T.astype(np.float32)

    i2t = evaluate_retrieval(
        score_matrix_i2t, case_ids, labels,
        direction="I2T", method=method, model=model, dataset=dataset,
        exclude_exact_pair=exclude_exact_pair, ks=ks,
    )
    t2i = evaluate_retrieval(
        score_matrix_t2i, case_ids, labels,
        direction="T2I", method=method, model=model, dataset=dataset,
        exclude_exact_pair=exclude_exact_pair, ks=ks,
    )
    return pd.concat([i2t, t2i], ignore_index=True)
