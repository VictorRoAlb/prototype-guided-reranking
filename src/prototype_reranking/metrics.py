"""
metrics.py
==========
Retrieval metrics for cross-modal histopathology evaluation.

All metrics use the **macro** aggregation: per-class mean, then unweighted
mean over classes. This is preferred over micro-average when classes are
imbalanced (as in most histopathology datasets).

Metrics
-------
Recall@K          : fraction of queries with a relevant item in top-K.
MacroRecall@K     : Recall@K averaged per class, then over classes.
MRR@10            : mean reciprocal rank (cutoff 10).
MacroMRR@10       : MRR@10 macro-averaged over classes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


RECALL_KS = [1, 3, 5, 10]
MRR_CUTOFF = 10


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------

def recall_at_k(ranked_labels: list[str], query_label: str, k: int) -> float:
    """1.0 if at least one of the top-k retrieved items matches the query label."""
    return float(any(lab == query_label for lab in ranked_labels[:k]))


def mrr_at_k(ranked_labels: list[str], query_label: str, k: int = 10) -> float:
    """Reciprocal rank of the first relevant item (0 if none in top-k)."""
    for i, lab in enumerate(ranked_labels[:k], start=1):
        if lab == query_label:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Macro aggregation
# ---------------------------------------------------------------------------

def macro_recall_at_k(query_df: pd.DataFrame, k: int, label_col: str = "query_label") -> float:
    classes = sorted(query_df[label_col].astype(str).unique())
    if not classes:
        return float("nan")
    return float(np.mean([
        pd.to_numeric(query_df[query_df[label_col].astype(str) == c][f"Recall@{k}"], errors="coerce").mean()
        for c in classes
    ]))


def macro_mrr(query_df: pd.DataFrame, label_col: str = "query_label") -> float:
    classes = sorted(query_df[label_col].astype(str).unique())
    if not classes:
        return float("nan")
    return float(np.mean([
        pd.to_numeric(query_df[query_df[label_col].astype(str) == c]["MRR@10"], errors="coerce").mean()
        for c in classes
    ]))


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def compute_all_metrics(
    query_df: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    method: str,
    label_col: str = "query_label",
) -> list[dict]:
    """Compute MacroRecall@K and MacroMRR@10 for each direction."""
    rows = []
    for direction in query_df["direction"].unique():
        sub = query_df[query_df["direction"] == direction]
        n_q = int(len(sub))
        n_cls = int(sub[label_col].astype(str).nunique())
        base = {"dataset": dataset, "model": model, "method": method, "direction": direction,
                "n_queries": n_q, "n_classes": n_cls}
        for kk in RECALL_KS:
            rows.append({**base, "metric": "MacroRecall", "K": kk, "value": macro_recall_at_k(sub, kk, label_col)})
        rows.append({**base, "metric": "MacroMRR", "K": MRR_CUTOFF, "value": macro_mrr(sub, label_col)})
    return rows
