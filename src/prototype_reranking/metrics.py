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
MacroMAP@10       : macro mean average precision at 10.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


RECALL_KS = [1, 3, 5, 10]
MRR_CUTOFF = 10
MAP_CUTOFF = 10


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


def average_precision_at_k(
    ranked_labels: list[str], query_label: str, n_relevant: int, k: int = 10
) -> float:
    """Average precision at k for one query.

    n_relevant is the number of relevant items in the candidate pool (excluding
    the query itself when exact-pair exclusion is applied).
    """
    if n_relevant <= 0:
        return 0.0
    hits, acc = 0, 0.0
    for i, lab in enumerate(ranked_labels[:k], start=1):
        if lab == query_label:
            hits += 1
            acc += hits / i
    return float(acc / min(n_relevant, k))


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


def macro_map(query_df: pd.DataFrame, k: int = 10, label_col: str = "query_label") -> float:
    """Macro mAP@k from the top10_labels column (pipe-separated string per row)."""
    if query_df.empty or "top10_labels" not in query_df.columns:
        return float("nan")
    counts = query_df[label_col].astype(str).value_counts().to_dict()
    ap_per_query = []
    for _, row in query_df.iterrows():
        ql = str(row[label_col])
        top = [t for t in str(row.get("top10_labels", "")).split("|") if t]
        n_rel = int(counts.get(ql, 0)) - 1  # exclude exact pair
        ap_per_query.append(average_precision_at_k(top, ql, n_rel, k))
    tmp = query_df.assign(_ap=ap_per_query)
    classes = sorted(tmp[label_col].astype(str).unique())
    return float(np.mean([tmp[tmp[label_col].astype(str) == c]["_ap"].mean() for c in classes]))


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
    """Compute MacroRecall@K, MacroMRR@10, MacroMAP@10 for each direction."""
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
        rows.append({**base, "metric": "MacroMAP", "K": MAP_CUTOFF, "value": macro_map(sub, MAP_CUTOFF, label_col)})
    return rows
