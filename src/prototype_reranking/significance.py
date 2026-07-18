"""Paired bootstrap significance testing for retrieval metric comparisons.

Reproduces the statistical procedure used to report significance in the
thesis this repository accompanies: a class-stratified paired bootstrap over
queries, a **bilateral** (two-sided) p-value, and Holm-Bonferroni correction
within a family of related hypotheses. See docs/method_overview.md for the
full description and the exact defaults (B=10,000).

The bilateral p-value is estimated from the bootstrap distribution of the
paired delta as:

    p = min(1, 2 * min(P(delta <= 0), P(delta >= 0)))

A single two-sided test already tells you whether the observed difference is
distinguishable from zero; the *direction* (better/worse) comes from the sign
of the observed point-estimate delta, not from running two separate one-sided
tests. Doing two one-sided tests at alpha=0.05 each is NOT equivalent to one
bilateral test at alpha=0.05 -- it is roughly as permissive as a bilateral
test at alpha=0.10, since a proper two-sided test at alpha=0.05 requires each
tail to fall below alpha/2=0.025.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def paired_bootstrap_delta(
    query_df: pd.DataFrame,
    metric_col: str,
    method_a: str,
    method_b: str,
    *,
    label_col: str = "query_label",
    method_col: str = "method",
    id_col: str = "query_case_id",
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict:
    """Bilateral paired bootstrap comparison of method_a vs. method_b on a
    per-query metric column, macro-averaged over classes in `label_col`.

    Resampling is class-stratified (queries are resampled with replacement
    *within* each class, preserving the macro-averaging structure) and
    paired (the same resampled indices are applied to both methods in every
    replicate), matching the query-level unit of analysis described in the
    thesis for per-model/per-cohort/per-direction comparisons.

    Returns a dict with the observed values for both methods, the observed
    delta, the 95% bootstrap CI of the delta, and the raw bilateral p-value
    (NOT Holm-adjusted -- see `holm_bonferroni` to correct a family of these).
    """
    rng = np.random.default_rng(seed)
    classes = sorted(query_df[label_col].astype(str).unique())

    per_method: dict[str, dict[str, np.ndarray]] = {}
    ids_by_class: dict[str, list[str]] = {}
    for cls in classes:
        cls_ids = sorted(
            query_df[query_df[label_col].astype(str) == cls][id_col].astype(str).unique()
        )
        ids_by_class[cls] = cls_ids
        for method in (method_a, method_b):
            sub = query_df[(query_df[method_col] == method) & (query_df[label_col].astype(str) == cls)]
            indexed = sub.set_index(sub[id_col].astype(str))
            per_method.setdefault(method, {})[cls] = indexed.reindex(cls_ids)[metric_col].to_numpy(dtype=float)

    def macro(method: str, sample_idx: dict[str, np.ndarray] | None = None) -> float:
        vals = []
        for cls in classes:
            arr = per_method[method][cls]
            if sample_idx is not None:
                arr = arr[sample_idx[cls]]
            vals.append(np.nanmean(arr) if len(arr) else np.nan)
        return float(np.nanmean(vals))

    obs_a, obs_b = macro(method_a), macro(method_b)

    boot_a = np.empty(n_boot)
    boot_b = np.empty(n_boot)
    for b in range(n_boot):
        sample_idx = {
            cls: rng.integers(0, len(ids_by_class[cls]), size=len(ids_by_class[cls]))
            for cls in classes
        }
        boot_a[b] = macro(method_a, sample_idx)
        boot_b[b] = macro(method_b, sample_idx)

    delta_boot = boot_a - boot_b
    ci_lo, ci_hi = np.percentile(delta_boot, [2.5, 97.5])
    raw_p = min(1.0, 2.0 * min(float(np.mean(delta_boot <= 0)), float(np.mean(delta_boot >= 0))))

    return {
        "method_a": method_a,
        "method_b": method_b,
        "n_boot": n_boot,
        "observed_a": obs_a,
        "observed_b": obs_b,
        "observed_delta": obs_a - obs_b,
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "raw_p_bilateral": raw_p,
    }


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Standard Holm-Bonferroni step-down correction.

    Returns adjusted p-values in the SAME order as the input list. Apply this
    to the raw p-values of every comparison that belongs to one family of
    hypotheses (e.g. all models x both PGR variants for a fixed metric,
    cohort, and retrieval direction) before deciding significance.
    """
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p_values[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def significance_symbol(holm_p: float, observed_delta: float, alpha: float = 0.05) -> str:
    """'*' if significantly better, '†' (dagger) if significantly worse,
    '' if not significant -- decided from the (already Holm-adjusted) p-value
    and the SIGN of the observed delta, not from a second one-sided test."""
    if holm_p >= alpha:
        return ""
    return "*" if observed_delta > 0 else "†"
