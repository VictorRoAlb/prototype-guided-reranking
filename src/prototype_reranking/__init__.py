"""Training-free prototype-guided reranking for cross-modal histopathology retrieval.

Quick start:
>>> from prototype_reranking.prototypes import build_fixed_prototypes, build_adaptive_entry
>>> from prototype_reranking.fixed import build_fixed_bank, score_matrix_fixed
>>> from prototype_reranking.adaptive import build_bank, score_matrix_adaptive
>>> from prototype_reranking.evaluation import evaluate_both_directions
>>> from prototype_reranking.metrics import compute_all_metrics
"""
from .evaluation import evaluate_both_directions, evaluate_retrieval
from .metrics import compute_all_metrics, macro_mrr, macro_recall_at_k
from .prototypes import build_adaptive_entry, build_fixed_prototypes, l2_normalize
from .significance import holm_bonferroni, paired_bootstrap_delta, significance_symbol

__all__ = [
    "l2_normalize",
    "build_fixed_prototypes",
    "build_adaptive_entry",
    "evaluate_retrieval",
    "evaluate_both_directions",
    "macro_recall_at_k",
    "macro_mrr",
    "compute_all_metrics",
    "paired_bootstrap_delta",
    "holm_bonferroni",
    "significance_symbol",
]
