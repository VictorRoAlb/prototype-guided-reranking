"""Baseline cross-modal retrieval: global cosine similarity (mean-pool image vs text).

Reads precomputed .npy embedding files and a CSV metadata file.

Usage:
    python scripts/run_baseline.py \
        --text-emb path/to/text.npy \
        --image-emb path/to/image_meanpool.npy \
        --meta path/to/metadata.csv \
        --id-col case_id \
        --label-col label \
        --out results/baseline_query_level.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prototype_reranking.evaluation import evaluate_both_directions
from prototype_reranking.metrics import compute_all_metrics
from prototype_reranking.prototypes import l2_normalize


def main() -> None:
    p = argparse.ArgumentParser(description="Baseline cross-modal retrieval.")
    p.add_argument("--text-emb", required=True, type=Path)
    p.add_argument("--image-emb", required=True, type=Path)
    p.add_argument("--meta", required=True, type=Path, help="CSV with id and label columns")
    p.add_argument("--id-col", default="case_id")
    p.add_argument("--label-col", default="label")
    p.add_argument("--out", type=Path, default=Path("results/baseline_query_level.csv"))
    p.add_argument("--dataset", default="")
    p.add_argument("--model", default="")
    args = p.parse_args()

    meta = pd.read_csv(args.meta)
    case_ids = meta[args.id_col].astype(str).tolist()
    labels = meta[args.label_col].astype(str).tolist()
    text = l2_normalize(np.load(args.text_emb).astype(np.float32))
    image = l2_normalize(np.load(args.image_emb).astype(np.float32))

    df = evaluate_both_directions(
        text, image, case_ids, labels,
        method="mean_pooling", dataset=args.dataset, model=args.model,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")

    metrics = pd.DataFrame(compute_all_metrics(
        df, dataset=args.dataset, model=args.model, method="mean_pooling"
    ))
    print(metrics[metrics.metric.isin(["MacroRecall", "MacroMRR"])].pivot_table(
        index="direction", columns=["metric", "K"], values="value"
    ).round(3).to_string())


if __name__ == "__main__":
    main()
