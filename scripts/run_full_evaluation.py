"""
run_full_evaluation.py
======================
Run all three methods (baseline, fixed PGR, adaptive PGR) and produce a
summary CSV with MacroRecall@{1,3,5,10} and MacroMRR@10.

Usage:
    python scripts/run_full_evaluation.py \
        --text-emb path/to/text.npy \
        --image-emb path/to/image_meanpool.npy \
        --patch-dir path/to/patches/ \
        --meta path/to/metadata.csv \
        --fixed-k 8 \
        --out-dir results/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prototype_reranking.adaptive import build_bank, score_matrix_adaptive
from prototype_reranking.evaluation import evaluate_both_directions
from prototype_reranking.fixed import build_fixed_bank, score_matrix_fixed
from prototype_reranking.metrics import compute_all_metrics
from prototype_reranking.prototypes import l2_normalize


def main() -> None:
    p = argparse.ArgumentParser(description="Full three-method evaluation.")
    p.add_argument("--text-emb", required=True, type=Path)
    p.add_argument("--image-emb", required=True, type=Path)
    p.add_argument("--patch-dir", required=True, type=Path)
    p.add_argument("--meta", required=True, type=Path)
    p.add_argument("--id-col", default="case_id")
    p.add_argument("--label-col", default="label")
    p.add_argument("--fixed-k", type=int, default=8)
    p.add_argument("--k-grid", default="2,4,6,8,12",
                   help="Comma-separated candidate K values for adaptive K* selection")
    p.add_argument("--rerank-top-n", type=int, default=50)
    p.add_argument("--dataset", default="")
    p.add_argument("--model", default="")
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    args = p.parse_args()

    meta = pd.read_csv(args.meta)
    case_ids = meta[args.id_col].astype(str).tolist()
    labels = meta[args.label_col].astype(str).tolist()
    text = l2_normalize(np.load(args.text_emb).astype(np.float32))
    image = l2_normalize(np.load(args.image_emb).astype(np.float32))

    patch_vectors = {}
    for cid in case_ids:
        f = args.patch_dir / f"{cid}.npy"
        if f.exists():
            patch_vectors[cid] = l2_normalize(np.load(f).astype(np.float32))
    print(f"Patches loaded: {len(patch_vectors)}/{len(case_ids)}")

    global_i2t = (image @ text.T).astype(np.float32)
    global_t2i = global_i2t.T.astype(np.float32)

    all_rows, all_metrics = [], []

    # ── Baseline ────────────────────────────────────────────────────────────
    df = evaluate_both_directions(
        text, image, case_ids, labels,
        method="mean_pooling", dataset=args.dataset, model=args.model,
    )
    all_rows.append(df)
    all_metrics += compute_all_metrics(df, dataset=args.dataset, model=args.model, method="mean_pooling")
    print("  [mean_pooling] done")

    # ── Fixed PGR ───────────────────────────────────────────────────────────
    fixed_bank = build_fixed_bank(patch_vectors, case_ids, K=args.fixed_k)
    s_i = score_matrix_fixed(text, case_ids, fixed_bank, global_i2t,
                             rerank_top_n=args.rerank_top_n)
    s_t = score_matrix_fixed(image, case_ids, fixed_bank, global_t2i,
                             rerank_top_n=args.rerank_top_n)
    df = evaluate_both_directions(
        text, image, case_ids, labels,
        method="fixed_PGR", dataset=args.dataset, model=args.model,
        score_matrix_i2t=s_i, score_matrix_t2i=s_t,
    )
    all_rows.append(df)
    all_metrics += compute_all_metrics(df, dataset=args.dataset, model=args.model, method="fixed_PGR")
    print(f"  [fixed_PGR K={args.fixed_k}] done")

    # ── Adaptive PGR ────────────────────────────────────────────────────────
    k_grid = tuple(int(k.strip()) for k in args.k_grid.split(",") if k.strip())
    adaptive_bank = build_bank(patch_vectors, case_ids, k_grid=k_grid)
    s_i = score_matrix_adaptive(text, case_ids, adaptive_bank, global_i2t,
                                rerank_top_n=args.rerank_top_n)
    s_t = score_matrix_adaptive(image, case_ids, adaptive_bank, global_t2i,
                                rerank_top_n=args.rerank_top_n)
    df = evaluate_both_directions(
        text, image, case_ids, labels,
        method="adaptive_PGR", dataset=args.dataset, model=args.model,
        score_matrix_i2t=s_i, score_matrix_t2i=s_t,
    )
    all_rows.append(df)
    all_metrics += compute_all_metrics(df, dataset=args.dataset, model=args.model, method="adaptive_PGR")
    print(f"  [adaptive_PGR k_grid={k_grid}] done")

    # ── Save ─────────────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(all_rows, ignore_index=True).to_csv(args.out_dir / "query_level.csv", index=False)
    mdf = pd.DataFrame(all_metrics)
    mdf.to_csv(args.out_dir / "macro_metrics.csv", index=False)

    summary = mdf[mdf.metric.isin(["MacroRecall", "MacroMRR"])].pivot_table(
        index=["method", "direction"], columns=["metric", "K"], values="value"
    ).round(3)
    print("\n=== Summary ===")
    print(summary.to_string())
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
