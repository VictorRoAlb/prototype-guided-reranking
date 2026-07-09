"""Adaptive prototype reranking.

K* is selected per-case from a fixed grid {2, 4, 6, 8, 12} using a utility
criterion (coverage x support): K* is the smallest grid value whose utility is
within 97% of the best utility across the grid. Prototypes are then refined
into softmax-weighted centroids (τ=0.05). The confidence c_i blends a fixed
baseline Q_FIXED=0.80 with the per-case prototype quality u(K*):

    c_i  = (1 - ρ) · Q_FIXED  +  ρ · u(K*)   (ρ = 0.50)
    score = (1 - c_i) · s_global  +  c_i · s_proto

Usage:
    python scripts/run_adaptive_reranking.py \
        --text-emb path/to/text.npy \
        --image-emb path/to/image_meanpool.npy \
        --patch-dir path/to/patches/ \
        --meta path/to/metadata.csv \
        --out results/adaptive_pgr_query_level.csv
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
from prototype_reranking.metrics import compute_all_metrics
from prototype_reranking.prototypes import l2_normalize


def main() -> None:
    p = argparse.ArgumentParser(description="Adaptive prototype reranking.")
    p.add_argument("--text-emb", required=True, type=Path)
    p.add_argument("--image-emb", required=True, type=Path)
    p.add_argument("--patch-dir", required=True, type=Path)
    p.add_argument("--meta", required=True, type=Path)
    p.add_argument("--id-col", default="case_id")
    p.add_argument("--label-col", default="label")
    p.add_argument("--k-grid", default="2,4,6,8,12",
                   help="Comma-separated candidate K values to explore for K* selection")
    p.add_argument("--rerank-top-n", type=int, default=50)
    p.add_argument("--out", type=Path, default=Path("results/adaptive_pgr_query_level.csv"))
    p.add_argument("--dataset", default="")
    p.add_argument("--model", default="")
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

    k_grid = tuple(int(k.strip()) for k in args.k_grid.split(",") if k.strip())
    print(f"Building adaptive bank (k_grid={k_grid}) for "
          f"{len(patch_vectors)}/{len(case_ids)} cases...")
    bank = build_bank(patch_vectors, case_ids, k_grid=k_grid)

    global_i2t = (image @ text.T).astype(np.float32)
    global_t2i = global_i2t.T.astype(np.float32)
    s_i2t = score_matrix_adaptive(text, case_ids, bank, global_i2t,
                                  rerank_top_n=args.rerank_top_n)
    s_t2i = score_matrix_adaptive(image, case_ids, bank, global_t2i,
                                  rerank_top_n=args.rerank_top_n)

    df = evaluate_both_directions(
        text, image, case_ids, labels,
        method="adaptive_PGR", dataset=args.dataset, model=args.model,
        score_matrix_i2t=s_i2t, score_matrix_t2i=s_t2i,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")

    metrics = pd.DataFrame(compute_all_metrics(
        df, dataset=args.dataset, model=args.model, method="adaptive_PGR"
    ))
    print(metrics[metrics.metric.isin(["MacroRecall", "MacroMRR"])].pivot_table(
        index="direction", columns=["metric", "K"], values="value"
    ).round(3).to_string())


if __name__ == "__main__":
    main()
