"""
run_fixed_reranking.py
======================
Fixed prototype reranking.

Patch embeddings are clustered into K centroids per case. The retrieval score
combines the global cosine similarity with the mean top-m prototype similarity:

    s_final = (1 - q_proto) * s_global  +  q_proto * s_proto   (q_proto = 0.80)

Usage:
    python scripts/run_fixed_reranking.py \
        --text-emb path/to/text.npy \
        --image-emb path/to/image_meanpool.npy \
        --patch-dir path/to/patches/ \
        --meta path/to/metadata.csv \
        --K 8 \
        --out results/fixed_pgr_query_level.csv
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
from prototype_reranking.fixed import build_fixed_bank, score_matrix_fixed
from prototype_reranking.metrics import compute_all_metrics
from prototype_reranking.prototypes import l2_normalize


def main() -> None:
    p = argparse.ArgumentParser(description="Fixed prototype reranking.")
    p.add_argument("--text-emb", required=True, type=Path)
    p.add_argument("--image-emb", required=True, type=Path)
    p.add_argument("--patch-dir", required=True, type=Path,
                   help="Directory with one <case_id>.npy per case (P x d patch matrix)")
    p.add_argument("--meta", required=True, type=Path)
    p.add_argument("--id-col", default="case_id")
    p.add_argument("--label-col", default="label")
    p.add_argument("--K", type=int, default=8, help="Number of fixed prototypes per case")
    p.add_argument("--top-m", type=int, default=5)
    p.add_argument("--rerank-top-n", type=int, default=100)
    p.add_argument("--out", type=Path, default=Path("results/fixed_pgr_query_level.csv"))
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

    print(f"Loaded patches for {len(patch_vectors)}/{len(case_ids)} cases. Building K={args.K} bank...")
    bank = build_fixed_bank(patch_vectors, case_ids, K=args.K)

    global_i2t = (image @ text.T).astype(np.float32)
    global_t2i = global_i2t.T.astype(np.float32)
    s_i2t = score_matrix_fixed(text, case_ids, bank, global_i2t,
                               top_m=args.top_m, rerank_top_n=args.rerank_top_n)
    s_t2i = score_matrix_fixed(image, case_ids, bank, global_t2i,
                               top_m=args.top_m, rerank_top_n=args.rerank_top_n)

    df = evaluate_both_directions(
        text, image, case_ids, labels,
        method="fixed_PGR", dataset=args.dataset, model=args.model,
        score_matrix_i2t=s_i2t, score_matrix_t2i=s_t2i,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")

    metrics = pd.DataFrame(compute_all_metrics(
        df, dataset=args.dataset, model=args.model, method="fixed_PGR"
    ))
    print(metrics[metrics.metric.isin(["MacroRecall", "MacroMRR"])].pivot_table(
        index="direction", columns=["metric", "K"], values="value"
    ).round(3).to_string())


if __name__ == "__main__":
    main()
