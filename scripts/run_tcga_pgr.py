"""Apply fixed + adaptive PGR to TCGA embeddings for all four patch-level models.

This mirrors run_full_evaluation.py but loops over the four models in one call
and picks the fixed K automatically from the pooled (cohort-wide, not
per-organ) mean patch count instead of taking it as a manual --fixed-k.

Fill in MODEL_PATHS below once the TCGA embeddings are exported, then run:

    python scripts/run_tcga_pgr.py --out-dir results/tcga --n-jobs -1

Nothing about the method itself changes for TCGA: same build_fixed_prototypes /
build_adaptive_entry / build_weighted_entry as every other dataset. The only
things specific to this script are (a) picking one fixed K for the whole
pooled cohort and (b) parallelizing the per-case bank construction with
joblib, via the n_jobs argument already added to build_fixed_bank/build_bank.
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

# ---------------------------------------------------------------------------
# TODO: fill in the four paths once the TCGA embeddings/patches are exported.
# Each model needs: text embeddings, mean-pooled image embeddings, a directory
# of per-case patch matrices ("<case_id>.npy"), and a metadata CSV with at
# least a case_id column (a label column is optional — see --label-col).
# ---------------------------------------------------------------------------
MODEL_PATHS: dict[str, dict[str, Path | None]] = {
    "KEEP": {
        "text_emb": None,     # e.g. Path("EMBEDDINGS/KEEP/TCGA/text.npy")
        "image_emb": None,    # e.g. Path("EMBEDDINGS/KEEP/TCGA/image_meanpool.npy")
        "patch_dir": None,    # e.g. Path("EMBEDDINGS/KEEP/TCGA/patches")
        "meta": None,         # e.g. Path("TCGA/tcga_metadata.csv")
    },
    "CONCH": {
        "text_emb": None,
        "image_emb": None,
        "patch_dir": None,
        "meta": None,
    },
    "MUSK": {
        "text_emb": None,
        "image_emb": None,
        "patch_dir": None,
        "meta": None,
    },
    "PATHO_CLIP": {
        "text_emb": None,
        "image_emb": None,
        "patch_dir": None,
        "meta": None,
    },
}


def suggest_k(mean_patches: float) -> int:
    if mean_patches < 50:
        return 2
    if mean_patches <= 200:
        return 6
    return 8


def pooled_mean_patch_count(patch_dir: Path, case_ids: list[str]) -> float:
    counts = []
    for cid in case_ids:
        f = patch_dir / f"{cid}.npy"
        if f.exists():
            counts.append(int(np.load(f, mmap_mode="r").shape[0]))
    if not counts:
        raise RuntimeError(f"No patch files found under {patch_dir}")
    return float(np.mean(counts))


def run_one_model(
    model: str,
    paths: dict[str, Path],
    *,
    id_col: str,
    label_col: str,
    k_grid: tuple[int, ...],
    rerank_top_n: int,
    n_jobs: int,
    fixed_k_override: int | None,
    out_dir: Path,
) -> None:
    meta = pd.read_csv(paths["meta"])
    case_ids = meta[id_col].astype(str).tolist()
    has_labels = label_col in meta.columns
    labels = meta[label_col].astype(str).tolist() if has_labels else ["_"] * len(case_ids)

    text = l2_normalize(np.load(paths["text_emb"]).astype(np.float32))
    image = l2_normalize(np.load(paths["image_emb"]).astype(np.float32))

    patch_vectors = {}
    for cid in case_ids:
        f = paths["patch_dir"] / f"{cid}.npy"
        if f.exists():
            patch_vectors[cid] = l2_normalize(np.load(f).astype(np.float32))
    print(f"[{model}] patches loaded: {len(patch_vectors)}/{len(case_ids)}")

    fixed_k = fixed_k_override
    if fixed_k is None:
        mean_patches = pooled_mean_patch_count(paths["patch_dir"], case_ids)
        fixed_k = suggest_k(mean_patches)
        print(f"[{model}] pooled mean patch count = {mean_patches:.1f} -> fixed K = {fixed_k}")

    global_i2t = (image @ text.T).astype(np.float32)
    global_t2i = global_i2t.T.astype(np.float32)

    all_rows, all_metrics = [], []

    df = evaluate_both_directions(text, image, case_ids, labels, method="mean_pooling",
                                   dataset="TCGA", model=model)
    all_rows.append(df)
    if has_labels:
        all_metrics += compute_all_metrics(df, dataset="TCGA", model=model, method="mean_pooling")

    fixed_bank = build_fixed_bank(patch_vectors, case_ids, K=fixed_k, n_jobs=n_jobs)
    s_i = score_matrix_fixed(text, case_ids, fixed_bank, global_i2t, rerank_top_n=rerank_top_n)
    s_t = score_matrix_fixed(image, case_ids, fixed_bank, global_t2i, rerank_top_n=rerank_top_n)
    df = evaluate_both_directions(text, image, case_ids, labels, method="fixed_PGR",
                                   dataset="TCGA", model=model,
                                   score_matrix_i2t=s_i, score_matrix_t2i=s_t)
    all_rows.append(df)
    if has_labels:
        all_metrics += compute_all_metrics(df, dataset="TCGA", model=model, method="fixed_PGR")
    print(f"[{model}] fixed_PGR K={fixed_k} done")

    adaptive_bank = build_bank(patch_vectors, case_ids, k_grid=k_grid, n_jobs=n_jobs)
    s_i = score_matrix_adaptive(text, case_ids, adaptive_bank, global_i2t, rerank_top_n=rerank_top_n)
    s_t = score_matrix_adaptive(image, case_ids, adaptive_bank, global_t2i, rerank_top_n=rerank_top_n)
    df = evaluate_both_directions(text, image, case_ids, labels, method="adaptive_PGR",
                                   dataset="TCGA", model=model,
                                   score_matrix_i2t=s_i, score_matrix_t2i=s_t)
    all_rows.append(df)
    if has_labels:
        all_metrics += compute_all_metrics(df, dataset="TCGA", model=model, method="adaptive_PGR")
    print(f"[{model}] adaptive_PGR k_grid={k_grid} done")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(all_rows, ignore_index=True).to_csv(out_dir / f"{model}_query_level.csv", index=False)
    if all_metrics:
        pd.DataFrame(all_metrics).to_csv(out_dir / f"{model}_macro_metrics.csv", index=False)
    else:
        print(f"[{model}] no '{label_col}' column in metadata -> skipped MacroRecall/MacroMRR "
              f"(ranked results were still saved; add a label column to get metrics).")


def main() -> None:
    p = argparse.ArgumentParser(description="Apply fixed + adaptive PGR to TCGA for all 4 models.")
    p.add_argument("--models", default="KEEP,CONCH,MUSK,PATHO_CLIP",
                   help="Comma-separated subset of MODEL_PATHS to run.")
    p.add_argument("--id-col", default="case_id")
    p.add_argument("--label-col", default="organ")
    p.add_argument("--k-grid", default="2,4,6,8,12")
    p.add_argument("--rerank-top-n", type=int, default=50)
    p.add_argument("--fixed-k", type=int, default=None,
                   help="Override the auto-computed pooled fixed K.")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Parallel workers for prototype construction (joblib). -1 = all cores.")
    p.add_argument("--out-dir", type=Path, default=Path("results/tcga"))
    args = p.parse_args()

    k_grid = tuple(int(k.strip()) for k in args.k_grid.split(",") if k.strip())
    requested = [m.strip() for m in args.models.split(",") if m.strip()]

    for model in requested:
        paths = MODEL_PATHS.get(model)
        if paths is None:
            print(f"[{model}] unknown model, skipping")
            continue
        missing = [k for k, v in paths.items() if v is None]
        if missing:
            print(f"[{model}] SKIPPED — fill in MODEL_PATHS['{model}'][{missing}] at the top "
                  f"of this script before running.")
            continue
        run_one_model(
            model, paths, id_col=args.id_col, label_col=args.label_col,
            k_grid=k_grid, rerank_top_n=args.rerank_top_n, n_jobs=args.n_jobs,
            fixed_k_override=args.fixed_k, out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
