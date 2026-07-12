"""Compute the pooled mean patch count for a cohort, and suggest a fixed K.

For fixed-K prototype reranking, K must be chosen once for the *whole* cohort
(not per organ/subtype) — computing it per organ would give each organ its own
K, which is not meaningful when candidates from different organs are ranked
against each other in the same retrieval pool. This script pools patch counts
across every case in --patch-dir and reports one recommended K for the cohort,
following the same mean-patch-count heuristic used for AI4SKIN/SICAP/ASSIST
(see README "Choosing K").

Usage:
    python scripts/compute_mean_patch_count.py --patch-dir data/tcga/patches/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def suggest_k(mean_patches: float) -> int:
    if mean_patches < 50:
        return 2
    if mean_patches <= 200:
        return 6  # midpoint of the 4-6 recommended range
    return 8


def main() -> None:
    p = argparse.ArgumentParser(description="Pooled mean patch count and fixed-K suggestion.")
    p.add_argument("--patch-dir", required=True, type=Path)
    p.add_argument("--meta", type=Path, default=None,
                   help="Optional metadata CSV with case_id + a group column, "
                        "for a per-group breakdown printed alongside the pooled result.")
    p.add_argument("--id-col", default="case_id")
    p.add_argument("--group-col", default="organ",
                   help="Column to break down by, for diagnostics only — the fixed K "
                        "recommendation always uses the pooled cohort, never per-group.")
    args = p.parse_args()

    files = sorted(args.patch_dir.glob("*.npy"))
    if not files:
        raise SystemExit(f"No .npy files found in {args.patch_dir}")

    counts = {f.stem: int(np.load(f, mmap_mode="r").shape[0]) for f in files}
    pooled_mean = float(np.mean(list(counts.values())))
    pooled_median = float(np.median(list(counts.values())))

    print(f"Cases found          : {len(counts)}")
    print(f"Pooled mean patches  : {pooled_mean:.1f}")
    print(f"Pooled median patches: {pooled_median:.1f}")
    print(f"Recommended fixed K  : {suggest_k(pooled_mean)}  (pooled cohort-wide, not per organ)")

    if args.meta is not None:
        import pandas as pd
        meta = pd.read_csv(args.meta)
        meta["_n_patches"] = meta[args.id_col].astype(str).map(counts)
        print(f"\nPer-{args.group_col} breakdown (diagnostic only — do not use these to pick K):")
        breakdown = meta.groupby(args.group_col)["_n_patches"].agg(["count", "mean", "median"])
        print(breakdown.round(1).to_string())


if __name__ == "__main__":
    main()
