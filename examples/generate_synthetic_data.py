"""
generate_synthetic_data.py
===========================
Generates a fully synthetic dataset with no relation to any real dataset, used
to smoke-test the scripts in this repository (see docs/data_privacy.md).

Each of n_classes classes gets a random Gaussian centroid in R^d. Cases are
sampled around their class centroid (text and image embeddings share the same
centroid so a learnable image-text relationship exists), and each case gets a
variable number of patch embeddings sampled around the image embedding.

Usage:
    python examples/generate_synthetic_data.py --out-dir examples/data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prototype_reranking.prototypes import l2_normalize


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic embeddings for a smoke test.")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    p.add_argument("--n-cases", type=int, default=100)
    p.add_argument("--n-classes", type=int, default=5)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--min-patches", type=int, default=20)
    p.add_argument("--max-patches", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir
    patch_dir = out_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)

    class_names = [f"class_{c}" for c in range(args.n_classes)]
    class_centroids = l2_normalize(rng.normal(size=(args.n_classes, args.dim)))

    case_ids, labels = [], []
    text_emb = np.zeros((args.n_cases, args.dim), dtype=np.float32)
    image_emb = np.zeros((args.n_cases, args.dim), dtype=np.float32)

    for i in range(args.n_cases):
        cls = i % args.n_classes
        case_id = f"SYN_{i:04d}"
        centroid = class_centroids[cls]

        text_emb[i] = l2_normalize(centroid + 0.3 * rng.normal(size=args.dim))
        image_emb[i] = l2_normalize(centroid + 0.3 * rng.normal(size=args.dim))

        n_patches = int(rng.integers(args.min_patches, args.max_patches + 1))
        patches = l2_normalize(image_emb[i] + 0.4 * rng.normal(size=(n_patches, args.dim)))
        np.save(patch_dir / f"{case_id}.npy", patches.astype(np.float32))

        case_ids.append(case_id)
        labels.append(class_names[cls])

    np.save(out_dir / "text.npy", text_emb)
    np.save(out_dir / "image_meanpool.npy", image_emb)
    pd.DataFrame({"case_id": case_ids, "label": labels}).to_csv(out_dir / "metadata.csv", index=False)

    print(f"Wrote synthetic data for {args.n_cases} cases ({args.n_classes} classes) to {out_dir}")


if __name__ == "__main__":
    main()
