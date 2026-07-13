"""Bridge script: turn one-file-per-case embedding folders into the single
stacked .npy arrays that run_baseline.py / run_fixed_reranking.py /
run_adaptive_reranking.py / run_full_evaluation.py expect for --text-emb and
--image-emb.

Patch embeddings do NOT need this: --patch-dir already expects one
"<case_id>.npy" file per case, which is how patches are naturally stored.
Only the per-case *text* and *mean-pooled image* embeddings need to be
stacked into a single (N, d) array, in the same row order as --meta.

Case order is taken from --meta (its case_id column), so the output arrays
are guaranteed aligned with the metadata the retrieval scripts also read.
Some labs store the text/image file for a case under a slightly different
stem than the case_id (e.g. "<case_id>_text.npy" instead of "<case_id>.npy")
— list every suffix you have with --suffixes; the first one found wins.

Another common mismatch (seen in the SICAP/KEEP layout): the case_id used
for image/patch files includes a per-fragment suffix ("16B0001851_1") but
the text file is keyed on the slide id alone ("16B0001851_text.npy"). Pass
--strip-id-suffix-regex '_\\d+$' to also try the case_id with that pattern
stripped before giving up on a directory.

Usage:
    python scripts/build_stacked_arrays.py \
        --meta data/metadata.csv \
        --text-dir /path/to/per_case_text_embeddings \
        --image-dir /path/to/per_case_meanpool_embeddings \
        --suffixes "",_text,_grouptext \
        --strip-id-suffix-regex '_\d+$' \
        --out-dir data/
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def find_case_file(directory: Path, case_id: str, suffixes: list[str], strip_regex: str | None) -> Path | None:
    candidate_ids = [case_id]
    if strip_regex:
        stripped = re.sub(strip_regex, "", case_id)
        if stripped != case_id:
            candidate_ids.append(stripped)
    for cid in candidate_ids:
        for suffix in suffixes:
            f = directory / f"{cid}{suffix}.npy"
            if f.exists():
                return f
    return None


def stack_directory(
    directory: Path, case_ids: list[str], suffixes: list[str], label: str, strip_regex: str | None
) -> tuple[np.ndarray, list[str]]:
    vectors, found_ids, missing = [], [], []
    for cid in case_ids:
        f = find_case_file(directory, cid, suffixes, strip_regex)
        if f is None:
            missing.append(cid)
            continue
        vectors.append(np.load(f).astype(np.float32))
        found_ids.append(cid)
    if missing:
        print(f"[{label}] {len(missing)}/{len(case_ids)} cases had no matching file under {directory} "
              f"(tried suffixes {suffixes}) — first few missing: {missing[:5]}")
    return np.stack(vectors, axis=0), found_ids


def main() -> None:
    p = argparse.ArgumentParser(description="Stack per-case embedding files into the arrays PGR scripts expect.")
    p.add_argument("--meta", required=True, type=Path, help="Metadata CSV; its case_id column fixes row order.")
    p.add_argument("--id-col", default="case_id")
    p.add_argument("--text-dir", required=True, type=Path)
    p.add_argument("--image-dir", required=True, type=Path)
    p.add_argument("--suffixes", default="",
                   help="Comma-separated filename suffixes to try before '.npy', e.g. '\"\",_text,_grouptext'.")
    p.add_argument("--strip-id-suffix-regex", default=None,
                   help=r"If a case_id has no direct file match, also try it with this regex stripped "
                        r"(e.g. '_\d+$' for a per-fragment suffix not present in the text filenames).")
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    suffixes = [s for s in args.suffixes.split(",")] or [""]
    meta = pd.read_csv(args.meta)
    case_ids = meta[args.id_col].astype(str).tolist()

    text_matrix, text_ids = stack_directory(args.text_dir, case_ids, suffixes, "text", args.strip_id_suffix_regex)
    image_matrix, image_ids = stack_directory(args.image_dir, case_ids, suffixes, "image", args.strip_id_suffix_regex)

    usable_ids = [cid for cid in case_ids if cid in set(text_ids) and cid in set(image_ids)]
    if len(usable_ids) < len(case_ids):
        print(f"Keeping the {len(usable_ids)}/{len(case_ids)} cases present in both text and image folders. "
              f"Filtering --meta accordingly in {args.out_dir / 'metadata.csv'}.")
        text_matrix, _ = stack_directory(args.text_dir, usable_ids, suffixes, "text (filtered)", args.strip_id_suffix_regex)
        image_matrix, _ = stack_directory(args.image_dir, usable_ids, suffixes, "image (filtered)", args.strip_id_suffix_regex)
        meta = meta[meta[args.id_col].astype(str).isin(usable_ids)].copy()
        meta = meta.set_index(meta[args.id_col].astype(str)).loc[usable_ids].reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "text.npy", text_matrix)
    np.save(args.out_dir / "image_meanpool.npy", image_matrix)
    meta.to_csv(args.out_dir / "metadata.csv", index=False)
    print(f"Wrote text.npy {text_matrix.shape}, image_meanpool.npy {image_matrix.shape}, "
          f"metadata.csv ({len(meta)} rows) to {args.out_dir}")


if __name__ == "__main__":
    main()
