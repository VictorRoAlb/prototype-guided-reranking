"""
visualize_wsi.py
================
Prototype activation map on a whole-slide image (TIF format).

For a given text query and a WSI with precomputed patch embeddings, the script
identifies which patches are activated by the winning prototype and overlays
them on the slide thumbnail as colored squares.

Requirements (in addition to requirements.txt):
    tifffile >= 2023.1.23     (pip install tifffile)
    Pillow                    (pip install Pillow)

Usage — fixed K:
    python scripts/visualize_wsi.py \
        --tif       path/to/slide.tif \
        --coords    path/to/patch_coords.csv \
        --patches   path/to/patch_embeddings.npy \
        --text-emb  path/to/text_query.npy \
        --method    fixed --K 8 \
        --out       outputs/activation_map.png

Usage — adaptive K* (selected automatically per slide):
    python scripts/visualize_wsi.py \
        --tif       path/to/slide.tif \
        --coords    path/to/patch_coords.csv \
        --patches   path/to/patch_embeddings.npy \
        --text-emb  path/to/text_query.npy \
        --method    adaptive \
        --out       outputs/activation_map.png

Input formats
-------------
patch_coords.csv : CSV with columns x, y (top-left corner of each patch in
    slide pixels). An optional patch_size_px column overrides --patch-size.
    Rows must be aligned with patch_embeddings.npy (row i = patch i).
patch_embeddings.npy : (P, d) float32 array, L2-normalised patch embeddings.
text_query.npy : (d,) float32 array, L2-normalised text embedding for the query.

Output
------
A two-panel PNG (and optionally PDF) figure:
    Left  — original slide thumbnail
    Right — same thumbnail with activated patches overlaid as colored squares
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prototype_reranking.prototypes import (
    build_fixed_prototypes,
    build_adaptive_entry,
    l2_normalize,
)

ACTIVATION_COLOR = "#E63946"   # red for activated patches
ALPHA = 0.45


def load_tif_thumbnail(tif_path: Path, max_dim: int = 1024) -> np.ndarray:
    """Load the lowest-resolution level of a TIF as an RGB uint8 array."""
    try:
        import tifffile
        with tifffile.TiffFile(str(tif_path)) as tif:
            # Try to load the smallest available series/level
            series = tif.series[0]
            level_idx = len(series.levels) - 1
            thumb = series.levels[level_idx].asarray()
    except Exception:
        # Fallback: use Pillow with thumbnail
        from PIL import Image
        img = Image.open(str(tif_path))
        img.thumbnail((max_dim, max_dim))
        return np.array(img.convert("RGB"))

    # Convert to RGB uint8
    if thumb.ndim == 2:
        thumb = np.stack([thumb] * 3, axis=-1)
    elif thumb.shape[0] in (3, 4) and thumb.ndim == 3:
        thumb = thumb[:3].transpose(1, 2, 0)
    if thumb.dtype != np.uint8:
        thumb = (thumb / thumb.max() * 255).clip(0, 255).astype(np.uint8)
    thumb = thumb[..., :3]

    # Downscale to max_dim if needed
    h, w = thumb.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        from PIL import Image
        pil = Image.fromarray(thumb)
        pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        thumb = np.array(pil)
    return thumb


def get_activated_patches(
    patches: np.ndarray,
    text_emb: np.ndarray,
    method: str,
    K: int,
) -> tuple[np.ndarray, int, int]:
    """Return boolean mask of activated patches and K* used.

    Activated patches = those assigned to the prototype cluster whose centroid
    is most similar to the text query (winning prototype).
    """
    patches = l2_normalize(np.asarray(patches, dtype=np.float32))
    text_emb = l2_normalize(np.asarray(text_emb, dtype=np.float32).ravel())

    if method == "fixed":
        protos, assignments = build_fixed_prototypes(patches, K)
        k_star = K
    else:
        entry = build_adaptive_entry("query_slide", patches)
        protos = entry["prototypes"]
        assignments = entry["assignments"]
        k_star = int(entry["K_star"])

    # Winning prototype: most similar to the text query
    sims = protos @ text_emb
    winning = int(np.argmax(sims))
    activated = assignments == winning
    n_activated = int(activated.sum())
    return activated, k_star, n_activated


def scale_coords(
    coords_px: np.ndarray,
    patch_size_px: int,
    slide_wh: tuple[int, int],
    thumb_wh: tuple[int, int],
) -> tuple[np.ndarray, float]:
    """Scale patch coordinates from slide pixels to thumbnail pixels."""
    sw, sh = slide_wh
    tw, th = thumb_wh
    scale = min(tw / sw, th / sh)
    coords_thumb = coords_px * scale
    patch_thumb = patch_size_px * scale
    return coords_thumb, patch_thumb


def make_figure(
    thumb: np.ndarray,
    coords_px: np.ndarray,
    patch_size_px: int,
    activated: np.ndarray,
    slide_wh: tuple[int, int],
    k_star: int,
    n_activated: int,
    out_path: Path,
    label: str = "",
) -> None:
    th, tw = thumb.shape[:2]
    coords_thumb, patch_thumb = scale_coords(
        coords_px, patch_size_px, slide_wh, (tw, th)
    )

    # Aspect ratio for each panel
    aspect = tw / th
    panel_h = 5.0
    fig, axes = plt.subplots(1, 2, figsize=(2 * panel_h * aspect + 0.4, panel_h))
    fig.subplots_adjust(wspace=0.03)

    for ax in axes:
        ax.imshow(thumb)
        ax.axis("off")

    # Right panel: overlay activated patches
    ax = axes[1]
    for i, (xy, act) in enumerate(zip(coords_thumb, activated)):
        if act:
            rect = mpatches.Rectangle(
                xy, patch_thumb, patch_thumb,
                linewidth=0, facecolor=ACTIVATION_COLOR, alpha=ALPHA,
            )
            ax.add_patch(rect)

    title = f"K*={k_star}  |  {n_activated}/{len(activated)} patches activated"
    if label:
        title = f"{label}  —  {title}"
    axes[0].set_title("Original", fontsize=10, pad=4)
    axes[1].set_title(title, fontsize=9, pad=4)

    for ext in ("png", "pdf"):
        p = out_path.with_suffix(f".{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path.with_suffix('.png')}  (+ .pdf)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype activation map on a WSI.")
    parser.add_argument("--tif",      required=True, type=Path, help="Path to WSI .tif file")
    parser.add_argument("--coords",   required=True, type=Path, help="CSV with x,y patch coords (slide pixels)")
    parser.add_argument("--patches",  required=True, type=Path, help="(P,d) .npy patch embeddings (L2-normalised)")
    parser.add_argument("--text-emb", required=True, type=Path, help="(d,) .npy text query embedding (L2-normalised)")
    parser.add_argument("--method",   default="adaptive", choices=["fixed", "adaptive"])
    parser.add_argument("--K",        type=int, default=8, help="Number of prototypes (fixed method only)")
    parser.add_argument("--patch-size", type=int, default=256, help="Patch size in slide pixels (default: 256)")
    parser.add_argument("--label",    default="", help="Optional label for the figure title")
    parser.add_argument("--out",      required=True, type=Path, help="Output path (extension replaced with .png/.pdf)")
    args = parser.parse_args()

    print(f"Loading slide: {args.tif}")
    thumb = load_tif_thumbnail(args.tif)
    th, tw = thumb.shape[:2]
    print(f"  Thumbnail: {tw}×{th} px")

    coords_df = pd.read_csv(args.coords)
    coords_px = coords_df[["x", "y"]].values.astype(np.float32)
    patch_size_px = int(coords_df["patch_size_px"].iloc[0]) if "patch_size_px" in coords_df.columns else args.patch_size

    patches = np.load(args.patches).astype(np.float32)
    text_emb = np.load(args.text_emb).astype(np.float32)

    if len(coords_px) != len(patches):
        raise ValueError(f"Coordinate rows ({len(coords_px)}) != patch rows ({len(patches)})")

    print(f"  Patches: {patches.shape}  |  method: {args.method}  |  K={args.K}")
    activated, k_star, n_act = get_activated_patches(patches, text_emb, args.method, args.K)
    print(f"  K*={k_star}, activated={n_act}/{len(patches)}")

    # Infer slide dimensions from coordinates + patch size
    x_max = int(coords_px[:, 0].max()) + patch_size_px
    y_max = int(coords_px[:, 1].max()) + patch_size_px
    slide_wh = (x_max, y_max)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    make_figure(
        thumb, coords_px, patch_size_px, activated,
        slide_wh, k_star, n_act,
        args.out.with_suffix(""),
        label=args.label,
    )


if __name__ == "__main__":
    main()
