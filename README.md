# Prototype-guided Reranking for Cross-modal Histopathology Retrieval

Training-free prototype-guided reranking (PGR) for image–text retrieval with
frozen pathology foundation-model embeddings.

Given precomputed patch and text embeddings, the method improves retrieval by
replacing the global mean-pool image representation with local prototype
representations derived from patch clustering, without any additional training.

> **No clinical data, pretrained model weights, or restricted embeddings are
> redistributed in this repository.**

---

## Methods

Three retrieval strategies are provided:

| Method | Description |
|---|---|
| `mean_pooling` | Global cosine similarity baseline (mean-pooled image vs. text) |
| `fixed_PGR` | K-Means prototypes with fixed K; score = α·global + (1−α)·proto |
| `adaptive_PGR` | Per-case K* selected by silhouette; softmax-weighted centroids; confidence-weighted combination |

The adaptive strategy uses confidence weighting:

```
c_i  = (1 − ρ) · Q_FIXED  +  ρ · u(K*)
score = (1 − c_i) · s_global  +  c_i · s_proto
```

where Q_FIXED = 0.80, ρ = 0.50, and u(K*) is the per-case prototype quality.
See [docs/method_overview.md](docs/method_overview.md) for full details.

## Requirements

```
Python >= 3.9
numpy, pandas, scikit-learn
```

```bash
pip install -r requirements.txt
```

## Using your own embeddings

Prepare three inputs:

```
data/
  embeddings_text.npy          # (N, d) — one text embedding per case
  embeddings_image.npy         # (N, d) — mean-pooled patch embedding per case
  embeddings_patches/
    CASE_0001.npy              # (P_i, d) — patch embeddings, one file per case
  metadata.csv                 # columns: case_id, label
```

All embedding matrices must be **L2-normalised** and share the same dimension d.

Then run:

```bash
# Baseline
python scripts/run_baseline.py \
    --text-emb data/embeddings_text.npy \
    --image-emb data/embeddings_image.npy \
    --meta data/metadata.csv

# Fixed prototype reranking (K=8)
python scripts/run_fixed_reranking.py \
    --text-emb data/embeddings_text.npy \
    --image-emb data/embeddings_image.npy \
    --patch-dir data/embeddings_patches/ \
    --meta data/metadata.csv \
    --K 8

# Adaptive prototype reranking
python scripts/run_adaptive_reranking.py \
    --text-emb data/embeddings_text.npy \
    --image-emb data/embeddings_image.npy \
    --patch-dir data/embeddings_patches/ \
    --meta data/metadata.csv \
    --k-min 1 --k-max 12

# All three methods in one run
python scripts/run_full_evaluation.py \
    --text-emb data/embeddings_text.npy \
    --image-emb data/embeddings_image.npy \
    --patch-dir data/embeddings_patches/ \
    --meta data/metadata.csv \
    --fixed-k 8 --k-min 1 --k-max 12 \
    --out-dir results/
```

## Evaluation metrics

All metrics use **macro aggregation** (per-class mean, then mean over classes):

- **MacroRecall@K** — K ∈ {1, 3, 5, 10}
- **MacroMRR@10** — mean reciprocal rank
- **MacroMAP@10** — mean average precision

Exact-pair exclusion is applied by default: the query's own case is removed
from the candidate ranking before computing any metric.

## Repository structure

```
src/prototype_reranking/
  __init__.py
  prototypes.py     build_fixed_prototypes, build_adaptive_entry
  fixed.py          build_fixed_bank, score_matrix_fixed
  adaptive.py       build_bank, score_matrix_adaptive
  metrics.py        MacroRecall@K, MacroMRR@10, MacroMAP@10
  evaluation.py     evaluate_retrieval, evaluate_both_directions

scripts/
  run_baseline.py
  run_fixed_reranking.py
  run_adaptive_reranking.py
  run_full_evaluation.py

docs/
  method_overview.md
  external_models.md
  data_privacy.md
```

## External models

This repository does not include third-party encoder weights. Users must
obtain access to each foundation model from its original authors and comply
with the corresponding license and model card.
See [docs/external_models.md](docs/external_models.md).

## Data privacy

No clinical data, whole-slide images, pathology reports, or patient identifiers
are included. See [docs/data_privacy.md](docs/data_privacy.md).

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this code, please cite:

```bibtex
@software{rodriguez_alba_pgr_2025,
  author  = {Rodriguez Alba, Victor},
  title   = {Prototype-guided Reranking for Cross-modal Histopathology Retrieval},
  year    = {2025},
  url     = {https://github.com/VictorRoAlb/prototype-guided-reranking},
  license = {MIT}
}
```
