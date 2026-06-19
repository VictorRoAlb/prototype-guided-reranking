# Prototype-guided Reranking for Cross-modal Histopathology Retrieval

Training-free method to improve image–text and text–image retrieval in
computational pathology. No retraining required.

> No clinical data, model weights, or restricted embeddings are redistributed
> in this repository.

---

## How it works

Standard cross-modal retrieval ranks candidates by the cosine similarity between
a global mean-pooled image embedding and a text embedding. This works but ignores
the internal patch structure of each slide.

This repository provides two reranking strategies that replace or complement the
global score with a **prototype score** derived from the slide's patch embeddings:

1. **Fixed PGR** — cluster each slide's patches into K prototypes (fixed per
   dataset). Score each candidate by the maximum cosine similarity between the
   query and its K prototypes, then combine with the global score.

2. **Adaptive PGR** — select K* per slide automatically using a patch-coverage
   criterion. Refine prototypes into softmax-weighted centroids. Use a
   per-slide confidence weight that blends a fixed baseline with the quality of
   the prototype structure.

Both strategies rerank the top-100 candidates by global similarity and leave
the rest in their original order.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare your embeddings

You need three precomputed numpy arrays from your model of choice, plus a
metadata CSV:

```
data/
  text.npy              # (N, d)  one L2-normalised text embedding per case
  image_meanpool.npy    # (N, d)  mean-pooled patch embedding per case
  patches/
    CASE_0001.npy       # (P_i, d)  patch matrix for each case (variable P_i)
  metadata.csv          # two columns: case_id, label
```

All embeddings must be **L2-normalised** and share the same dimension d.

### 3. Run

```bash
# Baseline — global cosine similarity
python scripts/run_baseline.py \
    --text-emb data/text.npy \
    --image-emb data/image_meanpool.npy \
    --meta data/metadata.csv

# Fixed prototype reranking (set K to match your dataset's mean patch count)
python scripts/run_fixed_reranking.py \
    --text-emb data/text.npy \
    --image-emb data/image_meanpool.npy \
    --patch-dir data/patches/ \
    --meta data/metadata.csv \
    --K 8

# Adaptive prototype reranking
python scripts/run_adaptive_reranking.py \
    --text-emb data/text.npy \
    --image-emb data/image_meanpool.npy \
    --patch-dir data/patches/ \
    --meta data/metadata.csv

# All three methods in one run → results/macro_metrics.csv
python scripts/run_full_evaluation.py \
    --text-emb data/text.npy \
    --image-emb data/image_meanpool.npy \
    --patch-dir data/patches/ \
    --meta data/metadata.csv \
    --fixed-k 8 \
    --out-dir results/
```

Each script prints a MacroRecall@K / MacroMRR@10 summary table to the terminal
and writes per-query results to a CSV.

---

## Choosing K (fixed reranking)

K should reflect the typical number of patches per slide in your dataset:

| Mean patch count | Recommended K |
|---|---|
| < 50 | 2 |
| 50 – 200 | 4 – 6 |
| > 200 | 8 |

For adaptive reranking K is selected automatically per slide; no manual tuning
is needed.

---

## Metrics

All metrics use **macro aggregation** (per-class mean, then mean over classes),
which is robust to class imbalance.

| Metric | Description |
|---|---|
| MacroRecall@K | K ∈ {1, 3, 5, 10} |
| MacroMRR@10 | Mean reciprocal rank at cutoff 10 |
| MacroMAP@10 | Mean average precision at cutoff 10 |

Exact-pair exclusion is applied by default: the query's own case is removed
from the candidate pool before ranking.

---

## Method details

See [docs/method_overview.md](docs/method_overview.md) for the full formulation,
hyperparameter definitions, and the K* selection criterion.

## Repository structure

```
src/prototype_reranking/
  prototypes.py     l2_normalize, build_fixed_prototypes, build_adaptive_entry
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

This repository does not include third-party encoder weights. See
[docs/external_models.md](docs/external_models.md).

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{rodriguez_albendea_pgr_2026,
  author  = {Rodríguez Albendea, Víctor},
  title   = {Prototype-guided Reranking for Cross-modal Histopathology Retrieval},
  year    = {2026},
  url     = {https://github.com/VictorRoAlb/prototype-guided-reranking},
  license = {MIT}
}
```
