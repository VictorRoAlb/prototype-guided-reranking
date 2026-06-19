# Prototype-guided Reranking for Cross-modal Histopathology Retrieval

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Training-free](https://img.shields.io/badge/training-free-orange.svg)]()

**Training-free reranking for image–text and text–image retrieval in computational pathology.**  
Works with any vision-language foundation model. Operates entirely on precomputed embeddings — no GPU or model loading at inference time.

> No clinical data, model weights, or restricted embeddings are redistributed in this repository.

---

## Method

![Method Overview](docs/figures/prototype_reranking_workflow.png)

*MPA = Mean-pool aggregation. f_V, f_T = frozen vision/text encoders (weights frozen at inference).  
Top-50 candidates retrieved by global cosine similarity are re-ranked by prototype similarity.*

---

## Quickstart

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Prepare your data

```
data/
  text.npy               # (N, d)   one L2-normalised text embedding per case
  image_meanpool.npy     # (N, d)   mean-pooled patch embedding per case
  patches/
    CASE_0001.npy        # (P_i, d) patch matrix per case  (variable P_i)
  metadata.csv           # two columns: case_id, label
```

All embeddings must be **L2-normalised** and share the same dimension d.

### 3. Run

```bash
# Baseline — global cosine similarity
python scripts/run_baseline.py \
    --text-emb data/text.npy --image-emb data/image_meanpool.npy \
    --meta data/metadata.csv

# Fixed prototype reranking  (K = mean patch count / 50, minimum 2)
python scripts/run_fixed_reranking.py \
    --text-emb data/text.npy --image-emb data/image_meanpool.npy \
    --patch-dir data/patches/ --meta data/metadata.csv --K 8

# Adaptive prototype reranking  (K* selected automatically per slide)
python scripts/run_adaptive_reranking.py \
    --text-emb data/text.npy --image-emb data/image_meanpool.npy \
    --patch-dir data/patches/ --meta data/metadata.csv

# All three methods in one run
python scripts/run_full_evaluation.py \
    --text-emb data/text.npy --image-emb data/image_meanpool.npy \
    --patch-dir data/patches/ --meta data/metadata.csv \
    --fixed-k 8 --out-dir results/
```

Each script prints a summary table (MacroRecall@K and MacroMRR@10) and writes
per-query results to CSV.

---

## Choosing K (fixed reranking)

| Mean patches per slide | Recommended K |
|---|---|
| < 50 | 2 |
| 50 – 200 | 4 – 6 |
| > 200 | 8 |

For adaptive reranking K* is selected automatically per slide — no tuning needed.

---

## Evaluation metrics

All metrics use **macro aggregation** (per-class mean → mean over classes), which
is robust to class imbalance.

| Metric | Description |
|---|---|
| MacroRecall@K | K ∈ {1, 3, 5, 10} |
| MacroMRR@10 | Mean reciprocal rank at cutoff 10 |
| MacroMAP@10 | Mean average precision at cutoff 10 |

**Exact-pair exclusion** is applied by default: the query's own case is removed
from the candidate pool before ranking.

---

## Repository structure

```
src/prototype_reranking/
  prototypes.py     build_fixed_prototypes, build_adaptive_entry (K* via utility)
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
  method_overview.md    Full formulation and hyperparameters
  external_models.md    Notes on third-party foundation models
  data_privacy.md       Data handling and privacy statement
```

---

## External models

This repository does not include third-party encoder weights. Users must obtain
access to each model from its original authors and comply with the corresponding
license and model card. See [docs/external_models.md](docs/external_models.md).

## License

[MIT](LICENSE) © 2026 Víctor Rodríguez Albendea

---

## Citation

```bibtex
@software{RodriguezAlbendea2026,
  author    = {Rodr{\'i}guez Albendea, V{\'i}ctor and
               Meseguer, Pablo and
               Terradez, Liria and
               Colomer, Adri{\'a}n},
  title     = {Prototype-guided Reranking for Cross-modal Histopathology Retrieval},
  year      = {2026},
  url       = {https://github.com/VictorRoAlb/prototype-guided-reranking},
  license   = {MIT}
}
```
