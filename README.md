# Prototype-guided Reranking for Cross-modal Histopathology Retrieval

Training-free prototype-guided reranking (PGR) for cross-modal retrieval in
computational pathology. The method improves image–text and text–image retrieval
by replacing the global mean-pool image representation with local prototype
representations derived from patch-level embeddings, with no retraining required.

The code operates on **precomputed embeddings** from any vision-language
foundation model. No model weights are loaded at inference time.

> No clinical data, model weights, or restricted embeddings are redistributed
> in this repository.

---

## Methods

| Method | Description |
|---|---|
| `mean_pooling` | Global cosine similarity baseline (mean-pooled image vs. text) |
| `fixed_PGR` | K-Means prototypes with fixed K; score = 0.20·global + 0.80·proto |
| `adaptive_PGR` | Per-case K* via utility criterion; softmax-weighted centroids; confidence-blended combination |

### Fixed prototype reranking

Patch embeddings of each candidate slide are clustered into K prototypes using
MiniBatchKMeans. The prototype score is the maximum cosine similarity between the
query embedding and any of the K prototypes. The final score is:

```
s_final = 0.20 · s_global  +  0.80 · max_k cos(query, prototype_k)
```

### Adaptive prototype reranking

K* is selected **per case** from the grid {2, 4, 6, 8, 12} using a utility
criterion (coverage × support) with a near-best parsimony rule. Prototypes are
refined into softmax-weighted centroids (τ = 0.05). A confidence weight c_i
blends a fixed baseline with the per-case prototype quality:

```
utility(K) = coverage(K) × support(K)
K*         = smallest K with utility(K) ≥ 0.97 × max_utility
c_i        = 0.50 × 0.80  +  0.50 × utility(K*)
s_final    = (1 − c_i) · s_global  +  c_i · max_k cos(query, weighted_prototype_k)
```

See [docs/method_overview.md](docs/method_overview.md) for the full derivation.

---

## Requirements

```
Python >= 3.9
numpy >= 1.24
pandas >= 1.5
scikit-learn >= 1.2
```

```bash
pip install -r requirements.txt
```

---

## Input format

Prepare three inputs for your dataset:

```
data/
  text.npy                 # (N, d)  one L2-normalised text embedding per case
  image_meanpool.npy       # (N, d)  mean-pooled patch embedding per case, L2-normalised
  patches/
    CASE_0001.npy          # (P_i, d)  L2-normalised patch matrix, one file per case
  metadata.csv             # columns: case_id, label  (same row order as the .npy files)
```

All embedding matrices must share the same dimension d and be **L2-normalised**.

---

## Usage

### Baseline

```bash
python scripts/run_baseline.py \
    --text-emb data/text.npy \
    --image-emb data/image_meanpool.npy \
    --meta data/metadata.csv \
    --dataset MyDataset --model MyModel \
    --out results/baseline_query_level.csv
```

### Fixed prototype reranking

```bash
python scripts/run_fixed_reranking.py \
    --text-emb data/text.npy \
    --image-emb data/image_meanpool.npy \
    --patch-dir data/patches/ \
    --meta data/metadata.csv \
    --K 8 \
    --out results/fixed_pgr_query_level.csv
```

K should match the mean patch count of your dataset (see [docs/method_overview.md](docs/method_overview.md)).

### Adaptive prototype reranking

```bash
python scripts/run_adaptive_reranking.py \
    --text-emb data/text.npy \
    --image-emb data/image_meanpool.npy \
    --patch-dir data/patches/ \
    --meta data/metadata.csv \
    --out results/adaptive_pgr_query_level.csv
```

### All three methods in one run

```bash
python scripts/run_full_evaluation.py \
    --text-emb data/text.npy \
    --image-emb data/image_meanpool.npy \
    --patch-dir data/patches/ \
    --meta data/metadata.csv \
    --fixed-k 8 \
    --out-dir results/
```

Outputs: `results/query_level.csv` and `results/macro_metrics.csv`.

---

## Evaluation metrics

All metrics use **macro aggregation**: per-class mean, then mean over classes
(unweighted). This is recommended over micro-average for imbalanced datasets.

| Metric | Description |
|---|---|
| MacroRecall@K | K ∈ {1, 3, 5, 10} |
| MacroMRR@10 | Mean reciprocal rank at cutoff 10 |
| MacroMAP@10 | Mean average precision at cutoff 10 |

Exact-pair exclusion is applied by default: the query's own case is removed from
the candidate ranking before computing any metric.

---

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
  method_overview.md     Full method description and hyperparameters
  external_models.md     Notes on third-party foundation models
  data_privacy.md        Data handling and privacy statement

configs/
  example_config.yaml    Annotated configuration template
```

---

## External models

This repository does not include third-party encoder weights. Users must obtain
access to each foundation model from its original authors and comply with the
corresponding license and model card. See [docs/external_models.md](docs/external_models.md).

## Data privacy

No clinical data, whole-slide images, pathology reports, or patient identifiers
are included. See [docs/data_privacy.md](docs/data_privacy.md).

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
