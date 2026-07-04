# Method Overview

## Problem setting

Given N histopathology cases, each represented by:
- a **text embedding** t_i ∈ ℝᵈ (encoded pathology report),
- a **global image embedding** v_i ∈ ℝᵈ (mean-pooled patch embeddings),
- a set of **patch embeddings** {p_i1, …, p_iP} ∈ ℝᵈ,

the goal is to retrieve the most relevant cases for a given text or image query
(cross-modal I2T and T2I retrieval).

All embeddings must be L2-normalised. The reranking operates entirely on
precomputed numpy arrays; no model loading or GPU is required at inference time.

## Baseline: mean-pool global retrieval

    s_global(query, candidate) = cos(v_query, t_candidate)   (I2T)
    s_global(query, candidate) = cos(t_query, v_candidate)   (T2I)

## Fixed prototype reranking

Each candidate slide is represented by K prototypes obtained from MiniBatchKMeans
clustering of its patch embeddings. The prototype score is the maximum cosine
similarity between the query and any of the K prototypes:

    s_proto = max_k  cos(query, prototype_k)

The final score combines global and prototype similarity:

    s_final = (1 − q_proto) · s_global  +  q_proto · s_proto

where **q_proto = 0.80** (fixed, corresponding to alpha_global = 0.20). Reranking
is applied to the top-50 candidates by global similarity; the rest retain their
global order.

K is chosen per dataset based on mean patch count per slide. A practical guide:

| Mean patch count | Recommended K |
|---|---|
| < 50 patches | 2 |
| 50–200 patches | 4–6 |
| > 200 patches | 8 |

## Adaptive prototype reranking

### K* selection (utility criterion)

For each case, several values of K are evaluated from the grid {2, 4, 6, 8, 12}.
A K value is valid only if it leaves at least `min_support = 6` patches per cluster.

For each valid K, two quantities are computed from the patch matrix alone:

    coverage = mean over patches of max_k cos(patch, prototype_k)
    support  = clip(patches_per_prototype / good_support, 0, 1)   [good_support = 20]
    utility  = coverage × support

**K* is the smallest valid K** whose utility is within 97% of the maximum
utility across the grid (near-best parsimony criterion). This selects compact
representations that still cover the patch space well.

### Softmax-weighted centroids (τ = 0.05)

Plain KMeans centroids are replaced by softmax-weighted centroids: within each
cluster, patches are weighted by their similarity to the cluster mean, sharpened
by temperature τ = 0.05. The resulting centroid is more representative of the
dominant patch direction.

### Confidence weighting

The prototype confidence c_i for each case blends a fixed baseline Q_FIXED with
the per-case utility:

    c_i = (1 − ρ) · Q_FIXED  +  ρ · utility(K*)     [Q_FIXED = 0.80, ρ = 0.50]

Cases with weak prototype structure (few patches, low coverage) receive c_i
closer to 0.40 and rely more on global similarity. Cases with strong prototype
structure receive c_i up to 0.80, giving more weight to local matching.

The final score uses the same formula as fixed reranking but with case-specific c_i:

    s_final = (1 − c_i) · s_global  +  c_i · s_proto

## Evaluation protocol

- **Exact-pair exclusion**: the query's own case is removed from the candidate
  pool before ranking. Enabled by default.
- **Macro aggregation**: Recall@K and MRR@10 are computed per class, then
  averaged over classes (unweighted). Recommended for imbalanced datasets.
- **Directions**: I2T (image query → text candidates) and T2I (text query → image
  candidates) are evaluated independently.
