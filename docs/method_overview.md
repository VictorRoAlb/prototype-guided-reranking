# Method Overview

## Problem setting

Given a dataset of N histopathology cases, each represented by:
- a **text embedding** t_i ∈ ℝᵈ (from a pathology report, encoded by a vision-language model),
- a **global image embedding** v_i ∈ ℝᵈ (mean-pooled patch embeddings),
- a set of **patch embeddings** {p_i1, …, p_iP} ∈ ℝᵈ,

the goal is to retrieve the most relevant cases given either a text or image query
(cross-modal retrieval, I2T and T2I directions).

## Baseline: mean-pool retrieval

Global cosine similarity between mean-pooled image and text embeddings:

    s_global(q, c) = cos(v_q, t_c)   (I2T)
    s_global(q, c) = cos(t_q, v_c)   (T2I)

## Fixed prototype reranking

Patch embeddings of each candidate case are clustered into **K fixed prototypes**
using K-Means. The prototype score is the mean of the top-m cosine similarities
between the query embedding and the K prototypes:

    s_proto(q, c) = mean_{top-m} cos(q, p_ck)

The final score is a weighted combination:

    s_final = α · s_global  +  (1 − α) · s_proto

where α = 0.20 (prototype weight q_proto = 0.80). K is chosen per dataset based
on mean patch count (recommended: K = round(mean_patches / 50), minimum 2).

## Adaptive prototype reranking

K* is selected **per case** by silhouette score over the range [k_min, k_max].
Prototypes are **softmax-weighted centroids**: each cluster member is weighted by
its similarity to the cluster mean (temperature τ = 0.05), making the centroid
more representative of the dominant patch direction.

The confidence c_i blends a fixed baseline weight Q_FIXED with the per-case
prototype quality u(K*):

    c_i = (1 − ρ) · Q_FIXED  +  ρ · u(K*)

where Q_FIXED = 0.80, ρ = 0.50. The final score is:

    s_final = (1 − c_i) · s_global  +  c_i · s_proto

This allows the method to rely more on global similarity when prototype quality
is low (e.g. very few patches) and more on prototypes when quality is high.

## Evaluation protocol

- **Exact-pair exclusion**: the query case itself is removed from the candidate
  ranking before computing metrics. This prevents trivial self-retrieval.
- **Macro aggregation**: Recall@K, MRR@10, and MAP@10 are computed per class then
  averaged over classes (unweighted). This is recommended over micro-average for
  imbalanced datasets.

## Choosing K (fixed) or K_range (adaptive)

A practical rule of thumb based on the experiments in the associated paper:

| Mean patch count | Recommended K (fixed) |
|-----------------|----------------------|
| < 50            | 2                    |
| 50 – 200        | 4 – 6                |
| > 200           | 8                    |

For adaptive reranking, set k_max ≈ recommended_K + 4.
