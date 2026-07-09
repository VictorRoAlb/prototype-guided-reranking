# Method Overview

## Problem setting

Given $N$ histopathology cases, each represented by:

- a **text embedding** $t_i \in \mathbb{R}^d$ (encoded pathology report),
- a **global image embedding** $v_i \in \mathbb{R}^d$ (mean-pooled patch embeddings),
- a set of **patch embeddings** $\{p_{i,1}, \dots, p_{i,P}\} \in \mathbb{R}^d$,

the goal is to retrieve the most relevant cases for a given text or image query
(cross-modal I2T and T2I retrieval).

All embeddings must be L2-normalised. The reranking operates entirely on
precomputed numpy arrays; no model loading or GPU is required at inference time.

## Baseline: mean-pool global retrieval

$$
s_{\text{global}}(\text{query}, \text{candidate}) = \cos(v_{\text{query}}, t_{\text{candidate}}) \quad \text{(I2T)}
$$

$$
s_{\text{global}}(\text{query}, \text{candidate}) = \cos(t_{\text{query}}, v_{\text{candidate}}) \quad \text{(T2I)}
$$

## Fixed prototype reranking

Each candidate slide is represented by $K$ prototypes obtained from MiniBatchKMeans
clustering of its patch embeddings. The prototype score is the maximum cosine
similarity between the query and any of the $K$ prototypes:

$$
s_{\text{proto}} = \max_k \cos(\text{query}, \text{prototype}_k)
$$

The final score combines global and prototype similarity:

$$
s_{\text{final}} = (1 - q_{\text{proto}}) \cdot s_{\text{global}} + q_{\text{proto}} \cdot s_{\text{proto}}
$$

where $q_{\text{proto}} = 0.80$ (fixed, corresponding to $\alpha_{\text{global}} = 0.20$).
Reranking is applied to the top-50 candidates by global similarity; the rest retain
their global order.

$K$ is chosen per dataset based on mean patch count per slide. A practical guide:

| Mean patch count | Recommended $K$ |
|---|---|
| < 50 patches | 2 |
| 50–200 patches | 4–6 |
| > 200 patches | 8 |

## Adaptive prototype reranking

### $K^*$ selection (utility criterion)

For each case, several values of $K$ are evaluated from the grid $\{2, 4, 6, 8, 12\}$.
A $K$ value is valid only if it leaves at least $\text{min\_support} = 6$ patches per cluster.

For each valid $K$, two quantities are computed from the patch matrix alone:

$$
\text{coverage} = \text{mean over patches of } \max_k \cos(\text{patch}, \text{prototype}_k)
$$

$$
\text{support} = \text{clip}\left(\frac{\text{patches per prototype}}{\text{good\_support}}, 0, 1\right), \quad \text{good\_support} = 20
$$

$$
\text{utility} = \text{coverage} \times \text{support}
$$

$K^*$ is the smallest valid $K$ whose utility is within 97% of the maximum utility
across the grid (near-best parsimony criterion). This selects compact
representations that still cover the patch space well.

### Softmax-weighted centroids ($\tau = 0.05$)

Plain KMeans centroids are replaced by softmax-weighted centroids: within each
cluster, patches are weighted by their similarity to the cluster mean, sharpened
by temperature $\tau = 0.05$. The resulting centroid is more representative of the
dominant patch direction.

### Confidence weighting

The prototype confidence $c_i$ for each case blends a fixed baseline $Q_{\text{fixed}}$
with the per-case utility:

$$
c_i = (1 - \rho) \cdot Q_{\text{fixed}} + \rho \cdot \text{utility}(K^*), \quad Q_{\text{fixed}} = 0.80, \ \rho = 0.50
$$

Cases with weak prototype structure (few patches, low coverage) receive $c_i$
closer to 0.40 and rely more on global similarity. Cases with strong prototype
structure receive $c_i$ up to 0.80, giving more weight to local matching.

The final score uses the same formula as fixed reranking but with case-specific $c_i$:

$$
s_{\text{final}} = (1 - c_i) \cdot s_{\text{global}} + c_i \cdot s_{\text{proto}}
$$

## Evaluation protocol

- **Exact-pair exclusion**: the query's own case is removed from the candidate
  pool before ranking. Enabled by default.
- **Macro aggregation**: Recall@K and MRR@10 are computed per class, then
  averaged over classes (unweighted). Recommended for imbalanced datasets.
- **Directions**: I2T (image query → text candidates) and T2I (text query → image
  candidates) are evaluated independently.
