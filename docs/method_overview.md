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

## Notation

$H$ denotes the **number of prototypes per case** (fixed variant), and $H_i^*$
the **adaptive, per-case prototype count** selected for case $i$ (written
$H_j^*$ in the thesis; $i$/$j$ both index a case). $K$ is reserved for
**ranking depth**, as in Recall@$K$ and MacroRecall@$K$ below — it is never the
number of prototypes. The CLI flags and Python parameters keep the shorter name
`K` / `K_star` (e.g. `--K`, `k_grid`, `K_star` in the returned dict); wherever
you see `K` in the code or CLI help, read it as $H$ in this document.

## Baseline: mean-pool global retrieval

$$
s_{\text{global}}(\text{query}, \text{candidate}) = \cos(v_{\text{query}}, t_{\text{candidate}}) \quad \text{(I2T)}
$$

$$
s_{\text{global}}(\text{query}, \text{candidate}) = \cos(t_{\text{query}}, v_{\text{candidate}}) \quad \text{(T2I)}
$$

## Fixed prototype reranking

Each candidate slide is represented by $H$ prototypes obtained from MiniBatchKMeans
clustering of its patch embeddings. The prototype score is the maximum cosine
similarity between the query and any of the $H$ prototypes:

$$
s_{\text{proto}} = \max_h \cos(\text{query}, \text{prototype}_h)
$$

The final score combines global and prototype similarity:

$$
s_{\text{final}} = (1 - q_{\text{proto}}) \cdot s_{\text{global}} + q_{\text{proto}} \cdot s_{\text{proto}}
$$

where $q_{\text{proto}} = 0.80$ (fixed, corresponding to $\alpha_{\text{global}} = 0.20$).
Reranking is applied to the top-50 candidates by global similarity; the rest retain
their global order.

$H$ is chosen per dataset based on mean patch count per slide. A practical guide:

| Mean patch count | Recommended $H$ |
|---|---|
| < 50 patches | 2 |
| 50–200 patches | 4–6 |
| > 200 patches | 8 |

## Adaptive prototype reranking

### $H_i^*$ selection (utility criterion)

For each case $i$ with $N_i$ patches, several candidate values of $H$ are
evaluated from the grid $\{2, 4, 6, 8, 12\}$. A candidate $H$ is valid only if
the **mean support** — the average number of patches per prototype — meets the
threshold $N_i / H \ge$ `min_support` $= 6$. This is a constraint on the
*average* cluster size, not a per-cluster guarantee: because MiniBatchKMeans
clusters can be unbalanced, an individual cluster may still end up with fewer
than 6 patches even when $N_i / H \ge 6$ holds.

For each valid $H$, two quantities are computed from the patch matrix alone:

$$
\text{coverage} = \text{mean over patches of } \max_h \cos(\text{patch}, \text{prototype}_h)
$$

$$
\text{support} = \text{clip}\left(\frac{N_i / H}{g}, 0, 1\right)
$$

where $g = 20$ (`good_support` in the code) and $N_i / H$ is the mean number of
patches per prototype.

$$
\text{utility} = \text{coverage} \times \text{support}
$$

$H_i^*$ is the smallest valid $H$ whose utility is within 97% of the maximum utility
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
c_i = (1 - \rho) \cdot Q_{\text{fixed}} + \rho \cdot \text{utility}(H_i^*), \quad Q_{\text{fixed}} = 0.80, \ \rho = 0.50
$$

Cases with weak prototype structure (few patches, low coverage) receive $c_i$
closer to 0.40 and rely more on global similarity. Cases with strong prototype
structure receive $c_i$ up to 0.90, giving more weight to local matching.

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

## Statistical significance

Implemented in `src/prototype_reranking/significance.py`. `paired_bootstrap_delta`
compares two methods on a per-query metric column: queries are resampled with
replacement *within each class* (class-stratified, preserving macro-averaging),
the same resampled indices are applied to both methods in every replicate
(paired), and the p-value is **bilateral** (two-sided):

$$
p = \min\left(1,\; 2 \min\big(P(\Delta^{(b)} \le 0),\; P(\Delta^{(b)} \ge 0)\big)\right)
$$

where $\Delta^{(b)} = M_a^{(b)} - M_b^{(b)}$ is the paired delta in bootstrap
replicate $b$ (B = 10,000, fixed seed). A single bilateral test is sufficient to
detect both improvements and degradations; the *direction* is read off the sign
of the observed (non-bootstrapped) delta, not from two separate one-sided tests.

Raw p-values from a family of related comparisons (e.g. all patch-level models
× both PGR variants, for a fixed metric/cohort/direction) are corrected with
**Holm-Bonferroni** (`holm_bonferroni`); `significance_symbol` then maps the
adjusted p-value and the delta's sign to `*` (significant improvement), `†`
(significant degradation), or `""` (not significant at $p^{\text{adj}} < 0.05$).

```python
from prototype_reranking import paired_bootstrap_delta, holm_bonferroni, significance_symbol

result = paired_bootstrap_delta(query_df, "MRR@10", "adaptive_PGR", "mean_pooling")
holm_adjusted = holm_bonferroni([r["raw_p_bilateral"] for r in family_results])
symbols = [significance_symbol(p, r["observed_delta"]) for p, r in zip(holm_adjusted, family_results)]
```

This module covers single-pair, query-level comparisons. The thesis also
reports two coarser aggregations (a global summary averaged over all cohorts,
models and directions, and a per-cohort/direction summary averaged over the
four patch-level models) built on top of this same procedure; those aggregation
scripts are analysis code specific to the thesis and are not included here.
