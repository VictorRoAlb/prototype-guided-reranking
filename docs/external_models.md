# External Models

This repository does **not** include or redistribute model weights, tokenizers,
or embeddings from third-party vision-language foundation models. Users must
obtain access to each model from its original authors and comply with the
corresponding license and usage terms.

## Models referenced in the associated thesis

| Model | Type | Access |
|---|---|---|
| KEEP | patch-level VL | [GitHub / HuggingFace — check model card] |
| CONCH | patch-level VL | [HuggingFace — gated, request access] |
| MUSK | patch-level VL | [HuggingFace — check model card] |
| PATHO-CLIP | patch-level VL | [HuggingFace / GitHub — check model card] |
| TITAN | WSI-level VL | [HuggingFace — gated, request access] |
| PRISM | WSI-level VL | [HuggingFace — gated, request access] |

**This repository does not grant access to any of these models.** If a model
requires a HuggingFace access request, submit it directly to the model authors.

## Python environment note

Each model above may require a different Python version, CUDA version, or set of
dependencies. This reranking repository is **independent** of those environments —
it only needs `numpy`, `scikit-learn`, `pandas`, and `matplotlib`. Once you have
your embeddings saved as `.npy` files, no model-specific environment is needed to
run the reranking scripts.

## What you need to use this code

1. Pre-encoded embeddings (text, BGAP image, and per-case patch matrices) from
   your model of choice, saved as `.npy` files.
2. A CSV metadata file with `case_id` and `label` columns.
3. No GPU or model loading is required at inference time.

## Producing compatible embeddings

Any vision-language model that produces L2-normalised patch embeddings and
corresponding text embeddings in the same shared embedding space is compatible.
The patch embedding matrix for each case should be stored as a `.npy` file of
shape `(n_patches, embedding_dim)`.
