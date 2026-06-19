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
| Patho-CLIP | patch-level VL | [HuggingFace / GitHub — check model card] |
| TITAN | WSI-level VL | [HuggingFace — gated, request access] |
| PRISM | WSI-level VL | [HuggingFace — gated, request access] |

**This repository does not grant access to any of these models.** If a model
requires a HuggingFace access request, submit it directly to the model authors.

## What you need to use this code

1. A set of pre-encoded embeddings (text, image mean-pool, and patch matrices)
   produced by your own model of choice.
2. A CSV metadata file with case identifiers and class labels.
3. No GPU or model loading is required at inference time — the reranking operates
   entirely on pre-computed numpy arrays.

## Producing compatible embeddings

Any vision-language model that produces L2-normalised patch embeddings and
corresponding text embeddings in the same shared embedding space is compatible.
The patch embedding matrix for each case should be stored as a `.npy` file of
shape `(n_patches, embedding_dim)`.
