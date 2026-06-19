# Data Privacy

## What this repository contains

This repository contains **only** model-agnostic Python code. It does not include:

- Whole-slide images (WSI) or image patches.
- Pathology reports or any free-text clinical content.
- Patient or case identifiers.
- Embeddings computed from real clinical data.
- Metadata tables derived from hospital or clinical systems.

The `examples/` folder uses **fully synthetic embeddings** generated from random
Gaussian noise with no relation to any real dataset.

## Intended use

The code is intended for researchers who already hold the appropriate institutional
and ethical approvals to work with histopathology data, and who have produced their
own embeddings from properly consented datasets.

## Responsible use

Users are responsible for:

1. Ensuring their data, embeddings, and metadata can be legally and ethically used
   for the intended purpose.
2. Complying with all applicable data-protection regulations (e.g. GDPR, HIPAA)
   before running the code on real clinical data.
3. Not committing real patient identifiers, embeddings, or report content to any
   version-control system.

## Contact

For questions about the methodology, open an issue in this repository. For
questions about data access or clinical governance, contact your local ethics
and data-governance team.
