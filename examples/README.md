# Examples

Fully synthetic smoke test — no real data, no clinical content (see
[docs/data_privacy.md](../docs/data_privacy.md)). Useful to verify the
installation and try out the four scripts end to end.

```bash
# 1. Generate synthetic embeddings (text, image mean-pool, patches, metadata)
python examples/generate_synthetic_data.py --out-dir examples/data

# 2. Baseline
python scripts/run_baseline.py \
    --text-emb examples/data/text.npy \
    --image-emb examples/data/image_meanpool.npy \
    --meta examples/data/metadata.csv \
    --out examples/results/baseline_query_level.csv

# 3. Fixed prototype reranking
python scripts/run_fixed_reranking.py \
    --text-emb examples/data/text.npy \
    --image-emb examples/data/image_meanpool.npy \
    --patch-dir examples/data/patches/ \
    --meta examples/data/metadata.csv \
    --K 8 \
    --out examples/results/fixed_pgr_query_level.csv

# 4. Adaptive prototype reranking
python scripts/run_adaptive_reranking.py \
    --text-emb examples/data/text.npy \
    --image-emb examples/data/image_meanpool.npy \
    --patch-dir examples/data/patches/ \
    --meta examples/data/metadata.csv \
    --out examples/results/adaptive_pgr_query_level.csv

# 5. All three methods in one run
python scripts/run_full_evaluation.py \
    --text-emb examples/data/text.npy \
    --image-emb examples/data/image_meanpool.npy \
    --patch-dir examples/data/patches/ \
    --meta examples/data/metadata.csv \
    --fixed-k 8 \
    --out-dir examples/results/
```

Each script prints a MacroRecall@{1,3,5,10} / MacroMRR@10 summary table.
Because the synthetic classes are randomly separated Gaussian blobs, absolute
numbers are meaningless — this is a plumbing check, not a benchmark.

`examples/data/` and `examples/results/` are git-ignored; regenerate them
locally whenever you need them.
