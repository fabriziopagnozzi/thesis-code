# Synthetic Medical RAG Benchmark MVP

Run the MVP with local-Ollama chunk generation:

```bash
uv run python -m experiments.medical_dataset_gen.run_pipeline --exp mvp_full
```

Fast smoke run:

```bash
uv run python -m experiments.medical_dataset_gen.run_pipeline --exp mvp_smoke --max-queries 8
```

Fast deterministic smoke run without LLM chunk calls:

```bash
uv run python -m experiments.medical_dataset_gen.run_pipeline \
  --exp mvp_smoke_fast \
  --max-queries 8 \
  --no-llm-chunks
```

Full Qwen 8B embedding run with LLM-generated chunks:

```bash
uv run python -m experiments.medical_dataset_gen.run_pipeline \
  --exp llm_chunks_qwen8b \
  --llm-name gemma4:12b \
  --llm-workers 2 \
  --embedding-model Qwen/Qwen3-Embedding-8B \
  --device cuda \
  --batch-size 8
```

With the default config, every chunk is generated with the local Ollama model.
The default retrieval scope is `query_local`, which means each query retrieves from its own generated gold and distractor pool.
Use `--no-llm-chunks` only for deterministic smoke tests.

Use a different SentenceTransformers model:

```bash
uv run python -m experiments.medical_dataset_gen.run_pipeline \
  --exp mvp_st \
  --embedding-model Qwen/Qwen3-Embedding-8B
```

The config must already exist at `_results/<exp>/_config.yaml` before you run the pipeline.
The pipeline no longer copies a source-level `_config.yaml`; the default ontology is `ontology.yaml`.
Outputs are written under `_results/<exp>/`.

## Stages

1. `generation/query_plans.py`: creates `query_plans.parquet`.
2. `generation/facts.py`: creates hidden structured `clinical_facts.parquet`.
3. `generation/chunks.py`: generates realistic note chunks with Ollama when LLM mode is enabled, or deterministic clinical prose when disabled.
4. `generation/queries_answers.py`: creates `queries.parquet` and `gold_answers.parquet`.
5. `generation/qrels.py`: creates `qrels.parquet` from structured facts and chunks.
6. `retrieval/embed.py`: creates the embedding memmaps and metadata files.
7. `retrieval/filter_geometry.py`: creates `geometry_stats.parquet`.
8. `evaluation/evaluate.py`: creates `evaluation_results.parquet` and `evaluation_stats.parquet`.
9. `embedding_geometry/run.py`: creates embedding geometry figures and diagnostics for the configured candidate-pool scope.
10. `evaluation/plots.py`: creates evaluation figures under `_results/<exp>/_figures/evaluation/`.

Each stage can run as a module with the same flags as the full pipeline.

Regenerate plots from existing evaluation tables:

```bash
uv run python -m experiments.medical_dataset_gen.run_pipeline \
  --exp llm_chunks_qwen8b \
  --start-at plots
```

Generate embedding-geometry figures from existing embeddings/evaluation artifacts:

```bash
uv run python -m experiments.medical_dataset_gen.run_pipeline \
  --exp llm_chunks_qwen8b \
  --start-at embedding_geometry \
  --stop-after embedding_geometry
```

Useful overrides:

```bash
uv run python -m experiments.medical_dataset_gen.run_pipeline \
  --exp llm_chunks_qwen8b \
  --start-at embedding_geometry \
  --stop-after embedding_geometry \
  --embedding-geometry-queries 10 \
  --embedding-geometry-k 20 \
  --embedding-geometry-reduction umap
```

Pool-scope options:

- `query_local`: only the chunks generated from the same hidden query plan. Good for debugging generation quality, too easy for the real benchmark.
- `same_condition`: all chunks whose `condition_id` matches the query condition. This is the main evaluation setting.
- `full_corpus`: all chunks across all conditions. This is a harder retrieval stress test and can blur the intended coverage effect if condition separation dominates the error mode.

The embedding-geometry stage writes:

- `_figures/embedding_geometry/<selection_group>/<query_id>/query_overview_4panel.png`
- `_figures/embedding_geometry/<selection_group>/<query_id>/candidate_pool_map.png`
- `_figures/embedding_geometry/<selection_group>/<query_id>/strategy_selection_overlay.png`
- `_figures/embedding_geometry/<selection_group>/<query_id>/full_strategy_selection_overlay_k<K>.png`
- `_figures/embedding_geometry/<selection_group>/<query_id>/query_cosine_similarity_map.png`
- `_figures/embedding_geometry/<selection_group>/<query_id>/query_similarity_rank.png`
- `_figures/embedding_geometry/<selection_group>/<query_id>/hdbscan_cluster_map.png`
- `_figures/embedding_geometry/cluster_quality_overview.png`
- `embedding_geometry_points.parquet`
- `embedding_geometry_query_stats.parquet`

`selection_group` is `good`, `mid`, or `bad` for automatic mixed query selection, `good` for best-only selection, and `manual` when `embedding_geometry.query_ids` is set.

The plotting stage writes:

- `strategy_comparison.png`
- `lambda_sensitivity.png`
- `per_query_distributions.png`
- `gain_over_topk.png`
- `coverage_precision_tradeoff.png`
- `selection_diagnostics.png`

## LLM Usage

LLM chunk generation is on by default and uses `generation.llm_name` from the experiment config. You can also override it with `--llm-name`.
Parallel chunk generation is controlled by `generation.llm_workers` or `--llm-workers`.
To get actual concurrent decoding, start the Ollama server with `OLLAMA_NUM_PARALLEL` at least as large as `llm_workers`.
Generated LLM chunks are cached incrementally in `_results/<exp>/chunk_generation_cache.jsonl`, so interrupted chunk generation can be restarted without repeating accepted rows. If you enable deterministic template rewriting, rewritten chunks are stored in the isolated global cache at `src/experiments/medical_dataset_gen/_cache/chunk_rewrite_cache.jsonl`.
Chunk generation is binary: either all chunks are LLM-generated, or all chunks are deterministic fallback chunks.

To disable LLM chunks for fast debugging, use `--no-llm-chunks` or set:

```yaml
generation:
  use_llm_chunk_generation: false
```

Query paraphrasing is still off by default. To enable it:

```yaml
generation:
  use_llm_query_paraphrase: true
```
