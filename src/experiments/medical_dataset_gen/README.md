## Main Entrypoint

Run the full pipeline with:
```bash
uv run python -m experiments.medical_dataset_gen.pipeline --exp <experiment_name>
```
The experiment config must already exist at:
```text
src/experiments/medical_dataset_gen/_results/<experiment_name>/_config.yaml
```
Outputs are written under:
```text
src/experiments/medical_dataset_gen/_results/<experiment_name>/
```

## Stage Selection

The orchestrator in [pipeline/__main__.py](./src/experiments/medical_dataset_gen/pipeline/__main__.py) supports:

- `--to <stage>`
- `--from <stage>`
- `--stages <stage1,stage2,...>`
- `--release-llm <bool>`
- `--no-log-tee`

Example: rerun only the evaluation and plotting tail of a completed experiment:

```bash
uv run python -m experiments.medical_dataset_gen.pipeline \
  --exp <experiment_name> \
  --from eval \
  --to eval_plots
```

Example: regenerate only geometry plots from existing artifacts:

```bash
uv run python -m experiments.medical_dataset_gen.pipeline \
  --exp <experiment_name> \
  --stages geom_plots
```

## Stage Order

The current pipeline stages are:

1. `plans`
2. `facts`
3. `chunks`
4. `queries_answers`
5. `qrels`
6. `embed`
7. `filter_queries`
8. `eval`
9. `geom_plots`
10. `eval_plots`

## Package Layout

- [pipeline](./src/experiments/medical_dataset_gen/pipeline): runnable stage entrypoints and the pipeline orchestrator
- [dataset_generation](./src/experiments/medical_dataset_gen/dataset_generation): ontology loading, plan helpers, fact construction, chunk rendering, caches, and query/answer templating
- [query_geometry](./src/experiments/medical_dataset_gen/query_geometry): geometry artifact building, dimensionality reduction, diagnostics, and plot helpers
- [evaluation](./src/experiments/medical_dataset_gen/evaluation): retrieval metrics, answer metrics, lambda selection, worker setup, and plotting
- [schemas](./src/experiments/medical_dataset_gen/schemas): typed models shared across the pipeline
- [utils](./src/experiments/medical_dataset_gen/utils): config loading, paths, I/O, and shared retrieval helpers
- [data_templates](./src/experiments/medical_dataset_gen/data_templates): ontology and template YAML resources

## Main Artifacts

Key outputs written by the pipeline:

- `query_plans.parquet`
- `clinical_facts.parquet`
- `chunk_documents.parquet`
- `chunk_memberships.parquet`
- `generation_rejects.parquet`
- `queries.parquet`
- `gold_answers.parquet`
- `qrels.parquet`
- `geometry_stats.parquet`
- `evaluation_results.parquet`
- `evaluation_stats.parquet`
- `query_geometry_points.parquet`
- `query_geometry_stats.parquet`

The embedding stage also writes resolved chunk/query vector arrays, chunk/query ID arrays, and embedding metadata. For subexperiments with `global.use_shared: true`, these live under the parent `_embeddings/` directory instead of each child result folder. Existing local/canonical arrays can be moved into that layout with:

```bash
uv run -m experiments.medical_dataset_gen.utils.migrate_shared_embedding_artifacts --summary-only
uv run -m experiments.medical_dataset_gen.utils.migrate_shared_embedding_artifacts --apply --strip-embedding-overrides
```

## Retrieval Pool

Schema v2 uses only `retrieval.pool_scope: query_local`: chunks linked to the query through `chunk_memberships.parquet`.

## LLM Usage

Chunk generation behavior is controlled from the experiment config:

- `generation.llm_config.use_llm_chunk_generation`
- `generation.llm_config.use_llm_chunk_rewriting`
- `generation.llm_config.use_llm_query_paraphrase`
- `generation.llm_config.model_name`
- `generation.llm_config.num_workers`

If LLM chunk generation is enabled, accepted generations are cached in:

- `_results/<exp>/chunk_generation_cache.jsonl`
- `src/experiments/medical_dataset_gen/_cache/chunk_generation_cache.jsonl`

If LLM rewrite is enabled, rewrite cache entries are stored in:

- `src/experiments/medical_dataset_gen/_cache/chunk_rewrite_cache.jsonl`

When `--release-llm true` is passed to the pipeline orchestrator, the configured Ollama model is released just before the `embed` stage.

## Direct Stage Execution

Some stages can also be run directly as modules when you want to rerun only one piece of the pipeline:

```bash
uv run python -m experiments.medical_dataset_gen.pipeline.p04_chunks --exp <experiment_name>
uv run python -m experiments.medical_dataset_gen.pipeline.p09_evaluate --exp <experiment_name>
uv run python -m experiments.medical_dataset_gen.pipeline.p10_query_geom_plots --exp <experiment_name>
```

Evaluation figures are normally generated through the orchestrator, but selective plot generation is supported by [pipeline/p11_eval_plots.py](./src/experiments/medical_dataset_gen/pipeline/p11_eval_plots.py) through its `--plots` parser.

## Documentation

The main code documentation lives in [docs/code_docs](./src/experiments/medical_dataset_gen/docs/code_docs):

- [pipeline_overview.md](./src/experiments/medical_dataset_gen/docs/code_docs/pipeline_overview.md)
- [configuration_and_artifacts.md](./src/experiments/medical_dataset_gen/docs/code_docs/configuration_and_artifacts.md)
- [generation_stages.md](./src/experiments/medical_dataset_gen/docs/code_docs/generation_stages.md)
- [deterministic_construction.md](./src/experiments/medical_dataset_gen/docs/code_docs/deterministic_construction.md)
- [retrieval_and_geometry.md](./src/experiments/medical_dataset_gen/docs/code_docs/retrieval_and_geometry.md)
- [evaluation_and_plots.md](./src/experiments/medical_dataset_gen/docs/code_docs/evaluation_and_plots.md)
