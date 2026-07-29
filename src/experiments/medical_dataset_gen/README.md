## Main Entrypoint

Run the full pipeline with:
```bash
uv run python -m experiments.medical_dataset_gen.pipeline --exp <experiment_name>
```
Pass `--version v2`, `--version v3`, or `--version v4` to override `dataset_schema_version` from the resolved parent/child configuration for that invocation. The override is applied before artifact paths are resolved and is forwarded when a parent invocation launches its child experiments.

```bash
uv run python -m experiments.medical_dataset_gen.pipeline \
  --exp <experiment_name> \
  --version v4
```

The experiment config must already exist at:
```text
src/experiments/medical_dataset_gen/_results/<experiment_name>/_config.yaml
```
Outputs are written under:
```text
src/experiments/medical_dataset_gen/_results/<experiment_name>/v<dataset-schema-version>/
```

For subexperiments, keep `_subconfig.yaml` in the unversioned child directory. Run-local, non-shared artifacts are written below the version leaf, for example `_results/<parent>/<child>/v4/` for schema v4. Shared generation artifacts are dependency-partitioned below `_results/<parent>/_shared_v4/`: `base/` for plans and facts, `chunks/<mode>/` for rendered documents, memberships, and qrels, and `queries/<mode>/` for queries and answers.

## Stage Selection

The orchestrator in [pipeline/__main__.py](./src/experiments/medical_dataset_gen/pipeline/__main__.py) supports:

- `--to <stage>`
- `--from <stage>`
- `--stages <stage1,stage2,...>`
- `--version <v2|v3|v4>`
- `--queries-only`
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

When chunk embeddings for the resolved schema version, chunk mode, and embedding model already exist, embed only a new query surface with:

```bash
uv run python -m experiments.medical_dataset_gen.pipeline \
  --exp <experiment_name> \
  --version v4 \
  --queries-only
```

The flag affects only the `embed` stage. It fails instead of rebuilding if the resolved chunk vectors or chunk IDs are absent or invalid.

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
- [dataset_generation](./src/experiments/medical_dataset_gen/dataset_generation): ontology loading, plan helpers, fact construction, deterministic chunk rendering, validation, and query/answer templating
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
- `queries.parquet`
- `gold_answers.parquet`
- `qrels.parquet`
- `geometry_stats.parquet`
- `evaluation_results.parquet`
- `evaluation_stats.parquet`
- `query_geometry_points.parquet`
- `query_geometry_stats.parquet`

The embedding stage also writes resolved chunk/query vector arrays, chunk/query ID arrays, and embedding metadata. For subexperiments with `global.use_shared: true`, these live under the parent schema-specific `_embeddings_v<dataset-schema-version>/` directory instead of each child result folder. Existing local/canonical arrays can be moved into that layout with:

```bash
uv run -m experiments.medical_dataset_gen.utils.migrate_shared_embedding_artifacts --summary-only
uv run -m experiments.medical_dataset_gen.utils.migrate_shared_embedding_artifacts --apply --strip-embedding-overrides
```

## Retrieval Pool

Schema v4 uses only `retrieval.pool_scope: query_local`: chunks linked to the query through `chunk_memberships.parquet`.

## Language Surfaces

Chunk and query text is generated deterministically from [medical_ontology.yaml](./data_templates/medical_ontology.yaml) and [chunk_templates.yaml](./data_templates/chunk_templates.yaml). `generation.chunk_text_style` selects either `semantic_hardened`, which keeps chart-like evidence only, or `ontology_explicit`, which adds one of several authored interpretation sentences that explicitly names the active axis (for example, `care intensity`) and states its value bin. Chunk outer templates contain no admission-presentation evidence: they combine one condition anchor, cohort evidence, and one axis payload so another diagnostic modality cannot leak through a shared presentation scaffold. The v4 review script exhaustively renders every ontology payload across note styles, chunk styles, and surface groups before embeddings are run:

```bash
uv run python -m experiments.medical_dataset_gen.scripts.render_v4_language_review
```

## Pipeline Execution

Run stages through the orchestrator; individual stage modules are domain implementation files and are not supported as direct entry points.

```bash
uv run python -m experiments.medical_dataset_gen.pipeline --exp <experiment_name> --from chunks --to qrels
uv run python -m experiments.medical_dataset_gen.pipeline --exp <experiment_name> --stages embed,filter_queries,eval
uv run python -m experiments.medical_dataset_gen.pipeline --exp <experiment_name> --run "eval --steps evaluation_stats,evaluation_slice_stats"
uv run python -m experiments.medical_dataset_gen.pipeline --exp <experiment_name> --run "eval_plots --plots metrics_k_curves_for_lambda"
```

Selective evaluation, evaluation-plot, and geometry-plot reruns use `--run` so the main pipeline remains the only supported command surface.

## Documentation

The main code documentation lives in [docs/code_docs](./src/experiments/medical_dataset_gen/docs/code_docs):

- [pipeline_overview.md](./src/experiments/medical_dataset_gen/docs/code_docs/pipeline_overview.md)
- [configuration_and_artifacts.md](./src/experiments/medical_dataset_gen/docs/code_docs/configuration_and_artifacts.md)
- [generation_stages.md](./src/experiments/medical_dataset_gen/docs/code_docs/generation_stages.md)
- [deterministic_construction.md](./src/experiments/medical_dataset_gen/docs/code_docs/deterministic_construction.md)
- [retrieval_and_geometry.md](./src/experiments/medical_dataset_gen/docs/code_docs/retrieval_and_geometry.md)
- [evaluation_and_plots.md](./src/experiments/medical_dataset_gen/docs/code_docs/evaluation_and_plots.md)
