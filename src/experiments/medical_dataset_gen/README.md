# Medical Dataset Generator

This package builds a deterministic synthetic clinical benchmark for multi-aspect retrieval. Each query compares two patient cohorts across two clinical axes, so a complete answer requires evidence from four distinct facets. The pipeline renders the evidence, preserves the hidden facet mapping as qrels, and compares standard top-$k$, MMR, and facility-location retrieval on the same query-local pool.

The construction is designed to be inspectable. The clinical ontology and language templates are authored YAML resources, experiment choices live in a validated configuration file, and every stage writes a persisted artifact that can be examined independently. The benchmark is intended to test whether coverage-oriented retrieval helps under these controlled conditions; it does not assume that any method will win.

## Setup

Run commands from the repository root. The project requires Python 3.12 or newer, `uv`, and Linux. The reference configuration uses CUDA for embedding, so a GPU is strongly recommended for a complete experiment. A CPU run is useful for a small construction check but is not a practical substitute for a full embedding run.

```bash
uv sync
```

The pipeline stores run-time configuration and outputs below `src/experiments/medical_dataset_gen/_results/`. Those files are intentionally excluded from source control: an experiment directory captures one concrete run without changing the package itself.

## Create an experiment

Start with the tracked configuration reference and give the experiment a descriptive name:

```bash
EXP=demo
mkdir -p "src/experiments/medical_dataset_gen/_results/$EXP"
cp src/experiments/medical_dataset_gen/config_example.yaml \
  "src/experiments/medical_dataset_gen/_results/$EXP/_config.yaml"
```

Edit `_config.yaml` before running it. In particular, set the seed, embedding device, model, retrieval grid, and the intended chunk-pool distribution. The configuration is strict: misspelled fields and incompatible retrieval-pool settings fail early rather than being silently ignored.

For a first construction check, reduce `global.conditions`, set a small `generation.query_limit`, and use a modest embedding batch size. Run the structural stages first:

```bash
uv run python -m experiments.medical_dataset_gen.pipeline \
  --exp "$EXP" \
  --to qrels
```

This produces plans, clinical facts, chunks, queries, canonical answers, and qrels without downloading or running an embedding model. Inspect these artifacts before starting the computationally heavier stages. Once the configuration and generated text look right, run the complete workflow:

```bash
uv run python -m experiments.medical_dataset_gen.pipeline --exp "$EXP"
```

The complete workflow evaluates the configured retrieval strategies, writes aggregate statistics, and produces the standard evaluation and query-geometry figures.

## Pipeline stages

The default command executes the following stages in order.

| Stage | Purpose | Principal outputs |
| --- | --- | --- |
| `plans` | Enumerate ontology-compliant cohort and axis comparisons. | `query_plans.parquet` |
| `facts` | Materialize gold evidence and controlled distractors. | `clinical_facts.parquet` |
| `chunks` | Render and validate clinical evidence documents and query-local memberships. | `chunk_documents.parquet`, `chunk_memberships.parquet` |
| `queries_answers` | Render query wording and canonical reference answers. | `queries.parquet`, `gold_answers.parquet` |
| `qrels` | Derive relevance labels from the hidden evidence structure. | `qrels.parquet` |
| `embed` | Encode chunks and queries; reuse compatible chunk embeddings when available. | embedding arrays and metadata |
| `filter_queries` | Record predeclared retrieval-geometry diagnostics for every query. | `geometry_stats.parquet`, `geometry_slice_stats.parquet` |
| `eval` | Score top-$k$, MMR, and facility location against facet-level relevance. | `evaluation_results.parquet`, evaluation statistics |
| `eval_plots` | Render aggregate retrieval and diagnostic figures. | evaluation figures |
| `geom_plots` | Render selected query-local embedding maps and retrieval overlays. | geometry figures and plot data |

The most useful audit path is usually `query_plans` → `clinical_facts` → `chunk_memberships` → `qrels`. These tables show how each query's evidence requirements become retrievable documents and relevance labels.

## Rerunning work safely

Use a contiguous range when downstream artifacts need to be refreshed:

```bash
uv run python -m experiments.medical_dataset_gen.pipeline \
  --exp "$EXP" \
  --from embed \
  --to eval
```

Use an explicit list for independent work, including a normal geometry-plot rerun:

```bash
uv run python -m experiments.medical_dataset_gen.pipeline \
  --exp "$EXP" \
  --stages geom_plots
```

Selective evaluation or plot work goes through the same entry point with `--run`:

```bash
uv run python -m experiments.medical_dataset_gen.pipeline \
  --exp "$EXP" \
  --run "eval --steps evaluation_stats,evaluation_slice_stats"

uv run python -m experiments.medical_dataset_gen.pipeline \
  --exp "$EXP" \
  --run "eval_plots --plots metrics_k_curves_for_lambda"

uv run python -m experiments.medical_dataset_gen.pipeline \
  --exp "$EXP" \
  --run "geom_plots --plots query_overview_4panel"
```

`--queries-only` is useful when only query wording has changed. It reuses existing chunk vectors for the same document surface and embedding setup, then writes fresh query vectors. It fails if the required chunk embeddings are absent or inconsistent.

## Comparative experiments

Use a parent experiment when several runs share the same generated dataset. Put the shared configuration in the parent `_config.yaml`, create one child directory per comparison, and place each child override in `_subconfig.yaml`.

```text
_results/
  comparison/
    _config.yaml
    bge_m3/
      _subconfig.yaml
    qwen_small/
      _subconfig.yaml
```

Run the parent name to launch its children sequentially:

```bash
uv run python -m experiments.medical_dataset_gen.pipeline --exp comparison
```

Keep the dataset-distribution settings in the parent. Children are intended for changes such as the embedding model, query wording, retrieval parameters, or evaluation options. Compatible generation and embedding artifacts are shared automatically; geometry, evaluation, figures, and logs remain local to each child. Pass `--parent` only when the parent itself should run despite having children.

## Key inputs and outputs

- [medical_ontology.yaml](./data_templates/medical_ontology.yaml) defines clinical conditions, cohort contrasts, axes, value bins, and pair-level generation policies.
- [chunk_templates.yaml](./data_templates/chunk_templates.yaml) and [query_answer_templates.yaml](./data_templates/query_answer_templates.yaml) define the authored language surfaces.
- [config_example.yaml](./config_example.yaml) is the complete configuration reference.
- The `embed` stage records model, prompts, normalization, dimensions, and artifact locations in embedding metadata.
- Logs and figures are written inside the experiment directory, alongside the tables that produced them.
