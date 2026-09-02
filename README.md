# Multi-aspect RAG benchmark

This repository contains reusable research code for studying multi-aspect retrieval in Retrieval-Augmented Generation (RAG). The main question is whether a retriever can select evidence from several distinct aspects of a clinical question when a simple similarity ranking tends to concentrate on only one of them.

The active implementation is `medical_dataset_gen`, a deterministic synthetic medical benchmark. The repository also contains an earlier MIMIC-IV-based pipeline, shared retrieval utilities, and a small vendored data viewer. The synthetic benchmark is the current research direction; the MIMIC-IV code is retained as historical context and for comparison with the earlier experiments.

## Research idea

The benchmark is designed around a specific, testable hypothesis:

> Facility-location retrieval should provide better multi-aspect evidence coverage than standard top-k retrieval and dispersion-oriented MMR when the candidate pool contains separable clinical evidence clusters.

Each synthetic query compares two cohorts across two clinical axes. This creates four equally relevant facets:

```text
                 clinical axis A       clinical axis B
cohort A         gold facet 1           gold facet 2
cohort B         gold facet 3           gold facet 4
```

The query-local candidate pool contains those gold facets together with configurable near-miss distractors and background outlier clusters. Top-k provides the relevance baseline, MMR trades relevance against redundancy, and facility location trades relevance against coverage of the candidate pool. The evaluation records both ordinary relevance metrics and facet-level coverage metrics.

The benchmark is synthetic by design. Earlier experiments built directly from MIMIC-IV discharge notes produced noisy, patient-specific pools and weak results. The current generator gives explicit control over ontology policies, facet membership, distractor placement, surface wording, and gold answers.

## The current benchmark pipeline

The main entry point is [`src/experiments/medical_dataset_gen/pipeline/__main__.py`](src/experiments/medical_dataset_gen/pipeline/__main__.py). Its stage order is registered in [`pipeline/stages.py`](src/experiments/medical_dataset_gen/pipeline/stages.py):

```text
ontology + configuration
          ↓
plans → facts → chunks and memberships → queries and answers → qrels
                                                               ↓
                                           embeddings → geometry diagnostics → evaluation
                                                               ↓
                                                   evaluation and geometry plots
```

The construction graph and rendered language are kept separate. Structured facts define which facet a document belongs to; templates turn those facts into clinical prose; qrels are projected from the hidden memberships instead of being inferred from the prose.

| Stage | Main implementation | Responsibility | Principal outputs |
| --- | --- | --- | --- |
| `plans` | [`dataset_generation/planning.py`](src/experiments/medical_dataset_gen/dataset_generation/planning.py) | Enumerate conditions, cohort contrasts, axis pairs, ontology profiles, and permitted primary-axis orientations. | `query_plans.parquet` |
| `facts` | [`dataset_generation/facts.py`](src/experiments/medical_dataset_gen/dataset_generation/facts.py) | Materialize typed gold facts, facet-attached distractors, and background outlier clusters. | `clinical_facts.parquet` |
| `chunks` | [`dataset_generation/chunk_materialization.py`](src/experiments/medical_dataset_gen/dataset_generation/chunk_materialization.py), [`chunk_rendering.py`](src/experiments/medical_dataset_gen/dataset_generation/chunk_rendering.py) | Render and validate document text, remove query-local duplicates, assign stable IDs, and record memberships. | `chunk_documents.parquet`, `chunk_memberships.parquet` |
| `queries_answers` | [`dataset_generation/queries_answers.py`](src/experiments/medical_dataset_gen/dataset_generation/queries_answers.py) | Render the query surface and derive canonical answers from the structured gold facts. | `queries.parquet`, `gold_answers.parquet` |
| `qrels` | [`dataset_generation/qrels.py`](src/experiments/medical_dataset_gen/dataset_generation/qrels.py) | Convert structured memberships into binary relevance labels while preserving facet and distractor provenance. | `qrels.parquet` |
| `embed` | [`embedding/stage.py`](src/experiments/medical_dataset_gen/embedding/stage.py) | Encode document and query text, validate ID alignment, and record embedding provenance. | `.npy` ID/vector arrays and metadata |
| `filter_queries` | [`query_geometry/filtering.py`](src/experiments/medical_dataset_gen/query_geometry/filtering.py) | Measure whether each query has the intended local retrieval geometry and attach a diagnostic pass flag. | geometry statistics and slice tables |
| `eval` | [`evaluation/stage.py`](src/experiments/medical_dataset_gen/evaluation/stage.py) | Compare top-k, MMR, and facility-location retrieval over the configured local pools. | evaluation results, aggregate statistics, and slices |
| `eval_plots`, `geom_plots` | `evaluation/plot_stage.py`, `query_geometry/plot_stage.py` | Produce secondary figures and query-level embedding diagnostics. | figures and plot data |

The geometry filter is only a diagnostic annotation. Evaluation retains the full generated query population and reports geometry-eligible and geometry-ineligible slices separately.

## What is generated

The authored source data is in [`data_templates/medical_ontology.yaml`](src/experiments/medical_dataset_gen/data_templates/medical_ontology.yaml), [`chunk_templates.yaml`](src/experiments/medical_dataset_gen/data_templates/chunk_templates.yaml), and [`query_answer_templates.yaml`](src/experiments/medical_dataset_gen/data_templates/query_answer_templates.yaml).

The ontology defines:

- clinical conditions and their axis-specific value bins;
- six clinical axes: treatment duration, rehabilitation outcome, complication burden, acute clinical course, care intensity, and diagnostic evidence type;
- demographic and comorbidity cohort contrasts;
- joint axis-pair profiles and policies that decide which axis can be primary;
- wording and surface terms used by the query and chunk templates.

The current schema construction uses query-local pools and normally gives every query four relevant facets. The frozen `thesis_v5` paper suite defines 41 evidence-space distributions crossed with four primary wording profiles, plus four proportional-budget cells, for 168 cells in total. Each distribution contains 5,304 deterministically generated queries, including 2,636 held-out test queries. Counts for standalone configurations can differ when the ontology, condition limit, excluded axes, or query limit changes.

The key structured identifiers are defined in [`dataset_generation/schemas.py`](src/experiments/medical_dataset_gen/dataset_generation/schemas.py):

- `evidence_profile_id` identifies the logical condition, cohort contrast, axis pair, value bins, and answer truth;
- `query_id` identifies one permitted query orientation and surface;
- `pool_id` identifies its query-local candidate-pool view;
- `facet_id`, `fact_id`, and `chunk_id` preserve evidence provenance through generation and evaluation.

The validation/test split is assigned deterministically during planning. There is no training split because the benchmark tunes retrieval hyperparameters and reports held-out test results; it is not training a retrieval model.

## Repository structure

```text
.
├── README.md
├── pyproject.toml                # dependencies, Ruff/Pyright, and task shortcuts
├── uv.lock                       # reproducible dependency resolution
├── setup.sh                      # optional systemd resource-managed command wrapper
├── src/
│   ├── experiments/
│   │   ├── medical_dataset_gen/   # active synthetic benchmark
│   │   │   ├── data_templates/   # ontology and authored language templates
│   │   │   ├── dataset_generation/
│   │   │   ├── embedding/
│   │   │   ├── evaluation/
│   │   │   ├── pipeline/
│   │   │   ├── query_geometry/
│   │   │   ├── reports/
│   │   │   ├── retrieval/
│   │   │   ├── scripts/
│   │   │   └── utils/
│   │   └── mimic/                 # earlier MIMIC-IV experiment pipeline
│   ├── helpers/                   # shared embedding, retrieval, metrics, and paths
│   └── thirdparty/                # optional vendored LanceDB viewer
└── .vscode/                       # editor settings examples
```

For the active pipeline, a useful reading order is:

1. [`pipeline/__main__.py`](src/experiments/medical_dataset_gen/pipeline/__main__.py) and [`pipeline/stages.py`](src/experiments/medical_dataset_gen/pipeline/stages.py) for orchestration;
2. [`utils/global_schemas.py`](src/experiments/medical_dataset_gen/utils/global_schemas.py) and [`utils/global_utils.py`](src/experiments/medical_dataset_gen/utils/global_utils.py) for configuration and artifact paths;
3. [`dataset_generation/schemas.py`](src/experiments/medical_dataset_gen/dataset_generation/schemas.py) and [`planning.py`](src/experiments/medical_dataset_gen/dataset_generation/planning.py) for the logical benchmark;
4. [`facts.py`](src/experiments/medical_dataset_gen/dataset_generation/facts.py), [`chunk_materialization.py`](src/experiments/medical_dataset_gen/dataset_generation/chunk_materialization.py), and [`queries_answers.py`](src/experiments/medical_dataset_gen/dataset_generation/queries_answers.py) for the construction-to-text boundary;
5. [`embedding/stage.py`](src/experiments/medical_dataset_gen/embedding/stage.py), [`query_geometry/filtering.py`](src/experiments/medical_dataset_gen/query_geometry/filtering.py), and [`evaluation/stage.py`](src/experiments/medical_dataset_gen/evaluation/stage.py) for the retrieval experiment.

The `reports/` package reads completed experiment artifacts and creates cross-experiment summaries for the accompanying research write-up. It is downstream reporting code, not part of the core synthetic construction.

## Installation and prerequisites

The project targets Python 3.12 or newer and Linux. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then run from the repository root:

```bash
uv sync
```

The complete synthetic pipeline uses `sentence-transformers` and usually needs a CUDA-capable GPU. A CPU configuration is useful for a small structural smoke run through `qrels`, but a full embedding and evaluation run can be slow and memory intensive. Some embedding or reranking models may also require the corresponding Hugging Face access setup.

`setup.sh` wraps commands in a user `systemd-run` unit with memory limits and worker defaults. The `uv run task ...` shortcuts use this wrapper. Direct `uv run python ...` commands are easier for local debugging and help output, but they do not add those resource controls.

## Run the medical benchmark

### Create an experiment

Experiment configurations live under the ignored `_results` directory:

```bash
EXP=demo
mkdir -p "src/experiments/medical_dataset_gen/_results/$EXP"
cp src/experiments/medical_dataset_gen/config_example.yaml \
  "src/experiments/medical_dataset_gen/_results/$EXP/_config.yaml"
```

Before running, optionally edit the values in your `_config.yaml`. The most important sections are:

- `generation.chunk_pools`, which controls gold facet sizes and distractor clusters;
- `embeddings`, including model, device, prompts, and normalization;
- `retrieval`, including the candidate-pool size, `k` values, strategies, and lambda grids;
- `evaluation.mode`, which controls exploratory versus held-out testing summaries.

The configuration is validated by the Pydantic models in [`utils/global_schemas.py`](src/experiments/medical_dataset_gen/utils/global_schemas.py). Unknown fields are rejected.

### Run a cheap structural check

For a first run, reduce `global.conditions`, set a small `generation.query_limit`, and run only the stages that do not load an embedding model:

```bash
uv run task pipeline \
  --exp "$EXP" \
  --to qrels
```

Inspect `query_plans.parquet`, `clinical_facts.parquet`, `chunk_memberships.parquet`, and `qrels.parquet` before committing to a full run. The most useful audit chain is:

```text
query_plans → clinical_facts → chunk_memberships → qrels
```

### Run the complete workflow

```bash
uv run task pipeline --exp "$EXP"
```

Useful controls are:

```bash
# Run a contiguous range.
uv run task pipeline --exp "$EXP" --from embed --to eval

# Run selected stages.
uv run task pipeline --exp "$EXP" --stages geom_plots
```

## Comparative experiments and artifact reuse
Parent/child experiments allow embedding, wording, or retrieval settings to be compared over the same generated distribution:

```text
src/experiments/medical_dataset_gen/_results/
└── comparison/
    ├── _config.yaml
    ├── bge_m3/
    │   └── _subconfig.yaml
    └── qwen_small/
        └── _subconfig.yaml
```

Run the parent name to launch its children sequentially:

```bash
uv run task pipeline --exp comparison
```

The parent configuration fixes the dataset distribution. Child subconfigs may change downstream choices such as the embedding model, wording mode, retrieval grid, or evaluation settings; they may not change generation semantics. With `global.use_shared: true`, invariant plans and facts are stored in a parent shared-generation store, compatible document/query embeddings in a shared-embedding store, and local evaluation, geometry, logs, and figures remain in each child’s schema-versioned directory.

## Artifact layout

The configuration files are kept at the experiment root, while versioned local artifacts are written below the schema-version directory:

```text
_results/<experiment>/
├── _config.yaml
├── _subconfig.yaml                 # child experiments only
└── <schema-version>/
    ├── query_plans.parquet         # or shared when reuse is enabled
    ├── clinical_facts.parquet
    ├── chunk_documents.parquet
    ├── chunk_memberships.parquet
    ├── queries.parquet
    ├── gold_answers.parquet
    ├── qrels.parquet
    ├── embeddings_*.npy
    ├── embeddings_metadata.json
    ├── geometry_*.parquet
    ├── evaluation_*.parquet
    ├── _logs/
    └── _figures/
```

## Reports and secondary tools

Once several experiments have completed, the report module discovers their persisted artifacts and writes aggregate tables, figures, validity analyses, and optional LaTeX outputs:

```bash
uv run task report \
  --results-dir src/experiments/medical_dataset_gen/_results \
  --output-dir ../thesis-documents/reports/experiment_comparison
```

For a thesis report that combines the native suite with every materialized derived embedding suite, use the base-suite selector. It scans `experiment_specs/`, follows derived specs whose source is `thesis_v5`, verifies their pinned source manifest, and deduplicates the shared native cells:

```bash
uv run task report \
  --suite-base thesis_v5 \
  --results-dir /mnt/d/thesis/last-results/_results \
  --output-dir ../thesis-documents/reports/experiment_comparison/thesis_v5_all_embeddings
```

Use `--suite-regex` with `--suite-base` to restrict the derived suites, for example `--suite-regex '^(thesis_v5_embedding_extension|thesis_v5_multi_mpnet_extension)$'`. The native `thesis_v5` cells remain included automatically.

By default, report output is written to `../thesis-documents/reports/experiment_comparison`. The thesis repository ignores everything under `reports/` except each report's `figures/` and `latex/` trees, so data, manifests, markdown, and warnings stay available locally without being committed. The `reports/` CLI also supports filtering experiments, selecting embedding models, refreshing only plots or LaTeX macros, and enabling optional lambda-transfer, paired-statistics, or validity analyses. The scripts under [`medical_dataset_gen/scripts`](src/experiments/medical_dataset_gen/scripts) provide smaller diagnostics and sample renderers.

## The earlier MIMIC-IV pipeline

[`src/experiments/mimic`](src/experiments/mimic) contains the previous data path based on MIMIC-IV discharge notes. It has separate modules for note chunking, query construction, embeddings, evaluation, and pool analysis. MIMIC-IV data is not included in this repository and must be obtained and configured separately; the code expects external dataset and third-party paths. The pipeline remains useful for reproducing historical experiments, but new benchmark design work belongs in `medical_dataset_gen`.

## Validation and development

The repository uses Ruff for formatting/linting, Pyright for static checks, and `uv.lock` for dependency reproducibility. The active benchmark has no large automated test suite; the preferred checks are targeted and proportional to the changed code:

```bash
uv run ruff check src/experiments/medical_dataset_gen
uv run ruff format --check src/experiments/medical_dataset_gen
uv run pyright src/experiments/medical_dataset_gen
uv run task pipeline --help
```

Generation is deterministic when the configuration seed, ontology, templates, and model settings are held fixed. Embedding metadata records the model, prompts, dimensions, normalization, and artifact counts. Evaluation must be interpreted together with the geometry diagnostics and the query construction metadata; a facility-location advantage is an empirical result to test, not a condition used to select the benchmark.
