# Multi-aspect retrieval for RAG

This repository contains the research code for a master's thesis on multi-aspect retrieval in Retrieval-Augmented Generation (RAG).

Most retrievers rank documents by how similar they are to a query. That works well when a question has one clear focus, but it can break down when the answer depends on several different pieces of evidence: the top results may all cover the same part of the question while leaving the other parts unanswered. This project asks whether a retriever can do a better job of covering all of those aspects.

The main implementation is `medical_dataset_gen`, a deterministic synthetic benchmark built around clinical questions. The repository also includes an earlier pipeline based on the MIMIC-IV dataset, shared retrieval utilities, and a small vendored data viewer.

## The idea behind the benchmark

The benchmark tests one main hypothesis:

> Facility-location retrieval should provide better multi-aspect evidence coverage than standard top-k retrieval and dispersion-oriented MMR when the candidate pool contains separable clinical evidence clusters.

Each query compares two patient cohorts along two clinical axes, giving it four equally relevant facets:

```text
                 clinical axis A       clinical axis B
cohort A         gold facet 1           gold facet 2
cohort B         gold facet 3           gold facet 4
```

For each query, the candidate pool mixes evidence for those four gold facets with configurable near-miss distractors and background outlier clusters. The three retrieval approaches make different trade-offs:

- **Top-k** is the relevance baseline: it simply returns the highest-scoring items.
- **MMR** balances relevance against redundancy.
- **Facility location** balances relevance against coverage of the candidate pool.

The evaluation measures both standard relevance and facet-level coverage.

## How the pipeline works

The main entry point is [`src/experiments/medical_dataset_gen/pipeline/__main__.py`](src/experiments/medical_dataset_gen/pipeline/__main__.py), and [`pipeline/stages.py`](src/experiments/medical_dataset_gen/pipeline/stages.py) defines the stage order. The same pipeline can be driven in two ways:

| Mode | Selector | Intended use |
| --- | --- | --- |
| Suite | `--suite` with `--cell` or `--where` | Reproduce the frozen thesis design from a materialized manifest. The suite fixes the distribution, wording profile, embedding configuration, artifact paths, and comparison metadata for every run. |
| Standalone | `--exp` | Run a custom schema-v5 configuration or a small experiment outside the thesis grid. |

Suite mode is the canonical path for the thesis results. A suite cell is one evidence-space distribution crossed with one run profile. The native `thesis_v5` suite therefore contains 41 distributions × 4 wording profiles = 164 cells. The three derived suites reuse those native datasets with different embedding models, producing 656 cells across all four suites.

```text
ontology + configuration
          ↓
plans → facts → chunks and memberships → queries and answers → qrels
                                                               ↓
                                           embeddings → geometry diagnostics → evaluation
                                                               ↓
                                                   evaluation and geometry plots
```

| Stage | Main implementation | Responsibility | Principal outputs |
| --- | --- | --- | --- |
| `plans` | [`dataset_generation/planning.py`](src/experiments/medical_dataset_gen/dataset_generation/planning.py) | Enumerates conditions, cohort contrasts, axis pairs, ontology profiles, and allowed primary-axis orientations. | `query_plans.parquet` |
| `facts` | [`dataset_generation/facts.py`](src/experiments/medical_dataset_gen/dataset_generation/facts.py) | Creates typed gold facts, globally configured near misses targeted at the gold facets, and background outlier clusters. | `clinical_facts.parquet` |
| `chunks` | [`dataset_generation/chunk_materialization.py`](src/experiments/medical_dataset_gen/dataset_generation/chunk_materialization.py), [`chunk_rendering.py`](src/experiments/medical_dataset_gen/dataset_generation/chunk_rendering.py) | Renders and validates document text, assigns stable semantic IDs, and records memberships. | `chunk_documents.parquet`, `chunk_memberships.parquet` |
| `queries_answers` | [`dataset_generation/queries_answers.py`](src/experiments/medical_dataset_gen/dataset_generation/queries_answers.py) | Renders the queries and derives canonical answers from the structured gold facts. | `queries.parquet`, `gold_answers.parquet` |
| `qrels` | [`dataset_generation/qrels.py`](src/experiments/medical_dataset_gen/dataset_generation/qrels.py) | Converts structured memberships into binary relevance labels while preserving facet and distractor provenance. | `qrels.parquet` |
| `embed` | [`embedding/stage.py`](src/experiments/medical_dataset_gen/embedding/stage.py) | Encodes documents and queries, checks ID alignment, and records embedding provenance. | `.npy` ID/vector arrays and metadata |
| `filter_queries` | [`query_geometry/filtering.py`](src/experiments/medical_dataset_gen/query_geometry/filtering.py) | Checks whether each query has the intended local retrieval geometry and attaches a diagnostic pass flag. | geometry statistics and slice tables |
| `eval` | [`evaluation/stage.py`](src/experiments/medical_dataset_gen/evaluation/stage.py) | Compares top-k, MMR, and facility-location retrieval over the configured local pools. | evaluation results, aggregate statistics, and slices |
| `eval_plots`, `geom_plots` | `evaluation/plot_stage.py`, `query_geometry/plot_stage.py` | Creates secondary figures and query-level embedding diagnostics. | figures and plot data |

## What the generator produces

The hand-authored source data lives in [`data_templates/medical_ontology.yaml`](src/experiments/medical_dataset_gen/data_templates/medical_ontology.yaml), [`chunk_templates.yaml`](src/experiments/medical_dataset_gen/data_templates/chunk_templates.yaml), and [`query_answer_templates.yaml`](src/experiments/medical_dataset_gen/data_templates/query_answer_templates.yaml).

The ontology defines:

- clinical conditions and their axis-specific value bins;
- six clinical axes: treatment duration, rehabilitation outcome, complication burden, acute clinical course, care intensity, and diagnostic evidence type;
- demographic and comorbidity cohort contrasts;
- joint axis-pair profiles and rules for deciding which axis can be primary;
- wording and surface terms used by the query and chunk templates.

The frozen `thesis_v5` suite combines 41 evidence-space distributions with four wording profiles, giving 164 configurations per embedding model. Together, the native suite and three derived embedding suites contain 656 completed cells. Each distribution has 5,304 deterministically generated queries, including 2,636 held-out test queries. Standalone configurations may have different counts if they change the ontology, condition limit, excluded axes, or query limit.

The main identifiers, defined in [`dataset_generation/schemas.py`](src/experiments/medical_dataset_gen/dataset_generation/schemas.py), keep provenance intact throughout the pipeline:

- `evidence_profile_id` captures the condition, cohort contrast, axis pair, value bins, and answer truth;
- `query_id` identifies an allowed query orientation and wording;
- `pool_id` identifies the query-local view of the candidate pool;
- `facet_id`, `fact_id`, and `chunk_id` preserve evidence provenance through generation and evaluation.

The validation/test split is assigned deterministically during planning. There is no training split: this benchmark tunes retrieval hyperparameters and reports held-out test results, but it does not train a retrieval model.

## Repository layout

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

The `reports/` package reads completed experiment artifacts and builds cross-experiment summaries for the thesis. It is downstream reporting code, not part of the synthetic data construction itself.

## Setup

The project targets Linux and Python 3.12 or newer. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then run this from the repository root:

```bash
uv sync
```

The full synthetic pipeline uses `sentence-transformers` and will usually need a CUDA-capable GPU. A CPU is enough for a small structural smoke test through `qrels`, but full embedding and evaluation runs can be slow and memory-intensive.

`setup.sh` runs commands in a user `systemd-run` unit and applies memory limits and default worker counts. The `uv run task ...` shortcuts go through this wrapper.

## Running the benchmark

### Run the frozen thesis suite

Suite specifications live in [`experiment_specs/`](src/experiments/medical_dataset_gen/experiment_specs). Before running cells, validate and materialize the specification:

```bash
uv run task suite_validate --suite thesis_v5
uv run task suite_materialize --suite thesis_v5
```

Each cell ID has the form `<distribution_id>__<run_profile_id>`. Use `--dry-run` to inspect a selection without executing stages:

```bash
uv run task pipeline \
  --suite thesis_v5 \
  --cell near_miss_h48__qwen_biased_simple \
  --dry-run
```

`--where` runs a batch selected from manifest metadata. Comma-separated clauses are combined as AND conditions and `|` gives alternatives within one clause. Selectors include `distribution_id`, `run_profile_id`, `family_id`, `analysis_tier`, `tag`, `analysis_block`, and the declared distribution/profile factors.

```bash
# Run both query structures for one distribution with category-explicit documents.
uv run task pipeline \
  --suite thesis_v5 \
  --where 'distribution_id=near_miss_h48,document_surface=category-explicit'
```

### Run a standalone configuration

Experiment configurations live under the ignored `_results` directory:

```bash
EXP=demo
mkdir -p "src/experiments/medical_dataset_gen/_results/$EXP"
cp src/experiments/medical_dataset_gen/config_example.yaml \
  "src/experiments/medical_dataset_gen/_results/$EXP/_config.yaml"
```

You can edit `_config.yaml` before the first run. The sections you are most likely to change are:

- `generation.chunk_pools` for gold-facet sizes and distractor clusters;
- `embeddings` for the model, device, prompts, and normalization;
- `retrieval` for candidate-pool size, `k` values, strategies, and lambda grids;
- `evaluation.mode` for exploratory or held-out test summaries.

The Pydantic models in [`utils/global_schemas.py`](src/experiments/medical_dataset_gen/utils/global_schemas.py) validate the configuration and reject unknown fields.

#### Start with a small structural check

For a quick first run, reduce `global.conditions`, set a small `generation.query_limit`, and stop before the stages that load an embedding model:

```bash
uv run task pipeline \
  --exp "$EXP" \
  --to qrels
```

#### Run the complete workflow

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

The pipeline skips compatible shared outputs and ready embedding artifacts. Use `--results-dir` to point either execution mode at an isolated result root.

## Comparing experiments and reusing artifacts

Parent/child experiments make it possible to compare embedding, wording, or retrieval settings over exactly the same generated distribution:

```text
src/experiments/medical_dataset_gen/_results/
└── comparison/
    ├── _config.yaml
    ├── bge_m3/
    │   └── _subconfig.yaml
    └── qwen_small/
        └── _subconfig.yaml
```

Run the parent experiment to launch each child in sequence:

```bash
uv run task pipeline --exp comparison
```

The parent configuration fixes the dataset distribution. Child subconfigs can change downstream choices such as the embedding model, wording mode, retrieval grid, or evaluation settings, but they cannot change generation semantics.

When `global.use_shared: true`, plans and facts that do not change are kept in the parent's shared-generation store. Compatible document and query embeddings go into a shared-embedding store, while each child keeps its own evaluation results, geometry diagnostics, logs, and figures in its `v5` directory.

## Artifact layout

Configuration files stay at the experiment root. The generated artifacts are written under the schema-v5 directory:

```text
_results/<experiment>/
├── _config.yaml
├── _subconfig.yaml                 # child experiments only
└── v5/
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

Materialized suites use a manifest-oriented layout instead:

```text
_results/v5/suites/<suite_id>/
├── suite_manifest.json
├── run_profiles/<run_profile_id>/resolved_run_profile.yaml
└── distributions/<distribution_id>/
    ├── resolved_distribution.yaml
    ├── data/schema-v5/                    # shared generation surfaces
    ├── shared_embeddings/                 # reusable document/query vectors
    └── runs/<run_profile_id>/
        ├── resolved_config.yaml
        └── attempts/initial/              # evaluation outputs and figures
```

The `attempts/initial` segment is retained in the frozen v5 artifact identity; named or additional evaluation attempts are not supported.

## Reports and smaller tools

After several experiments have finished, the report module finds their saved artifacts and produces aggregate tables, figures, validity analyses, and optional LaTeX output:

```bash
uv run task report \
  --results-dir <res_dir> \
  --output-dir <out_dir>
```

To build a thesis report that combines the native suite with every available derived embedding suite, use the base-suite selector. It scans `experiment_specs/`, follows derived specs sourced from `thesis_v5`, checks their pinned source manifests, and deduplicates the shared native cells:

```bash
uv run task report \
  --suite-base thesis_v5 \
  --results-dir <res_dir> \
  --output-dir <out_dir>
```