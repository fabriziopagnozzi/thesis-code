# Medical Dataset Generation Pipeline Overview

This note explains the implemented end-to-end pipeline under [src/experiments/medical_dataset_gen](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen). It is written from the code that currently exists, not from the design sketch. The central design choice is that the benchmark is generated from hidden structure first, then rendered into text, embedded, filtered by geometry, and evaluated as a coverage-sensitive retrieval problem.

The current pipeline entrypoint is [run_pipeline.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/run_pipeline.py). A normal run has this shape:

```bash
uv run python -m experiments.medical_dataset_gen.run_pipeline --exp <experiment_name>
```

The experiment config must already exist at:

```text
src/experiments/medical_dataset_gen/_results/<experiment_name>/_config.yaml
```

The code does not copy or overwrite this file. `load_config()` reads the per-experiment YAML, validates it with Pydantic models in [global_configs.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/global_configs.py), and then sets `global.output_experiment` to the `--exp` value.

## The Stage Graph

`run_pipeline.py` defines the executed stages in the `STAGES` list:

| stage name | function | main output |
| --- | --- | --- |
| `plans` | `run_make_query_plans()` | `query_plans.parquet` |
| `calibrate_plans` | `run_calibrate_query_plans()` | `query_plans.parquet`, `query_plan_calibration.parquet` |
| `facts` | `run_make_facts()` | `clinical_facts.parquet` |
| `chunks` | `run_make_chunks()` | `chunk_documents.parquet`, `chunk_memberships.parquet`, `generation_rejects.parquet` |
| `queries_answers` | `run_make_queries_answers()` | `queries.parquet`, `gold_answers.parquet` |
| `qrels` | `run_make_qrels()` | `qrels.parquet` |
| `embed` | `run_embed()` | embedding memmap arrays and metadata |
| `geom_filter` | `run_filter_geometry()` | `geometry_stats.parquet` |
| `eval` | `run_evaluate()` | `evaluation_results.parquet`, `evaluation_stats.parquet` |
| `geom_plots` | `run_embedding_geometry()` | embedding geometry figures and diagnostic parquet files |
| `eval_plots` | `store_eval_figures()` | evaluation figures |

The stage boundaries are real artifact boundaries. Every downstream stage reads persisted parquet or `.npy` files rather than keeping an in-memory object graph from the previous stage.

## The Main Invariant

The pipeline keeps three layers separate:

1. Hidden structured design: query plans, facets, facts, roles, and labels.
2. Textual surface form: clinical note chunks and natural-language queries.
3. Retrieval behavior: embeddings, candidate pools, selectors, and metrics.

That separation matters because the benchmark is not trying to infer qrels from generated prose. The qrels are projected from the hidden labels. The prose exists to create a realistic retrieval surface that still has known facet structure.

## Why The Pipeline Is Ordered This Way

The first stages define the answer geometry before final chunk generation. A query has exactly four planned facets in the current MVP shape: two subgroups crossed with two clinical axes. With `generation.dominance_mode: rotating`, one facet is selected by deterministic rotation. With `generation.dominance_mode: embedding_calibrated`, `calibrate_plans` keeps the natural symmetric query text, embeds neutral probe chunks for all four facets, chooses the facet that is naturally closest to the query, and only then marks that facet as overrepresented. Each query also receives hard distractors.

The later stages ask whether a retrieval method can recover this planned multi-facet evidence. The evaluation does not only ask whether selected chunks are relevant. It asks whether selected chunks cover distinct facets, avoid same-facet redundancy, and avoid hard negatives.

## Candidate Pools Are Configurable, But `query_local` Is The Current Focus

The retrieval code supports three pool scopes in [retrieval/utils.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/retrieval/utils.py):

- `query_local`: only chunk documents linked to the current query through `chunk_memberships.parquet`.
- `same_condition`: all chunk documents with the same `condition_id` as the query.
- `full_corpus`: every generated chunk document.

The current experimental focus is `query_local`. In that setting, the candidate pool is the intentionally generated local pool for the hidden query plan: gold chunks plus hard distractors. Off-condition chunks inside that pool are generated distractors, not broad-corpus retrieval misses.

## How A Query Moves Through The System

A single query begins as a `QueryPlan` row. That row contains the condition, the two compared subgroups, four hidden facets, a dominant facet id, and a JSON logical form. If calibrated dominance is enabled, `run_calibrate_query_plans()` rewrites the dominant facet id, facet roles, and target gold counts from probe embedding statistics while leaving the query wording neutral. `run_make_facts()` then expands each facet into multiple `ClinicalFact` rows, with more rows for the calibrated dominant facet than for the complementary facets. It also adds distractor facts by perturbing condition, subgroup, or both.

`run_make_chunks()` turns each fact into validated note text. Depending on config, this can be deterministic template text, LLM-generated text, or deterministic text rewritten by an LLM. It then normalizes repeated structural chunks into `chunk_documents.parquet` and stores query/facet links in `chunk_memberships.parquet`.

`run_make_queries_answers()` renders the natural-language query from the same plan and constructs a canonical answer from the gold fact rows. `run_make_qrels()` then derives binary relevance labels directly from membership `is_gold` values.

After that, `run_embed()` embeds unique chunk-document text and query text. `run_filter_geometry()` checks whether the resulting embedding geometry is suitable for a coverage benchmark. `run_evaluate()` runs top-k, MMR, and facility-location over the configured candidate pools and computes facet-aware metrics from query-local qrels.

## The Thesis-Relevant Mechanism

The benchmark is constructed so that a nearest-neighbor top-k list can be relevant but redundant. One facet is intentionally dominant, so high query similarity can select many chunks from that facet. A good coverage method should spend some retrieval budget on the other facets. Facility-location is expected to help because its marginal gain rewards candidates that cover parts of the candidate pool not already represented by the selected set.

This is why the core headline metrics are facet-oriented:

- `FacetCoverage@k`: whether every hidden answer facet appears at least once.
- `FacetMRR@k`: how early the first evidence for each facet appears.
- `alpha-nDCG@k`: whether ranked evidence covers facets early without excessive same-facet repetition.
- `DominantFacetRate` and `RedundantGoldRate`: whether a method is collapsing into repeated evidence.

## Related Technical Notes

- [deterministic_construction.md](/home/pagnozzi/thesis/docs/medical_dataset_gen/code_pipeline_explanation/deterministic_construction.md) explains the deterministic query/fact/template construction path.
- [configuration_and_artifacts.md](/home/pagnozzi/thesis/docs/medical_dataset_gen/code_pipeline_explanation/configuration_and_artifacts.md) documents every config block and persisted artifact.
- [generation_stages.md](/home/pagnozzi/thesis/docs/medical_dataset_gen/code_pipeline_explanation/generation_stages.md) explains the implemented plan, fact, chunk, query, answer, and qrel stages.
- [retrieval_and_geometry.md](/home/pagnozzi/thesis/docs/medical_dataset_gen/code_pipeline_explanation/retrieval_and_geometry.md) explains embeddings, candidate pools, selectors, and geometry filtering.
- [evaluation_and_plots.md](/home/pagnozzi/thesis/docs/medical_dataset_gen/code_pipeline_explanation/evaluation_and_plots.md) explains metric computation, summaries, and figures.
