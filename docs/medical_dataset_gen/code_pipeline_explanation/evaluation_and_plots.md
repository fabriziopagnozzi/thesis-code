# Evaluation And Plotting

This note explains how retrieval results are scored and plotted in the implemented synthetic medical benchmark, and how the current `evaluation/` code is modularized by responsibility.

Source files:

- [evaluation/evaluate.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/evaluation/evaluate.py)
- [evaluation/evaluation_workers.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/evaluation/evaluation_workers.py)
- [evaluation/metrics_retrieval.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/evaluation/metrics_retrieval.py)
- [evaluation/metrics_answer.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/evaluation/metrics_answer.py)
- [evaluation/lambda_agreement.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/evaluation/lambda_agreement.py)
- [evaluation/plots.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/evaluation/plots.py)
- [evaluation/types.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/evaluation/types.py)
- [evaluation/utils.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/evaluation/utils.py)
- [retrieval/utils.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/retrieval/utils.py)
- [helpers/query_algorithms.py](/home/pagnozzi/thesis/src/helpers/query_algorithms.py)

## Evaluation Module Structure

The evaluation code is now split into focused modules:

- `evaluate.py`: the main stage entrypoint, query-level evaluation loop, summary-table aggregation, and the canonical `METRIC_NAME_TO_FIELD` registry that maps summary metric names to raw per-query result columns.
- `evaluation_workers.py`: worker initialization for multiprocessing evaluation, artifact loading, and construction of the typed worker-state object shared by `_evaluate_query()`.
- `metrics_retrieval.py`: retrieval-side scoring, including binary gold metrics, facet coverage metrics, diversified ranking metrics, and redundancy/distractor diagnostics.
- `metrics_answer.py`: answer-side ROUGE preprocessing, typed answer-reference containers, and a query-scoped cached `AnswerRougeScorer`.
- `lambda_agreement.py`: comparison of FacLoc/MMR lambda pairs across the aggregated metric space.
- `plots.py`: figure generation, with plotted metric groups selected by summary-metric name and resolved through `METRIC_NAME_TO_FIELD` rather than duplicating raw-column mappings.
- `types.py`: typed row shapes and worker-state definitions used across the evaluation code.
- `utils.py`: small shared helpers such as harmonic means, confidence intervals, and qrel-to-facet mapping helpers.

## Stage 8: Retrieval Evaluation

`run_evaluate()` creates:

- `evaluation_results.parquet`
- `evaluation_stats.parquet`
- `lambda_pair_agreement.parquet`

The relevant config fields are:

- `retrieval.pool_scope`
- `retrieval.candidate_pool_n`
- `retrieval.k_values`
- `retrieval.lambda_values`
- `retrieval.strategies`
- `retrieval.mmr_window`
- `retrieval.only_pass_geometry`

The stage-level entrypoint reads:

- `queries.parquet`
- `qrels.parquet`
- `geometry_stats.parquet`

The worker initializer then loads the full evaluation artifacts:

- `chunk_documents.parquet`
- `chunk_memberships.parquet`
- `queries.parquet`
- `gold_answers.parquet`
- `qrels.parquet`
- `geometry_stats.parquet`
- embedding arrays

It first verifies that the stored `geometry_stats.pool_scope` matches the current `retrieval.pool_scope`. This matters because candidate-pool scope changes the meaning of all retrieval metrics.

`run_evaluate()` orchestrates three outputs:

- `evaluation_results.parquet`: one row per `query_id x strategy x k x lam`
- `evaluation_stats.parquet`: aggregated means over queries
- `lambda_pair_agreement.parquet`: FacLoc/MMR lambda-pair similarity table

The current implementation evaluates queries through `_evaluate_queries()`, which can execute either in-process or through a multiprocessing worker pool. The worker state is initialized once in `evaluation_workers.py` and then reused across `_evaluate_query()` calls.

## Per-Query Evaluation Loop

For each query:

1. Skip the query if `retrieval.only_pass_geometry` is true and the query did not pass the geometry filter.
2. Read the typed query row, qrels, answer references, and retrieval index maps from the preloaded worker state.
3. Build the candidate pool from `retrieval.pool_scope`.
4. Keep the top `retrieval.candidate_pool_n` candidates by query similarity.
5. Compute the candidate-candidate similarity matrix.
6. Build a query-scoped answer-ROUGE scorer if `retrieval.compute_answer_rouge` is enabled.
7. Evaluate every configured strategy, k, and lambda combination.

Top-k is evaluated once per k with `lam = null`. MMR and facility-location are evaluated for every value in `retrieval.lambda_values`.

The row key for `evaluation_results.parquet` is effectively:

```text
query_id, strategy, k, lam, pool_scope
```

Each row also stores query type, condition id, split, pool size, gold/facet metrics, redundancy diagnostics, and embedding-selection diagnostics.

## Binary Gold Metrics

`metrics_retrieval.py` contains the retrieval-side scoring code. `_relevance_metrics()` treats every gold chunk as relevant, regardless of which facet it supports.

### `gold_precision`

```text
gold_precision = selected_gold / selected_total
```

This measures how many selected chunks are positive qrels.

### `gold_recall`

```text
gold_recall = selected_gold / total_gold_chunks
```

This is usually low when the query has many redundant gold chunks and k is small. It is retained as a conventional metric, but it is not the main thesis metric.

### `gold_f1`

The harmonic mean of `gold_precision` and `gold_recall`.

### `average_precision_at_k`

The implementation scans the selected ranked list. Every time it sees a gold chunk, it adds precision at that rank. It divides by:

```text
min(total_gold_chunks, k)
```

This is a conventional ranking metric. It still does not know whether selected gold chunks cover distinct facets.

## Facet Coverage Metrics

`_facet_coverage_metrics()` is the main benchmark-specific scoring block.

### `facet_coverage`

```text
facet_coverage = number_of_facets_with_at_least_one_selected_gold / total_facets
```

This is the main coverage metric. A query has four planned facets, so `facet_coverage = 1.0` means all four answer facets were represented at least once.

### `weighted_facet_coverage`

This is mean per-facet recall:

```text
mean over facets of selected_gold_for_facet / total_gold_for_facet
```

This gives credit for retrieving more evidence within a facet, but the thesis-relevant first-order goal is still hitting distinct facets.

### `facet_hit_density`

```text
facet_hit_density = facets_hit / selected_total
```

This measures how efficiently the selected set spends retrieval budget on new facets.

### `facet_f1`

The harmonic mean of `facet_hit_density` and `facet_coverage`.

## Diversified Ranking Metrics

`_diversified_ranking_metrics()` computes ranking-sensitive metrics over facet labels.

### `facet_mrr_at_k`

For every facet, find the rank of the first selected gold chunk supporting that facet. Missing facets contribute zero:

```text
facet_mrr_at_k = mean_facet 1 / first_rank(facet)
```

This rewards methods that expose each answer facet early in the selected context.

### `alpha_ndcg`

`_alpha_ndcg()` uses `facet_id` as the subtopic label. Repeated chunks from the same facet receive diminishing gain:

```text
gain_at_rank = (1 - alpha) ** previous_count_for_that_facet
discount = log2(rank + 1)
```

The current code sets:

```python
ALPHA_NDCG_REDUNDANCY = 0.5
```

The ideal ranking is constructed greedily from the available facet counts, always choosing the facet with the largest remaining novelty gain. This gives a normalized score between the selected ranking and the best possible facet-diverse ranking at the same k.

## Redundancy And Distractor Metrics

`_redundancy_metrics()` explains why a method is succeeding or failing.

### `distractor_rate`

```text
distractor_rate = selected_non_gold / selected_total
```

Lower is better. This is critical for interpreting MMR: a method can be diverse by selecting irrelevant distractors.

The current code also exposes:

- `near_miss_distractor_rate`: selected non-gold chunks from facet-like hard negatives.
- `background_outlier_rate`: selected non-gold chunks from the coherent background clinical island.

### `dominant_facet_rate`

```text
dominant_facet_rate = selected_chunks_from_planned_dominant_facet / selected_total
```

High values indicate collapse into the deliberately overrepresented facet.

### `max_facet_concentration`

The fraction of the selected set occupied by the most frequent selected gold facet.

### `redundant_gold_rate`

```text
redundant_gold_rate = max(selected_gold - facets_hit, 0) / selected_total
```

This counts gold chunks that are relevant but do not add a new facet. It is one of the clearest diagnostics for top-k redundancy.

### `n_unique_hadms`

The number of unique synthetic admission ids represented in the selected set. This is a patient/admission diversity diagnostic, not a headline metric.

## Embedding Diagnostics In Evaluation Rows

Every result row also includes `retrieval_diagnostics()`:

- `fac_cov_score`: facility-location coverage objective over the candidate similarity matrix.
- `avg_cos`: average query cosine similarity of selected chunks.
- `jaccard_vs_topk`: overlap with the top-k selected set at the same k.

These are not qrel metrics. They explain the behavior of the selection rule in embedding space.

## Summary Table

`stats_aggregated_results_df()` groups `evaluation_results.parquet` by:

```text
strategy, lam, k
```

It computes means over queries and writes `evaluation_stats.parquet` with display-oriented column names:

- `Precision@k`
- `Recall@k`
- `F1@k`
- `MAP@k`
- `MeanFacetHitRate@k`
- `MeanFacetRecall@k`
- `FacetMRR@k`
- `alpha-nDCG@k`
- `DistractorRate`
- `NearMissDistractorRate`
- `BackgroundOutlierRate`
- `DominantFacetRate`
- `RedundantGoldRate`
- `fac`
- `avg_cos`
- `jac`
- optional answer-ROUGE summary metrics when enabled

If `retrieval.compute_answer_rouge` is set to `false` in the experiment config, the evaluator skips all answer-ROUGE computation and the ROUGE-specific columns and figures are omitted.

For top-k, `lam` is null. For MMR and facility-location, every configured lambda has its own summary row.

## Lambda-Pair Agreement Table

`lambda_pair_agreement.parquet` compares `fac_loc` and `mmr` over the full cartesian
product of lambda pairs within each `k`.

The row key is:

```text
k, fac_loc_lam, mmr_lam
```

Each row stores:

- one `abs_diff__...` column per included summary metric
- `mean_abs_diff`, the arithmetic mean across all included absolute-difference columns
- `weighted_mean_abs_diff`, the kernel-weighted agreement score used by the current heatmap
- `rank_within_k`, where rank 1 is the closest overall FacLoc/MMR lambda pair at that `k`
- `weighted_rank_within_k`, where rank 1 is the closest pair after benchmark-interest weighting

The agreement metric set includes the current summary metrics except identifiers and `n_queries`. When answer-ROUGE metrics are present in `evaluation_stats.parquet`, their absolute differences are included too.

The optional `evaluation.fac_loc_mmr_comparison_kernels` config block further annotates
each lambda pair with benchmark-interest signals such as gain vs top-k and paired 95%
lower-bound gain. The current default applies this only to `MeanFacetHitRate@k` and
excludes lambda pairs above `0.80` from the FacLoc/MMR cartesian product. The weighted
score divides raw disagreement by the pair-quality kernel raised to `agreement_alpha`,
with `kernel_floor` bounding the maximum penalty. The heatmap uses a logarithmic color
scale so these deliberately aggressive multiplicative penalties remain visible.

## Stage 10: Evaluation Plots

`store_eval_figures()` reads `evaluation_stats.parquet` and `evaluation_results.parquet`. If either table is missing or empty, plotting is skipped. Figures are written under:

```text
_results/<exp>/_figures/evaluation/
```

The plotting code has two important policies:

1. Top-k is the baseline.
2. For many comparison plots, each diversity method uses a coverage-first `lambda*` path.

The lambda selection policy is:

```text
choose max mean MeanFacetHitRate@k within strategy x k;
ties prefer higher Precision@k;
then lower DistractorRate;
then higher alpha-nDCG@k when available.
```

This policy is encoded through `_PRIMARY_SORT` and `_PRIMARY_DESC` in [evaluation/plots.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/evaluation/plots.py).

The plotted metric groups in `plots.py` are now specified as lists of summary metric names, such as `PLOTTED_MAIN_METRIC_NAMES` and `PLOTTED_DIAGNOSTIC_METRIC_NAMES`. The file resolves each summary name to its raw per-query column and optimization direction through `METRIC_NAME_TO_FIELD` imported from `evaluate.py`. This keeps the summary-metric registry centralized and avoids repeating raw-column mappings across the plotting layer.

## Figure Types

### `strategy_comparison.png`

This is a metric-vs-k grid. Top-k is plotted directly. MMR and facility-location are plotted along their selected `lambda*` path, with lambda labels at each k.

The included primary metrics are:

- `MeanFacetHitRate@k`
- `MeanFacetRecall@k`
- `alpha-nDCG@k`
- `DistractorRate`
- `Recall@k`
- `AnswerROUGE2Recall@k`

That last metric appears only when `retrieval.compute_answer_rouge: true`.

### `strategy_comparison_heatmap.png`

This compares MMR and facility-location at every lambda, split into one column per lambda. It is useful for checking whether a result is robust or only appears after picking the best lambda.

### `lambda_sensitivity.png`

This shows how each metric changes as lambda changes. Each row is a metric, each column is a diversity strategy, and each line color identifies k. Dashed horizontal lines show the top-k reference at the same k.

### `per_query_distributions.png`

This uses violin plots to show per-query metric distributions at a selected k. The k is chosen from top-k by highest median facet coverage, then each diversity strategy uses its best coverage-first lambda at that k.

### `gain_over_topk.png`

This plots paired metric deltas against top-k for every strategy, k, and lambda. It includes paired 95% confidence intervals over query-level deltas.

For higher-is-better metrics, positive bars are favorable. For lower-is-better metrics such as `DistractorRate`, negative bars are favorable.

### `gain_over_topk_simple.png`

This is the simpler paired-delta view using only the coverage-first lambda path for each strategy and k. It is usually easier to read than the full lambda-sweep delta plot.

### `gain_over_topk_similar_lambda.png`

This plots one heatmap per `k` over the full `fac_loc lambda x mmr lambda` grid.
Each cell is the `weighted_mean_abs_diff` from `lambda_pair_agreement.parquet` when the
kernel columns are available, otherwise it falls back to `mean_abs_diff`. Lower values
mean the two strategies behave more similarly across the full summary metric set, with
extra emphasis on lambda pairs that improve the configured kernel metrics over top-k.
The best lambda pair in each panel is highlighted directly.

### `selection_diagnostics.png`

This plots diagnostic metrics that explain method behavior:

- `DistractorRate`
- `DominantFacetRate`
- `RedundantGoldRate`
- `fac`
- `avg_cos`
- `jac`

It is useful for distinguishing a true coverage improvement from a method that simply diverged from top-k.

## How To Read The Evaluation In This Benchmark

The primary thesis question is not whether a method retrieves more positive chunks overall. The query-local benchmark deliberately contains redundant positives. The important question is whether the selected context covers distinct planned answer facets without filling the budget with distractors.

A strong facility-location result should therefore look like this:

- higher `MeanFacetHitRate@k`
- higher `MeanFacetRecall@k`
- higher `alpha-nDCG@k`
- lower `RedundantGoldRate`
- lower or comparable `DistractorRate`
- lower `DominantFacetRate`
- some drop in `avg_cos` relative to top-k, because pure query similarity is no longer the only objective

An MMR result needs extra care. If MMR improves `MeanFacetHitRate@k` but also sharply increases `DistractorRate`, it may be dispersing into wrong clusters rather than covering the hidden answer facets.
