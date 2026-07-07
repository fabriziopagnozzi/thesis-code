# Experiment Report Aggregation

## Unit of Aggregation

The main unit is an `experiment x k x strategy` row.

* `strategy_by_k.csv` keeps one row per experiment, retrieval budget `k`, and strategy.
* `comparison_by_k.csv` pivots those rows into one row per experiment and `k`, with TopK, MMR, and FacLoc metrics side by side.
* `headline_strategy_summary.csv` keeps one headline row per experiment. The headline row is the smallest `k` where all three strategies are available.

The headline row is intentionally conservative: it summarizes each experiment at the tightest complete budget, while the full per-`k` evidence remains in `comparison_by_k.csv` and `strategy_by_k.csv`.

## Lambda Selection

For MMR and FacLoc, the report uses one selected lambda per `strategy x k`. Selection follows this order:

1. If `evaluation_stats.parquet` already contains held-out selected-lambda columns, those persisted selected rows are used directly. These rows have `SelectionSource = heldout_selected`.
2. If the artifact already contains only one row per `strategy x k`, it is treated as preselected. These rows have `SelectionSource = preselected`.
3. Otherwise, the report selects the best lambda post-hoc within each `strategy x k` grid by `FacetCoveragePurity@k`, using the experiment's lambda-selection tie-breaking config. These rows have `SelectionSource = posthoc_selected`.

`lambda_norm` is the selected lambda position normalized to the available grid range for that strategy and experiment. Use it when comparing MMR and FacLoc lambda stability, because their raw lambda grids have different ranges and objective scales.
lambda_norm = (selected_lambda - grid_start) / (grid_stop - grid_start) # basically percentage of the lambda space where the selected lambda lands

## Main Metrics

The report maximizes and compares `FacetCoveragePurity@k` (`FCP` in wide tables). This is the mean of the per-query product:

```text
FacetCoveragePurity@k = FacetCoverage@k * Precision@k
```

Important comparison columns:

* `Delta_FacLoc_MMR_FCP`: FacLoc FCP minus MMR FCP.
* `Delta_FacLoc_TopK_FCP`: FacLoc FCP minus TopK FCP.
* `FacLocVsMMR_FCPOutcome`: `facloc_better`, `tied`, or `facloc_worse`. The tie threshold is an absolute FCP delta of `0.01`.
* `AllFacetCleanRate@k`: fraction of queries where all four facets are covered and precision is at least `0.8`.

## Output Artifacts

### `_figures/*`

Matplotlib visualizations derived from the same CSV rows:

* `fcp_delta_by_experiment.*`: **headline** FacLoc-minus-MMR FCP deltas.
* `facloc_vs_topk_delta_by_experiment.*`: **headline** FacLoc-minus-TopK FCP deltas.
* `all_facet_clean_rate_by_experiment.*`: **headline** AllFacetCleanRate comparison.
* `geometry_pass_rate_by_embedding.*`: geometry pass rate grouped by embedding.
* `lambda_stability_boxplot.*`: for `mmr` and `fac_loc`, shows the mean selected `lambda_norm` with one standard-deviation error bar across all selected `experiment x k` rows. `lambda_norm` maps the selected raw lambda onto that experiment's available lambda grid, with `0` at the smallest grid value and `1` at the largest grid value. Use this to see whether a strategy tends to select low, middle, or high lambdas, and how variable that selection is across experiments and budgets.
* `near_optimal_lambda_width.*`: for `mmr` and `fac_loc`, shows the distribution of `NearOptimalLambdaSpanNorm` across `experiment x k` lambda grids. For each grid, lambdas are near-optimal when their `FacetCoveragePurity@k` is within `--near-optimal-epsilon` of the best FCP for that strategy and `k`; the plotted value is the raw span from the smallest to largest near-optimal lambda divided by the full lambda-grid span. Values near `0` mean only a narrow lambda region is competitive, while values near `1` mean performance is flat across most of the grid.
* `dataset_composition_stacked.*`: gold, near-miss, and background proportions.

Figures are for visual inspection; cite the CSV values for exact numbers.

### Human-readable text summaries
#### `report.md`
Human-readable overview with the main tables, generated with `tabulate`. It is good for quick inspection, but CSV files are the authoritative machine-readable outputs.

#### `interesting_findings.md`
Shorter diagnostic report focused on notable patterns: largest FacLoc gains, FacLoc worse/tied rows, low geometry pass rates, embedding summaries, and pass-only versus all-query comparisons.

### CSV Files

### `experiment_manifest.csv`

One row per discovered experiment. It records config/load metadata, distribution ID, subexperiment label, embedding model/dimension, `OnlyPassGeometry`, configured `k` values, evaluation mode, and key artifact paths.
Use this to audit what experiments were included and whether they were pass-only or all-query evaluations.

### `dataset_distribution.csv`

One row per experiment, aggregated from qrels by first computing per-query pool statistics and then reporting mean/min/max across queries.

Key columns:

* `PoolSizeMean`
* `GoldPercentage`
* `NearMissDistractorPercentage`
* `BackgroundOutlierPercentage`
* facet-role counts such as `DominantPrimaryGoldCountMean`, `OtherPrimaryGoldCountMean`, `SecondaryGoldCountMean`, and `NicheGoldCountMean`
* `PrimaryDominanceRatio`
* `DistributionCategory`

Use this to interpret each experiment's data-distribution.

### `geometry_filter_summary.csv`

One row per experiment, aggregated from `geometry_stats.parquet`.
It reports how many queries were checked by the geometry filter, how many passed, pass rate, top-k facet behavior, primary-axis dominance diagnostics, and the most common filter failure modes.
Use this to understand whether an embedding model and distribution produced the intended benchmark geometry before retrieval evaluation.

### `strategy_by_k.csv`

One row per experiment, strategy, and `k` after lambda selection.
For MMR and FacLoc, each row contains the selected `lam`, normalized `lambda_norm`, number of evaluated queries, and all evaluation metrics. For TopK, lambda columns are empty or not meaningful.
Use this when you need the full per-strategy evidence and selected-lambda details.

### `comparison_by_k.csv`

One row per experiment and `k`, with TopK, MMR, and FacLoc metrics pivoted side by side.

This is the main comparison table for method behavior across retrieval budgets. It includes deltas for FCP, facet coverage, AllFacetCleanRate, and precision.

Use this for claims like "FacLoc beats MMR for most experiment-k rows" or for finding budgets where FacLoc becomes worse/tied.

### `headline_strategy_summary.csv`

One row per experiment. It selects the smallest complete `k` for that experiment, where complete means TopK, MMR, and FacLoc are all present.

Use this for compact thesis tables where each experiment should contribute only one row. Do not use it to make claims about behavior at larger budgets; use `comparison_by_k.csv` for that.

### `lambda_stability.csv`

One row per diversifying strategy (`mmr`, `fac_loc`), aggregated over all selected strategy rows.

It reports selected-lambda mean/std/min/max/median/IQR, the same statistics for normalized lambda position, boundary-selection rate, and near-optimal lambda width summaries.

Use normalized lambda columns when comparing sensitivity across methods.

### `near_optimal_lambda_width.csv`

One row per experiment, strategy, and `k` when a lambda grid is available.

A lambda is near-optimal when its FCP is within `--near-optimal-epsilon` of the best FCP for that `strategy x k`. The default epsilon is `0.01`.

Key columns:

* `BestFCP`
* `WorstFCP`
* `FCPRange`
* `NearOptimalLambdaFraction`
* `NearOptimalLambdaSpanNorm`

Use this to see whether a method has a broad plateau of good lambdas or is sensitive to a narrow optimum.

### `embedding_model_summary.csv`

One row per embedding model, aggregated from experiment-level headline rows and geometry summaries.

It reports run counts, embedding dimensions, geometry pass rates, headline FCP means, FacLoc deltas, and counts of pass-only versus all-query runs.

Use this for broad embedding-level trends. Treat it as descriptive rather than strictly paired, because different embeddings can yield different geometry-pass query sets.

### `embedding_query_scope_pairs.csv`

Rows for distribution and embedding combinations that have both pass-only and all-query headline runs.

It reports pass-only metrics, all-query metrics, and all-minus-pass-only deltas for TopK, MMR, FacLoc, and FacLoc-minus-MMR FCP.

Use this to inspect how much the geometry filter changes the evaluation for the same distribution and embedding model.
