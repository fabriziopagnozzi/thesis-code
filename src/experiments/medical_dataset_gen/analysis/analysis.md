# Experiment Report Aggregation

## Unit of Aggregation

The main unit is an `experiment x k x strategy` row.

* `strategy_by_k.csv` keeps one row per experiment, retrieval budget `k`, and strategy.
* `comparison_by_k.csv` pivots those rows into one row per experiment and `k`, with TopK, MMR, and FacLoc metrics side by side.
* `metric_aggregate_summary.csv` aggregates the main metric deltas by metric and budget view.
* `budget_strategy_summary.csv` keeps one row per experiment for each budget category: `headline`, `medium_budget`, and `high_budget`.
* `headline_strategy_summary.csv` keeps one headline row per experiment. The headline row is the smallest `k` where all three strategies are available.

The headline row is intentionally conservative: it summarizes each experiment at the tightest complete budget, while the full per-`k` evidence remains in `comparison_by_k.csv` and `strategy_by_k.csv`. The medium-budget row uses the sorted `k` index `floor(len(k_values) / 2)`, and the high-budget row uses the largest available `k`.

The active report excludes experiments whose config has `retrieval.only_pass_geometry: false`, because all-query branches are no longer part of the primary thesis analysis.

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
* `FacLocVsMMR_FCPOutcome`: `facloc_better`, `tied`, or `facloc_worse`. The tie threshold is an absolute FCP delta of `0.05`.
* `AllFacetCleanRate@k`: fraction of queries where all four facets are covered and precision is at least `0.8`.

## Output Artifacts

### `_figures/*`

Matplotlib visualizations derived from the same CSV rows:

* `metrics/<metric>_<budget>_deltas_by_experiment.*`: FacLoc-minus-MMR and FacLoc-minus-TopK deltas in a two-column plot for each budget category and metric. Budgets are `headline`, `medium_budget`, and `high_budget`; metrics are `fcp`, `facet_coverage`, `all_facet_clean_rate`, `precision`, `recall`, and `alpha_ndcg`.
* In the delta plots, bars are color-coded and grouped by experiment family. Families are ordered by their mean FacLoc-minus-MMR value for the plotted metric; rows inside a family are ordered by the same value.
* `geometry_pass_rate_by_embedding.*`: geometry pass rate by experiment and embedding in a portrait horizontal bar plot, color-coded and grouped by experiment family.
* `lambda_stability_boxplot.*`: for `mmr` and `fac_loc`, shows the mean selected `lambda_norm` with one standard-deviation error bar across all selected `experiment x k` rows. `lambda_norm` maps the selected raw lambda onto that experiment's available lambda grid, with `0` at the smallest grid value and `1` at the largest grid value. Use this to see whether a strategy tends to select low, middle, or high lambdas, and how variable that selection is across experiments and budgets.
* `near_optimal_lambda_width.*`: for `mmr` and `fac_loc`, shows the distribution of `NearOptimalLambdaSpanNorm` across `experiment x k` lambda grids. For each grid, lambdas are near-optimal when their `FacetCoveragePurity@k` is within `--near-optimal-epsilon` of the best FCP for that strategy and `k`; the plotted value is the raw span from the smallest to largest near-optimal lambda divided by the full lambda-grid span. Values near `0` mean only a narrow lambda region is competitive, while values near `1` mean performance is flat across most of the grid.
* `dataset_composition_stacked.*`: portrait stacked composition plot for gold, near-miss, and background proportions. It shows one representative child per parent distribution because sibling children share the same generated data distribution. The stack colors encode pool component, while y-axis label colors encode experiment family.

Figures are for visual inspection; cite the CSV values for exact numbers.

### Human-readable text summaries
#### `report.md`
Human-readable overview with the main tables, generated with `tabulate`. It is good for quick inspection, but CSV files are the authoritative machine-readable outputs.

#### `report_interesting_findings.md`
Shorter diagnostic report focused on notable patterns: largest FacLoc gains, FacLoc worse/tied rows, low geometry pass rates, and embedding summaries.

### CSV Files

### `experiment_manifest.csv`

One row per discovered pass-filter experiment. It records config/load metadata, distribution ID, subexperiment label, embedding model/dimension, `OnlyPassGeometry`, human-readable `QueryScope`, configured `k` values, evaluation mode, and key artifact paths.
Use this to audit what experiments were included. `OnlyPassGeometry` remains the machine-readable boolean; `QueryScope` is the report-facing label. `ExperimentFamily` and `ExperimentFamilyLabel` come from `_exp_family.yaml` metadata stored in the experiment directory, normally at the parent distribution level.

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

This is the main comparison table for method behavior across retrieval budgets. It includes deltas for FCP, facet coverage, AllFacetCleanRate, precision, recall, and alpha-nDCG.

Use this for claims like "FacLoc beats MMR for most experiment-k rows" or for finding budgets where FacLoc becomes worse/tied.

### `metric_aggregate_summary.csv`

One row per metric and budget view. It summarizes FacLoc-minus-MMR and baseline deltas for `FCP`, `FacetCoverage@k`, `AllFacetCleanRate@k`, `Precision@k`, `Recall@k`, and `alpha-nDCG@k`.

Key columns:

* `Metric`
* `BudgetView`
* `Rows`
* `FacLocBetterPct`, `FacLocTiedPct`, `FacLocWorsePct`
* `FacLocTopKBetterPct`, `MMRTopKBetterPct`
* `MeanDeltaFacLocMMR`
* `MeanDeltaFacLocTopK`, `MeanDeltaMMRTopK`

Use this for thesis tables that compare aggregate behavior across metrics. The percentage columns are normalized by `Rows`; the FacLoc-vs-MMR win/tie/loss percentages use the report tie threshold `TieEpsilon`.

### `experiment_family_budget_summary.csv`

One row per experiment family and budget category. It aggregates the same representative rows used by `budget_strategy_summary.csv`, grouping by `ExperimentFamilyLabel` and `BudgetCategoryLabel`.

Use this for thesis tables that need to separate distribution-family effects from retrieval-budget effects. It preserves the family-level win/tie/loss percentages and FacLoc-minus-MMR/TopK deltas, but reports them separately for headline, medium-budget, and high-budget views.

### `metric_family_summary.csv`

One row per evaluation metric and experiment family. It aggregates all experiment-$k$ rows within each family and reports FacLoc-vs-MMR win/tie/loss percentages, FacLoc-vs-TopK win percentages, and mean FacLoc-minus-MMR/TopK deltas.

Use this for aggregate thesis tables that compare whether the family-level pattern is specific to \FCP{} or also appears in the secondary metrics.

### `metric_family_budget_summary.csv`

One row per evaluation metric, experiment family, and budget category. It uses the same representative headline, medium-budget, and high-budget rows as `budget_strategy_summary.csv`, then groups them by `Metric`, `ExperimentFamilyLabel`, and `BudgetCategoryLabel`.

Use this for aggregate thesis tables that jointly condition on metric, data-distribution family, and retrieval budget. The \FCP{} subset corresponds to the budget-resolved family table used to support the data-distribution claim.

### `budget_strategy_summary.csv`

One row per experiment and budget category. It selects the lowest complete `k` for `headline`, the median-index `k` for `medium_budget`, and the largest complete `k` for `high_budget`.

Use this for compact thesis figures that compare the same semantic budget category across experiments with different raw `k` grids.

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

It reports run counts, embedding dimensions, geometry pass rates, headline FCP means, FacLoc deltas, and active pass-filter run counts.

Use this for broad embedding-level trends. Treat it as descriptive rather than strictly paired, because different embeddings can yield different geometry-pass query sets.
