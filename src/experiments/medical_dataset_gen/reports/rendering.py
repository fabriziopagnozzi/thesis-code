from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from experiments.medical_dataset_gen.evaluation.lambda_selection import (
    LAMBDA_SELECTION_MAXIMIZING_METRIC,
)
from experiments.medical_dataset_gen.reports.analysis_constants import (
    REPORT_FILES,
    practical_effect_threshold,
)
from experiments.medical_dataset_gen.reports.helpers import (
    bullets,
    family_balanced_mean,
    float_or_none,
    numeric_values,
    section_with_table,
    sorted_rows,
)
from experiments.medical_dataset_gen.reports.models import CliArgs


def render_report(
    *,
    args: CliArgs,
    experiment_count: int,
    dataset_rows: Sequence[Mapping[str, object]],
    geometry_rows: Sequence[Mapping[str, object]],
    comparison_rows: Sequence[Mapping[str, object]],
    family_summary_rows: Sequence[Mapping[str, object]],
    family_budget_summary_rows: Sequence[Mapping[str, object]],
    metric_family_summary_rows: Sequence[Mapping[str, object]],
    metric_family_budget_summary_rows: Sequence[Mapping[str, object]],
    metric_summary_rows: Sequence[Mapping[str, object]],
    low_budget_rows: Sequence[Mapping[str, object]],
    lambda_rows: Sequence[Mapping[str, object]],
    lambda_safety_rows: Sequence[Mapping[str, object]],
    lambda_robustness_rows: Sequence[Mapping[str, object]],
    embedding_summary_rows: Sequence[Mapping[str, object]],
    embedding_metric_rows: Sequence[Mapping[str, object]],
    embedding_metric_range_rows: Sequence[Mapping[str, object]],
    embedding_geometry_rows: Sequence[Mapping[str, object]],
    model_grid_rows: Sequence[Mapping[str, object]],
    paired_config_suite_rows: Sequence[Mapping[str, object]],
    figures: Sequence[Path],
) -> str:
    lines: list[str] = [
        '# Medical Dataset Experiment Comparison',
        '',
        f'Generated at `{datetime.now(UTC).isoformat()}`.',
        '',
        'This report compares completed experiment folders discovered from persisted '
        '`evaluation_stats.parquet` artifacts. When an experiment already stores held-out '
        'selected-lambda rows, those rows are used directly. Otherwise, lambdas are selected '
        f'post-hoc by `{LAMBDA_SELECTION_MAXIMIZING_METRIC}` within each `strategy x k` grid.',
        '',
        'The low-budget row for each experiment is the smallest `k` with all three strategies '
        'available, which keeps the summary demanding while preserving the full per-k output in '
        '`data/comparison_by_k.csv` and `data/strategy_by_k.csv`.',
        'Budget-category summaries are also written for `low_budget`, `medium_budget`, and '
        '`high_budget`, using the lowest, median-index, and highest available `k` per experiment.',
        'A parent/child configuration recap for appendix use is written to '
        '`txt_experiments_config_recap.md`.',
        '',
        '## Run Scope',
        '',
        f'- Results dir: `{args.results_dir}`',
        f'- Output dir: `{args.output_dir}`',
        f'- Experiments discovered: `{experiment_count}`',
        f'- Scrapped experiments included: `{args.include_scrapped}`',
        f'- Near-optimal lambda epsilon: `{args.near_optimal_epsilon}`',
        '',
    ]

    lines.extend(
        section_with_table(
            'Low Budget FacetCoveragePurity',
            sorted_rows(low_budget_rows, 'Delta_FacLoc_MMR_FCP'),
            columns=[
                'ShortExperiment',
                'k',
                'EmbeddingModel',
                'QueryScope',
                'ExperimentFamilyLabel',
                'TopK_FCP',
                'MMR_FCP',
                'FacLoc_FCP',
                'Delta_FacLoc_MMR_FCP',
                'Delta_FacLoc_TopK_FCP',
                'FacLocVsMMR_FCPOutcome',
                'TopK_AllFacetCleanRate',
                'MMR_AllFacetCleanRate',
                'FacLoc_AllFacetCleanRate',
                'MMR_lambda',
                'FacLoc_lambda',
            ],
            tablefmt=args.tablefmt,
            max_rows=args.max_table_rows,
        )
    )
    lines.extend(
        section_with_table(
            'Where FacLoc Is Worse Or Tied With MMR',
            sorted_rows(
                [
                    row
                    for row in comparison_rows
                    if row.get('FacLocVsMMR_FCPOutcome') in {'facloc_worse', 'tied'}
                ],
                'Delta_FacLoc_MMR_FCP',
                descending=False,
            ),
            columns=[
                'ShortExperiment',
                'k',
                'EmbeddingModel',
                'ExperimentFamilyLabel',
                'TopK_FCP',
                'MMR_FCP',
                'FacLoc_FCP',
                'Delta_FacLoc_MMR_FCP',
                'MMR_AllFacetCleanRate',
                'FacLoc_AllFacetCleanRate',
                'FacLocVsMMR_FCPOutcome',
            ],
            tablefmt=args.tablefmt,
            max_rows=args.max_table_rows,
        )
    )
    lines.extend(
        section_with_table(
            'By Evaluation Metric and Retrieval Budget',
            metric_summary_rows,
            columns=[
                'Metric',
                'BudgetView',
                'Rows',
                'FacLocBetterPct',
                'FacLocTiedPct',
                'FacLocWorsePct',
                'FacLocTopKBetterPct',
                'MMRTopKBetterPct',
                'MeanDeltaFacLocMMR',
                'MeanDeltaFacLocTopK',
                'MeanDeltaMMRTopK',
            ],
            tablefmt=args.tablefmt,
            max_rows=len(metric_summary_rows),
        )
    )
    lines.extend(
        section_with_table(
            'Experiment Family Summary',
            family_summary_rows,
            columns=[
                'ExperimentFamilyLabel',
                'Rows',
                'FacLocBetterPct',
                'FacLocTiedPct',
                'FacLocWorsePct',
                'Delta_FacLoc_MMR_FCP_mean',
                'Delta_FacLoc_TopK_FCP_mean',
                'Delta_MMR_TopK_FCP_mean',
                'Delta_FacLoc_MMR_AllFacetCleanRate_mean',
            ],
            tablefmt=args.tablefmt,
            max_rows=args.max_table_rows,
        )
    )
    lines.extend(
        section_with_table(
            'Experiment Family By Budget Summary',
            family_budget_summary_rows,
            columns=[
                'ExperimentFamilyLabel',
                'BudgetCategoryLabel',
                'Rows',
                'FacLocBetterPct',
                'FacLocTiedPct',
                'FacLocWorsePct',
                'Delta_FacLoc_MMR_FCP_mean',
                'Delta_FacLoc_TopK_FCP_mean',
                'Delta_MMR_TopK_FCP_mean',
                'Delta_FacLoc_MMR_AllFacetCleanRate_mean',
            ],
            tablefmt=args.tablefmt,
            max_rows=len(family_budget_summary_rows),
        )
    )
    lines.extend(
        section_with_table(
            'By Evaluation Metric and Experiment Family',
            metric_family_summary_rows,
            columns=[
                'Metric',
                'ExperimentFamilyLabel',
                'Rows',
                'FacLocBetterPct',
                'FacLocTiedPct',
                'FacLocWorsePct',
                'FacLocTopKBetterPct',
                'MeanDeltaFacLocMMR',
                'MeanDeltaFacLocTopK',
                'MeanDeltaMMRTopK',
            ],
            tablefmt=args.tablefmt,
            max_rows=len(metric_family_summary_rows),
        )
    )
    lines.extend(
        section_with_table(
            'By Evaluation Metric, Experiment Family, and Retrieval Budget',
            metric_family_budget_summary_rows,
            columns=[
                'Metric',
                'ExperimentFamilyLabel',
                'BudgetCategoryLabel',
                'Rows',
                'FacLocBetterPct',
                'FacLocTiedPct',
                'FacLocWorsePct',
                'FacLocTopKBetterPct',
                'MeanDeltaFacLocMMR',
                'MeanDeltaFacLocTopK',
                'MeanDeltaMMRTopK',
            ],
            tablefmt=args.tablefmt,
            max_rows=len(metric_family_budget_summary_rows),
        )
    )
    lines.extend(
        section_with_table(
            'Dataset Distributions',
            dataset_rows,
            columns=[
                'ShortExperiment',
                'ExperimentFamilyLabel',
                'DistributionCategory',
                'PoolSizeMean',
                'GoldPercentage',
                'NearMissDistractorPercentage',
                'BackgroundOutlierPercentage',
                'DominantPrimaryGoldCountMean',
                'OtherPrimaryGoldCountMean',
                'SecondaryGoldCountMean',
                'NicheGoldCountMean',
                'HardDistractorCountMean',
            ],
            tablefmt=args.tablefmt,
            max_rows=args.max_table_rows,
        )
    )
    lines.extend(
        section_with_table(
            'Geometry Filter And Embeddings',
            geometry_rows,
            columns=[
                'ShortExperiment',
                'EmbeddingModel',
                'EmbeddingDimension',
                'GeometryQueries',
                'GeometryPassQueries',
                'GeometryPassRate',
                'NTopkRetrievedFacetsMean',
                'PrimaryAxisTopkFractionMean',
                'DominantPrimaryTopkFractionMean',
                'TopFailureModes',
            ],
            tablefmt=args.tablefmt,
            max_rows=args.max_table_rows,
        )
    )
    lines.extend(
        section_with_table(
            'Lambda Stability',
            lambda_rows,
            columns=[
                'strategy',
                'n_selected',
                'distinct_lambda_count',
                'selected_lambda_mean',
                'selected_lambda_std',
                'selected_lambda_norm_mean',
                'selected_lambda_norm_std',
                'boundary_selection_rate',
                'near_optimal_fraction_mean',
                'near_optimal_span_norm_mean',
            ],
            tablefmt=args.tablefmt,
            max_rows=args.max_table_rows,
        )
    )
    lines.extend(
        section_with_table(
            'Lambda Safety On Validation FCP',
            sorted_rows(lambda_safety_rows, 'WorstDeltaStrategyTopK_FCP', descending=False),
            columns=[
                'ShortExperiment',
                'k',
                'strategy',
                'EmbeddingModel',
                'SafeLambdaFraction',
                'WorstDeltaStrategyTopK_FCP',
                'MedianDeltaStrategyTopK_FCP',
                'BestDeltaStrategyTopK_FCP',
                'WorstLambda',
                'BestLambda',
            ],
            tablefmt=args.tablefmt,
            max_rows=args.max_table_rows,
        )
    )
    lines.extend(
        section_with_table(
            'Lambda Robustness Summary',
            lambda_robustness_rows,
            columns=[
                'Scope',
                'ScopeValue',
                'strategy',
                'Rows',
                'MeanWeighting',
                'MeanSafeLambdaFraction',
                'MedianSafeLambdaFraction',
                'AllGridSafeRows',
                'AllGridSafeRate',
                'MedianWorstDeltaStrategyTopK_FCP',
                'MinWorstDeltaStrategyTopK_FCP',
            ],
            tablefmt=args.tablefmt,
            max_rows=args.max_table_rows,
        )
    )
    lines.extend(
        section_with_table(
            'Embedding Model Summary',
            embedding_summary_rows,
            columns=[
                'EmbeddingModel',
                'Runs',
                'EmbeddingDimension_mean',
                'GeometryPassRate_mean',
                'GeometryPassQueries_mean',
                'TopK_FCP_mean',
                'MMR_FCP_mean',
                'FacLoc_FCP_mean',
                'Delta_FacLoc_MMR_FCP_mean',
                'PassFilterRuns',
            ],
            tablefmt=args.tablefmt,
            max_rows=args.max_table_rows,
        )
    )
    lines.extend(
        section_with_table(
            'Embedding Model Grid Coverage',
            model_grid_rows,
            columns=[
                'EmbeddingModelLabel',
                'ExpectedCells',
                'ObservedCells',
                'MissingCells',
                'Complete',
            ],
            tablefmt=args.tablefmt,
            max_rows=args.max_table_rows,
        )
    )
    lines.extend(
        section_with_table(
            'Model-First FCP Summary',
            [
                row
                for row in embedding_metric_rows
                if row.get('MetricLabel') == 'FCP' and row.get('Scope') == 'overall'
            ],
            columns=[
                'EmbeddingModelLabel',
                'Rows',
                'MeanTopK',
                'MeanMMR',
                'MeanFacLoc',
                'MeanDeltaFacLocMMR',
                'MeanDeltaFacLocTopK',
            ],
            tablefmt=args.tablefmt,
            max_rows=args.max_table_rows,
        )
    )
    lines.extend(
        section_with_table(
            'Cross-Model FCP Ranges',
            [
                row
                for row in embedding_metric_range_rows
                if row.get('MetricLabel') == 'FCP'
                and row.get('Scope') in {'overall', 'budget', 'family_budget'}
            ],
            columns=[
                'Scope',
                'ScopeValue',
                'ModelCount',
                'CompleteCrossing',
                'MeanDeltaFacLocMMR',
                'MeanDeltaFacLocMMRMinModel',
                'MeanDeltaFacLocMMRMaxModel',
            ],
            tablefmt=args.tablefmt,
            max_rows=args.max_table_rows,
        )
    )
    lines.extend(
        section_with_table(
            'Representation-Space Audit By Model',
            embedding_geometry_rows,
            columns=[
                'EmbeddingModelLabel',
                'Rows',
                'CoverageStressRate',
                'GoldNearMissMargin',
                'GoldBackgroundMargin',
            ],
            tablefmt=args.tablefmt,
            max_rows=args.max_table_rows,
        )
    )
    lines.extend(
        section_with_table(
            'Low-Budget Wording Configuration Paired FCP',
            [
                row
                for row in paired_config_suite_rows
                if row.get('BudgetCategory') == 'low_budget' and row.get('Scope') == 'Configuration'
            ],
            columns=[
                'WordingConfigLabel',
                'Distributions',
                'Runs',
                'MeanDeltaFacLocMMR',
                'CI95Low',
                'CI95High',
                'PracticalConclusion',
            ],
            tablefmt=args.tablefmt,
            max_rows=args.max_table_rows,
        )
    )
    lines.extend(
        [
            '## Output Files',
            '',
            *bullets(f'`{file_name}`' for file_name in REPORT_FILES),
        ]
    )
    if figures:
        lines.extend(
            [
                '',
                '## Figures',
                '',
                *bullets(f'`{path.relative_to(args.output_dir)}`' for path in figures),
            ]
        )
    return '\n'.join(lines) + '\n'


def render_interesting_findings(
    *,
    comparison_rows: Sequence[Mapping[str, object]],
    low_budget_rows: Sequence[Mapping[str, object]],
    family_summary_rows: Sequence[Mapping[str, object]],
    family_budget_summary_rows: Sequence[Mapping[str, object]],
    metric_family_summary_rows: Sequence[Mapping[str, object]],
    metric_family_budget_summary_rows: Sequence[Mapping[str, object]],
    metric_summary_rows: Sequence[Mapping[str, object]],
    geometry_rows: Sequence[Mapping[str, object]],
    lambda_rows: Sequence[Mapping[str, object]],
    lambda_safety_rows: Sequence[Mapping[str, object]],
    embedding_summary_rows: Sequence[Mapping[str, object]],
    tablefmt: str,
    max_table_rows: int,
) -> str:
    fcp_deltas = numeric_values(comparison_rows, 'Delta_FacLoc_MMR_FCP')
    topk_deltas = numeric_values(comparison_rows, 'Delta_FacLoc_TopK_FCP')
    complete_fcp_rows = [
        row for row in comparison_rows if float_or_none(row.get('Delta_FacLoc_MMR_FCP')) is not None
    ]
    facloc_better_rows = [
        row for row in complete_fcp_rows if row.get('FacLocVsMMR_FCPOutcome') == 'facloc_better'
    ]
    facloc_worse_rows = [
        row for row in complete_fcp_rows if row.get('FacLocVsMMR_FCPOutcome') == 'facloc_worse'
    ]
    facloc_tied_rows = [
        row for row in complete_fcp_rows if row.get('FacLocVsMMR_FCPOutcome') == 'tied'
    ]
    facloc_beats_topk = sum(
        (float_or_none(row.get('Delta_FacLoc_TopK_FCP')) or 0.0) > 0.0
        for row in comparison_rows
        if float_or_none(row.get('Delta_FacLoc_TopK_FCP')) is not None
    )

    lines: list[str] = ['# Interesting Findings', '']
    lines.append(
        f'- FacLoc beats MMR on `{LAMBDA_SELECTION_MAXIMIZING_METRIC}` in '
        f'{len(facloc_better_rows)}/{len(complete_fcp_rows)} experiment-k comparisons; '
        f'it is worse in `{len(facloc_worse_rows)}` and tied within '
        f'`±{practical_effect_threshold("FCP"):.3f}` FCP in `{len(facloc_tied_rows)}`.'
    )
    lines.append(
        f'- FacLoc beats top-k on `{LAMBDA_SELECTION_MAXIMIZING_METRIC}` in '
        f'{facloc_beats_topk}/{len(topk_deltas)} experiment-k comparisons.'
    )
    if fcp_deltas:
        lines.append(
            '- Family-balanced mean FacLoc - MMR FCP delta: '
            f'`{(family_balanced_mean(comparison_rows, "Delta_FacLoc_MMR_FCP") or 0.0):.4f}`; '
            f'median: `{statistics.median(fcp_deltas):.4f}`.'
        )
    if topk_deltas:
        lines.append(
            '- Family-balanced mean FacLoc - top-k FCP delta: '
            f'`{(family_balanced_mean(comparison_rows, "Delta_FacLoc_TopK_FCP") or 0.0):.4f}`; '
            f'median: `{statistics.median(topk_deltas):.4f}`.'
        )
    if family_summary_rows:
        strongest_family = family_summary_rows[0]
        weakest_family = family_summary_rows[-1]
        lines.append(
            '- Strongest family-level FacLoc - MMR FCP margin: '
            f'`{strongest_family.get("ExperimentFamilyLabel")}` '
            f'(`{(float_or_none(strongest_family.get("Delta_FacLoc_MMR_FCP_mean")) or 0.0):.4f}` mean). '
            'Weakest family-level margin: '
            f'`{weakest_family.get("ExperimentFamilyLabel")}` '
            f'(`{(float_or_none(weakest_family.get("Delta_FacLoc_MMR_FCP_mean")) or 0.0):.4f}` mean).'
        )
    facloc_worst_deltas = [
        value
        for value in (
            float_or_none(row.get('WorstDeltaStrategyTopK_FCP'))
            for row in lambda_safety_rows
            if row.get('strategy') == 'fac_loc'
        )
        if value is not None
    ]
    mmr_worst_deltas = [
        value
        for value in (
            float_or_none(row.get('WorstDeltaStrategyTopK_FCP'))
            for row in lambda_safety_rows
            if row.get('strategy') == 'mmr'
        )
        if value is not None
    ]
    if facloc_worst_deltas and mmr_worst_deltas:
        lines.append(
            '- Validation lambda-safety check: median worst-case FacLoc - top-k FCP delta '
            f'is `{statistics.median(facloc_worst_deltas):.4f}`, while the corresponding '
            f'MMR value is `{statistics.median(mmr_worst_deltas):.4f}`.'
        )

    lambda_std = {
        str(row.get('strategy')): float_or_none(row.get('selected_lambda_norm_std'))
        for row in lambda_rows
    }
    facloc_lambda_std = lambda_std.get('fac_loc')
    mmr_lambda_std = lambda_std.get('mmr')
    if facloc_lambda_std is not None and mmr_lambda_std is not None:
        less_sensitive = 'FacLoc' if facloc_lambda_std <= mmr_lambda_std else 'MMR'
        lines.append(
            f'- Normalized selected-lambda std is lower for `{less_sensitive}` in the '
            'aggregate lambda-stability table.'
        )

    lines.append('')
    lines.extend(
        section_with_table(
            'Largest FacLoc Over MMR Gains',
            sorted_rows(low_budget_rows, 'Delta_FacLoc_MMR_FCP', descending=True),
            columns=[
                'ShortExperiment',
                'k',
                'EmbeddingModel',
                'ExperimentFamilyLabel',
                'TopK_FCP',
                'MMR_FCP',
                'FacLoc_FCP',
                'Delta_FacLoc_MMR_FCP',
                'Delta_FacLoc_TopK_FCP',
            ],
            tablefmt=tablefmt,
            max_rows=max_table_rows,
        )
    )
    lines.extend(
        section_with_table(
            'By Evaluation Metric and Retrieval Budget',
            metric_summary_rows,
            columns=[
                'Metric',
                'BudgetView',
                'Rows',
                'FacLocBetterPct',
                'FacLocTiedPct',
                'FacLocWorsePct',
                'FacLocTopKBetterPct',
                'MeanDeltaFacLocMMR',
                'MeanDeltaFacLocTopK',
            ],
            tablefmt=tablefmt,
            max_rows=len(metric_summary_rows),
        )
    )
    lines.extend(
        section_with_table(
            'Experiment Family Summary',
            family_summary_rows,
            columns=[
                'ExperimentFamilyLabel',
                'Rows',
                'FacLocBetterPct',
                'FacLocTiedPct',
                'FacLocWorsePct',
                'Delta_FacLoc_MMR_FCP_mean',
                'Delta_FacLoc_TopK_FCP_mean',
                'Delta_MMR_TopK_FCP_mean',
            ],
            tablefmt=tablefmt,
            max_rows=max_table_rows,
        )
    )
    lines.extend(
        section_with_table(
            'Experiment Family By Budget Summary',
            family_budget_summary_rows,
            columns=[
                'ExperimentFamilyLabel',
                'BudgetCategoryLabel',
                'Rows',
                'FacLocBetterPct',
                'FacLocTiedPct',
                'FacLocWorsePct',
                'Delta_FacLoc_MMR_FCP_mean',
                'Delta_FacLoc_TopK_FCP_mean',
                'Delta_MMR_TopK_FCP_mean',
            ],
            tablefmt=tablefmt,
            max_rows=len(family_budget_summary_rows),
        )
    )
    lines.extend(
        section_with_table(
            'By Evaluation Metric and Experiment Family',
            metric_family_summary_rows,
            columns=[
                'Metric',
                'ExperimentFamilyLabel',
                'Rows',
                'FacLocBetterPct',
                'FacLocTiedPct',
                'FacLocWorsePct',
                'FacLocTopKBetterPct',
                'MeanDeltaFacLocMMR',
                'MeanDeltaFacLocTopK',
            ],
            tablefmt=tablefmt,
            max_rows=len(metric_family_summary_rows),
        )
    )
    lines.extend(
        section_with_table(
            'By Evaluation Metric, Experiment Family, and Retrieval Budget',
            metric_family_budget_summary_rows,
            columns=[
                'Metric',
                'ExperimentFamilyLabel',
                'BudgetCategoryLabel',
                'Rows',
                'FacLocBetterPct',
                'FacLocTiedPct',
                'FacLocWorsePct',
                'FacLocTopKBetterPct',
                'MeanDeltaFacLocMMR',
                'MeanDeltaFacLocTopK',
            ],
            tablefmt=tablefmt,
            max_rows=len(metric_family_budget_summary_rows),
        )
    )
    lines.extend(
        section_with_table(
            'FacLoc Worse Or Tied With MMR',
            sorted_rows(
                [
                    row
                    for row in comparison_rows
                    if row.get('FacLocVsMMR_FCPOutcome') in {'facloc_worse', 'tied'}
                ],
                'Delta_FacLoc_MMR_FCP',
                descending=False,
            ),
            columns=[
                'ShortExperiment',
                'k',
                'EmbeddingModel',
                'ExperimentFamilyLabel',
                'TopK_FCP',
                'MMR_FCP',
                'FacLoc_FCP',
                'Delta_FacLoc_MMR_FCP',
                'MMR_AllFacetCleanRate',
                'FacLoc_AllFacetCleanRate',
                'FacLocVsMMR_FCPOutcome',
            ],
            tablefmt=tablefmt,
            max_rows=max_table_rows,
        )
    )
    lines.extend(
        section_with_table(
            'Lowest Geometry Pass Rates',
            sorted_rows(geometry_rows, 'GeometryPassRate', descending=False),
            columns=[
                'ShortExperiment',
                'EmbeddingModel',
                'EmbeddingDimension',
                'GeometryQueries',
                'GeometryPassQueries',
                'GeometryPassRate',
                'TopFailureModes',
            ],
            tablefmt=tablefmt,
            max_rows=max_table_rows,
        )
    )
    lines.extend(
        section_with_table(
            'Embedding Summary',
            embedding_summary_rows,
            columns=[
                'EmbeddingModel',
                'Runs',
                'GeometryPassRate_mean',
                'GeometryPassRate_min',
                'GeometryPassRate_max',
                'Delta_FacLoc_MMR_FCP_mean',
                'PassFilterRuns',
            ],
            tablefmt=tablefmt,
            max_rows=max_table_rows,
        )
    )
    return '\n'.join(lines) + '\n'
