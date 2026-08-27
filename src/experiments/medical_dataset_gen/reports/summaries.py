from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import cast

from experiments.medical_dataset_gen.reports.analysis_constants import (
    DIVERSIFYING_STRATEGIES,
    STRATEGIES,
    DeltaMetricLabel,
    practical_effect_threshold,
)
from experiments.medical_dataset_gen.reports.helpers import (
    boundary_rate,
    delta_outcome,
    family_balanced_mean,
    family_balanced_rate,
    float_or_none,
    int_or_none,
    numeric_stats,
    numeric_values,
    quantile,
    sorted_rows,
    strategy_label,
    subtract,
    winner_for_metric,
)
from experiments.medical_dataset_gen.reports.models import BudgetCategory
from experiments.medical_dataset_gen.reports.report_config import (
    BUDGET_CATEGORIES,
    BUDGET_CATEGORY_LABELS,
    LOW_BUDGET_K,
    REPORT_METRIC_LABEL_TO_SPEC,
    REPORT_METRIC_LABELS,
    REPORT_METRIC_NAME_TO_LABEL,
    REPORT_METRIC_SPECS,
)


def comparison_by_k_rows(strategy_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str, str], dict[str, Mapping[str, object]]] = {}
    for row in strategy_rows:
        experiment = str(row.get('Experiment') or '')
        k = int_or_none(row.get('k'))
        strategy = row.get('strategy')
        if not experiment or k is None or strategy not in STRATEGIES:
            continue
        geometry_population = str(row.get('GeometryPopulation') or '')
        lambda_policy = str(row.get('LambdaPolicy') or '')
        grouped.setdefault((experiment, k, geometry_population, lambda_policy), {})[
            cast(str, strategy)
        ] = row

    rows: list[dict[str, object]] = []
    for (experiment, k, _geometry_population, _lambda_policy), by_strategy in sorted(
        grouped.items()
    ):
        first = next(iter(by_strategy.values()))
        out = {
            'Experiment': experiment,
            'ShortExperiment': first.get('ShortExperiment'),
            'Distribution': first.get('Distribution'),
            'ShortDistribution': first.get('ShortDistribution'),
            'ExperimentFamily': first.get('ExperimentFamily'),
            'ExperimentFamilyLabel': first.get('ExperimentFamilyLabel'),
            'RunLabel': first.get('RunLabel'),
            'ArtifactOrigin': first.get('ArtifactOrigin'),
            'DatasetSchemaVersion': first.get('DatasetSchemaVersion'),
            'EvaluationSchemaVersion': first.get('EvaluationSchemaVersion'),
            'IncludeInCausalSummaries': first.get('IncludeInCausalSummaries'),
            'IncludeInFamilySummary': first.get('IncludeInFamilySummary'),
            'SuiteTags': first.get('SuiteTags'),
            'AnalysisBlocks': first.get('AnalysisBlocks'),
            'AnalysisTier': first.get('AnalysisTier'),
            'RunProfileFactors': first.get('RunProfileFactors'),
            'QueryMode': first.get('QueryMode'),
            'FocusMode': first.get('FocusMode'),
            'ChunkTextMode': first.get('ChunkTextMode'),
            'QueryStructure': first.get('QueryStructure'),
            'ChunkTextStyle': first.get('ChunkTextStyle'),
            'WordingConfig': first.get('WordingConfig'),
            'WordingConfigLabel': first.get('WordingConfigLabel'),
            'EmbeddingModel': first.get('EmbeddingModel'),
            'EmbeddingDimension': first.get('EmbeddingDimension'),
            'CandidatePoolMass': first.get('CandidatePoolMass'),
            'k_over_pool': first.get('k_over_pool'),
            'OnlyPassGeometry': first.get('OnlyPassGeometry'),
            'QueryScope': first.get('QueryScope'),
            'k': k,
            'SelectionSource': first.get('SelectionSource'),
            'LambdaPolicy': first.get('LambdaPolicy'),
            'GeometrySourceExperiment': first.get('GeometrySourceExperiment'),
            'GeometryPopulation': first.get('GeometryPopulation'),
            'GeometryPopulationLabel': first.get('GeometryPopulationLabel'),
            'PopulationPassFilterValue': first.get('PopulationPassFilterValue'),
        }
        for strategy in STRATEGIES:
            row = by_strategy.get(strategy)
            label = strategy_label(strategy)
            out[f'{label}_lambda'] = row.get('lam') if row else None
            out[f'{label}_lambda_norm'] = row.get('lambda_norm') if row else None
            out[f'{label}_n_queries'] = row.get('n_queries') if row else None
            for metric, metric_label in REPORT_METRIC_NAME_TO_LABEL.items():
                out[f'{label}_{metric_label}'] = row.get(metric) if row else None

        for metric_label in REPORT_METRIC_LABELS:
            fac_loc = float_or_none(out.get(f'FacLoc_{metric_label}'))
            mmr = float_or_none(out.get(f'MMR_{metric_label}'))
            top_k = float_or_none(out.get(f'TopK_{metric_label}'))
            higher_is_better = REPORT_METRIC_LABEL_TO_SPEC[metric_label].higher_is_better
            out[f'Delta_FacLoc_MMR_{metric_label}'] = _oriented_delta(
                fac_loc, mmr, higher_is_better=higher_is_better
            )
            out[f'Delta_FacLoc_TopK_{metric_label}'] = _oriented_delta(
                fac_loc, top_k, higher_is_better=higher_is_better
            )
            out[f'Delta_MMR_TopK_{metric_label}'] = _oriented_delta(
                mmr, top_k, higher_is_better=higher_is_better
            )

        out['FacLocVsMMR_FCPOutcome'] = delta_outcome(
            float_or_none(out.get('Delta_FacLoc_MMR_FCP')),
            epsilon=practical_effect_threshold('FCP'),
        )
        out['FacLocVsMMR_AllFacetCleanRateOutcome'] = delta_outcome(
            float_or_none(out.get('Delta_FacLoc_MMR_AllFacetCleanRate')),
            epsilon=practical_effect_threshold('AllFacetCleanRate'),
        )
        out['FCPWinner'] = winner_for_metric(out, 'FCP')
        out['AllFacetCleanRateWinner'] = winner_for_metric(out, 'AllFacetCleanRate')
        rows.append(out)
    return rows


def experiment_family_summary_rows(
    comparison_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in comparison_rows:
        family = str(row.get('ExperimentFamilyLabel') or 'Unknown')
        grouped.setdefault(family, []).append(row)

    rows = [
        _experiment_family_summary_row(family=family, group=group)
        for family, group in grouped.items()
    ]

    return sorted_rows(rows, 'Delta_FacLoc_MMR_FCP_mean')  # type: ignore


def experiment_family_budget_summary_rows(
    budget_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, BudgetCategory, str], list[Mapping[str, object]]] = {}
    family_groups: dict[str, list[Mapping[str, object]]] = {}
    for row in budget_rows:
        family = str(row.get('ExperimentFamilyLabel') or 'Unknown')
        category_value = str(row.get('BudgetCategory') or '')
        if category_value not in BUDGET_CATEGORIES:
            continue
        category = cast(BudgetCategory, category_value)
        budget_label = str(row.get('BudgetCategoryLabel') or BUDGET_CATEGORY_LABELS[category])
        grouped.setdefault((family, category, budget_label), []).append(row)
        family_groups.setdefault(family, []).append(row)

    budget_order = {category: index for index, category in enumerate(BUDGET_CATEGORIES)}
    family_mean_delta = {
        family: (
            statistics.fmean(values)
            if (values := numeric_values(group, 'Delta_FacLoc_MMR_FCP'))
            else -math.inf
        )
        for family, group in family_groups.items()
    }

    rows: list[dict[str, object]] = []
    for (family, category, budget_label), group in grouped.items():
        out = _experiment_family_summary_row(
            family=family,
            group=group,
            budget_category=category,
            budget_label=budget_label,
        )
        out['_FamilySort'] = family_mean_delta.get(family, -math.inf)
        out['_BudgetSort'] = budget_order[category]
        rows.append(out)

    rows.sort(
        key=lambda row: (
            -cast(float, row['_FamilySort']),
            cast(int, row['_BudgetSort']),
            str(row.get('ExperimentFamilyLabel') or ''),
        )
    )
    for row in rows:
        row.pop('_FamilySort', None)
        row.pop('_BudgetSort', None)
    return rows


def _experiment_family_summary_row(
    *,
    family: str,
    group: Sequence[Mapping[str, object]],
    budget_category: BudgetCategory | None = None,
    budget_label: str | None = None,
) -> dict[str, object]:
    facloc_better = sum(row.get('FacLocVsMMR_FCPOutcome') == 'facloc_better' for row in group)
    facloc_tied = sum(row.get('FacLocVsMMR_FCPOutcome') == 'tied' for row in group)
    facloc_worse = sum(row.get('FacLocVsMMR_FCPOutcome') == 'facloc_worse' for row in group)
    row_count = len(group)
    out: dict[str, object] = {
        'ExperimentFamilyLabel': family,
        'Rows': row_count,
        'FacLocBetterRows': facloc_better,
        'FacLocTiedRows': facloc_tied,
        'FacLocWorseRows': facloc_worse,
        'FacLocBetterPct': _fraction_or_none(facloc_better, row_count),
        'FacLocTiedPct': _fraction_or_none(facloc_tied, row_count),
        'FacLocWorsePct': _fraction_or_none(facloc_worse, row_count),
    }
    if budget_category is not None and budget_label is not None:
        out['BudgetCategory'] = budget_category
        out['BudgetCategoryLabel'] = budget_label
    out.update(
        numeric_stats(
            numeric_values(group, 'Delta_FacLoc_MMR_FCP'),
            'Delta_FacLoc_MMR_FCP',
        )
    )
    out.update(
        numeric_stats(
            numeric_values(group, 'Delta_FacLoc_TopK_FCP'),
            'Delta_FacLoc_TopK_FCP',
        )
    )
    out.update(
        numeric_stats(
            numeric_values(group, 'Delta_MMR_TopK_FCP'),
            'Delta_MMR_TopK_FCP',
        )
    )
    out.update(
        numeric_stats(
            numeric_values(group, 'Delta_FacLoc_MMR_AllFacetCleanRate'),
            'Delta_FacLoc_MMR_AllFacetCleanRate',
        )
    )
    return out


def metric_aggregate_summary_rows(
    *,
    comparison_rows: Sequence[Mapping[str, object]],
    budget_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    view_rows: tuple[tuple[str, str, Sequence[Mapping[str, object]]], ...] = (
        ('all_k', 'All k', comparison_rows),
        (
            'low_budget',
            'Low Budget',
            [row for row in budget_rows if row.get('BudgetCategory') == 'low_budget'],
        ),
        (
            'medium_budget',
            'Medium budget',
            [row for row in budget_rows if row.get('BudgetCategory') == 'medium_budget'],
        ),
        (
            'high_budget',
            'High budget',
            [row for row in budget_rows if row.get('BudgetCategory') == 'high_budget'],
        ),
    )
    for spec in REPORT_METRIC_SPECS:
        for budget_category, budget_view, source_rows in view_rows:
            rows.append(
                _metric_aggregate_row(
                    metric=spec.metric_label,
                    metric_title=spec.title_label,
                    budget_category=budget_category,
                    budget_view=budget_view,
                    rows=source_rows,
                )
            )
    return rows


def metric_family_summary_rows(
    comparison_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metric_order = {spec.metric_label: index for index, spec in enumerate(REPORT_METRIC_SPECS)}
    for spec in REPORT_METRIC_SPECS:
        grouped: dict[str, list[Mapping[str, object]]] = {}
        for row in comparison_rows:
            family = str(row.get('ExperimentFamilyLabel') or 'Unknown')
            grouped.setdefault(family, []).append(row)
        for family, group in grouped.items():
            out = _metric_group_summary_row(
                metric=spec.metric_label,
                metric_title=spec.title_label,
                rows=group,
                family=family,
            )
            out['_MetricSort'] = metric_order[spec.metric_label]
            rows.append(out)

    rows.sort(
        key=lambda row: (
            cast(int, row['_MetricSort']),
            -(float_or_none(row.get('MeanDeltaFacLocMMR')) or -math.inf),
            str(row.get('ExperimentFamilyLabel') or ''),
        )
    )
    for row in rows:
        row.pop('_MetricSort', None)
    return rows


def metric_family_budget_summary_rows(
    budget_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metric_order = {spec.metric_label: index for index, spec in enumerate(REPORT_METRIC_SPECS)}
    budget_order = {category: index for index, category in enumerate(BUDGET_CATEGORIES)}

    for spec in REPORT_METRIC_SPECS:
        grouped: dict[tuple[str, BudgetCategory, str], list[Mapping[str, object]]] = {}
        family_groups: dict[str, list[Mapping[str, object]]] = {}
        for row in budget_rows:
            family = str(row.get('ExperimentFamilyLabel') or 'Unknown')
            category_value = str(row.get('BudgetCategory') or '')
            if category_value not in BUDGET_CATEGORIES:
                continue
            category = cast(BudgetCategory, category_value)
            budget_label = str(row.get('BudgetCategoryLabel') or BUDGET_CATEGORY_LABELS[category])
            grouped.setdefault((family, category, budget_label), []).append(row)
            family_groups.setdefault(family, []).append(row)

        family_mean_delta = {
            family: (
                statistics.fmean(values)
                if (
                    values := numeric_values(
                        group,
                        f'Delta_FacLoc_MMR_{spec.metric_label}',
                    )
                )
                else -math.inf
            )
            for family, group in family_groups.items()
        }
        for (family, category, budget_label), group in grouped.items():
            out = _metric_group_summary_row(
                metric=spec.metric_label,
                metric_title=spec.title_label,
                rows=group,
                family=family,
                budget_category=category,
                budget_label=budget_label,
            )
            out['_MetricSort'] = metric_order[spec.metric_label]
            out['_FamilySort'] = family_mean_delta.get(family, -math.inf)
            out['_BudgetSort'] = budget_order[category]
            rows.append(out)

    rows.sort(
        key=lambda row: (
            cast(int, row['_MetricSort']),
            -cast(float, row['_FamilySort']),
            cast(int, row['_BudgetSort']),
            str(row.get('ExperimentFamilyLabel') or ''),
        )
    )
    for row in rows:
        row.pop('_MetricSort', None)
        row.pop('_FamilySort', None)
        row.pop('_BudgetSort', None)
    return rows


def _metric_group_summary_row(
    *,
    metric: DeltaMetricLabel,
    metric_title: str,
    rows: Sequence[Mapping[str, object]],
    family: str,
    budget_category: BudgetCategory | None = None,
    budget_label: str | None = None,
) -> dict[str, object]:
    delta_fm_col = f'Delta_FacLoc_MMR_{metric}'
    delta_ft_col = f'Delta_FacLoc_TopK_{metric}'
    delta_mt_col = f'Delta_MMR_TopK_{metric}'
    topk_col = f'TopK_{metric}'
    mmr_col = f'MMR_{metric}'
    facloc_col = f'FacLoc_{metric}'
    complete_rows = [row for row in rows if float_or_none(row.get(delta_fm_col)) is not None]
    deltas_fm = numeric_values(complete_rows, delta_fm_col)
    deltas_ft = numeric_values(complete_rows, delta_ft_col)
    deltas_mt = numeric_values(complete_rows, delta_mt_col)
    out: dict[str, object] = {
        'Metric': metric_title,
        'MetricLabel': metric,
        'ExperimentFamilyLabel': family,
        'Rows': len(complete_rows),
        'MeanDeltaFacLocMMR': statistics.fmean(deltas_fm) if deltas_fm else None,
        'MeanDeltaFacLocTopK': statistics.fmean(deltas_ft) if deltas_ft else None,
        'MeanDeltaMMRTopK': statistics.fmean(deltas_mt) if deltas_mt else None,
        'MeanTopK': _mean_or_none(numeric_values(complete_rows, topk_col)),
        'MeanMMR': _mean_or_none(numeric_values(complete_rows, mmr_col)),
        'MeanFacLoc': _mean_or_none(numeric_values(complete_rows, facloc_col)),
        'TieEpsilon': practical_effect_threshold(metric),
    }
    _add_outcome_percentages(
        out,
        rows=len(complete_rows),
        facloc_better=sum(delta > practical_effect_threshold(metric) for delta in deltas_fm),
        facloc_tied=sum(abs(delta) <= practical_effect_threshold(metric) for delta in deltas_fm),
        facloc_worse=sum(delta < -practical_effect_threshold(metric) for delta in deltas_fm),
        facloc_topk_better=sum(delta > 0.0 for delta in deltas_ft),
        mmr_topk_better=sum(delta > 0.0 for delta in deltas_mt),
    )
    if budget_category is not None and budget_label is not None:
        out['BudgetCategory'] = budget_category
        out['BudgetCategoryLabel'] = budget_label
    return out


def _add_outcome_percentages(
    out: dict[str, object],
    *,
    rows: int,
    facloc_better: int,
    facloc_tied: int,
    facloc_worse: int,
    facloc_topk_better: int,
    mmr_topk_better: int,
) -> None:
    out['FacLocBetterRows'] = facloc_better
    out['FacLocTiedRows'] = facloc_tied
    out['FacLocWorseRows'] = facloc_worse
    out['FacLocTopKBetterRows'] = facloc_topk_better
    out['MMRTopKBetterRows'] = mmr_topk_better
    out['FacLocBetterPct'] = _fraction_or_none(facloc_better, rows)
    out['FacLocTiedPct'] = _fraction_or_none(facloc_tied, rows)
    out['FacLocWorsePct'] = _fraction_or_none(facloc_worse, rows)
    out['FacLocTopKBetterPct'] = _fraction_or_none(facloc_topk_better, rows)
    out['MMRTopKBetterPct'] = _fraction_or_none(mmr_topk_better, rows)


def _fraction_or_none(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _metric_aggregate_row(
    *,
    metric: DeltaMetricLabel,
    metric_title: str,
    budget_category: str,
    budget_view: str,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    delta_fm_col = f'Delta_FacLoc_MMR_{metric}'
    delta_ft_col = f'Delta_FacLoc_TopK_{metric}'
    delta_mt_col = f'Delta_MMR_TopK_{metric}'
    topk_col = f'TopK_{metric}'
    mmr_col = f'MMR_{metric}'
    facloc_col = f'FacLoc_{metric}'
    complete_rows = [row for row in rows if float_or_none(row.get(delta_fm_col)) is not None]
    threshold = practical_effect_threshold(metric)
    deltas_fm = numeric_values(complete_rows, delta_fm_col)
    deltas_ft = numeric_values(complete_rows, delta_ft_col)
    deltas_mt = numeric_values(complete_rows, delta_mt_col)
    facloc_better = sum(delta > threshold for delta in deltas_fm)
    facloc_tied = sum(abs(delta) <= threshold for delta in deltas_fm)
    facloc_worse = sum(delta < -threshold for delta in deltas_fm)
    facloc_topk_better = sum(delta > 0.0 for delta in deltas_ft)
    mmr_topk_better = sum(delta > 0.0 for delta in deltas_mt)
    return {
        'Metric': metric_title,
        'MetricLabel': metric,
        'BudgetCategory': budget_category,
        'BudgetView': budget_view,
        'Rows': len(complete_rows),
        'FacLocBetterRows': facloc_better,
        'FacLocTiedRows': facloc_tied,
        'FacLocWorseRows': facloc_worse,
        'FacLocTopKBetterRows': facloc_topk_better,
        'MMRTopKBetterRows': mmr_topk_better,
        'FacLocBetterPct': family_balanced_rate(
            complete_rows,
            lambda row: (float_or_none(row.get(delta_fm_col)) or 0.0) > threshold,
        ),
        'FacLocTiedPct': family_balanced_rate(
            complete_rows,
            lambda row: abs(float_or_none(row.get(delta_fm_col)) or 0.0) <= threshold,
        ),
        'FacLocWorsePct': family_balanced_rate(
            complete_rows,
            lambda row: (float_or_none(row.get(delta_fm_col)) or 0.0) < -threshold,
        ),
        'FacLocTopKBetterPct': family_balanced_rate(
            complete_rows,
            lambda row: (float_or_none(row.get(delta_ft_col)) or 0.0) > 0.0,
        ),
        'MMRTopKBetterPct': family_balanced_rate(
            complete_rows,
            lambda row: (float_or_none(row.get(delta_mt_col)) or 0.0) > 0.0,
        ),
        'MeanDeltaFacLocMMR': family_balanced_mean(complete_rows, delta_fm_col),
        'MedianDeltaFacLocMMR': statistics.median(deltas_fm) if deltas_fm else None,
        'MeanDeltaFacLocTopK': family_balanced_mean(complete_rows, delta_ft_col),
        'MeanDeltaMMRTopK': family_balanced_mean(complete_rows, delta_mt_col),
        'MeanTopK': family_balanced_mean(complete_rows, topk_col),
        'MeanMMR': family_balanced_mean(complete_rows, mmr_col),
        'MeanFacLoc': family_balanced_mean(complete_rows, facloc_col),
        'TieEpsilon': threshold,
    }


def _mean_or_none(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _oriented_delta(
    left: float | None,
    right: float | None,
    *,
    higher_is_better: bool,
) -> float | None:
    delta = subtract(left, right)
    if delta is None:
        return None
    return delta if higher_is_better else -delta


def low_budget_rows_from_comparisons(
    comparison_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        row
        for row in budget_category_rows_from_comparisons(comparison_rows)
        if row.get('BudgetCategory') == 'low_budget'
    ]


def budget_category_rows_from_comparisons(
    comparison_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_experiment: dict[str, list[Mapping[str, object]]] = {}
    for row in comparison_rows:
        experiment = str(row.get('Experiment') or '')
        if experiment:
            by_experiment.setdefault(experiment, []).append(row)

    rows: list[dict[str, object]] = []
    for _experiment, group in sorted(by_experiment.items()):
        complete_rows = [
            row
            for row in group
            if all(
                row.get(f'{strategy_label(strategy)}_FCP') is not None for strategy in STRATEGIES
            )
        ]
        candidates = sorted(
            complete_rows or group,
            key=lambda row: int(cast(int, row.get('k') or 0)),
        )
        if not candidates:
            continue
        selected_by_category: dict[BudgetCategory, Mapping[str, object]] = {
            'medium_budget': candidates[len(candidates) // 2],
            'high_budget': candidates[-1],
        }
        low_budget_row = next(
            (row for row in candidates if int(cast(int, row.get('k') or 0)) == LOW_BUDGET_K),
            None,
        )
        if low_budget_row is not None:
            selected_by_category['low_budget'] = low_budget_row
        completeness = 'complete strategies' if complete_rows else 'available rows'
        for category in BUDGET_CATEGORIES:
            selected = selected_by_category.get(category)
            if selected is None:
                continue
            out = dict(selected)
            out['BudgetCategory'] = category
            out['BudgetCategoryLabel'] = BUDGET_CATEGORY_LABELS[category]
            out['BudgetCategoryRule'] = _budget_category_rule(
                category, len(candidates), completeness
            )
            if category == 'low_budget':
                out['LowBudgetRule'] = f'fixed global k={LOW_BUDGET_K}'
            rows.append(out)
    return rows


def _budget_category_rule(
    category: BudgetCategory,
    k_count: int,
    completeness: str,
) -> str:
    if category == 'low_budget':
        return f'fixed global k={LOW_BUDGET_K}'
    if category == 'medium_budget':
        return f'k at index floor({k_count}/2) among {completeness}'
    return f'highest k among {completeness}'


def lambda_stability_rows(
    strategy_rows: Sequence[Mapping[str, object]],
    near_optimal_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for strategy in DIVERSIFYING_STRATEGIES:
        selected = [row for row in strategy_rows if row.get('strategy') == strategy]
        lambdas = [float_or_none(row.get('lam')) for row in selected]
        lambda_norms = [float_or_none(row.get('lambda_norm')) for row in selected]
        lambdas = [value for value in lambdas if value is not None]
        lambda_norms = [value for value in lambda_norms if value is not None]
        near_rows = [row for row in near_optimal_rows if row.get('strategy') == strategy]
        near_fractions = [
            value
            for value in (float_or_none(row.get('NearOptimalLambdaFraction')) for row in near_rows)
            if value is not None
        ]
        near_spans = [
            value
            for value in (float_or_none(row.get('NearOptimalLambdaSpanNorm')) for row in near_rows)
            if value is not None
        ]
        row: dict[str, object] = {
            'strategy': strategy,
            'n_selected': len(lambdas),
            'distinct_lambda_count': len(set(round(value, 6) for value in lambdas)),
            'boundary_selection_rate': boundary_rate(lambda_norms),
            'near_optimal_rows': len(near_rows),
        }
        row.update(numeric_stats(lambdas, prefix='selected_lambda'))
        row.update(numeric_stats(lambda_norms, prefix='selected_lambda_norm'))
        row.update(numeric_stats(near_fractions, prefix='near_optimal_fraction'))
        row.update(numeric_stats(near_spans, prefix='near_optimal_span_norm'))
        rows.append(row)
    return rows


def lambda_curve_summary_rows(
    lambda_grid_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Aggregate validation-grid deltas at each normalized lambda position.

    The mean gives every observed model/family stratum equal weight, while the
    quantiles and cell-level safe fraction retain the unweighted experiment-
    budget distribution.  Keeping both views prevents a large family from
    hiding reversals in smaller but methodologically important cells.
    """
    grouped: dict[tuple[str, float], list[Mapping[str, object]]] = {}
    for row in lambda_grid_rows:
        strategy = row.get('strategy')
        lambda_norm = float_or_none(row.get('lambda_norm'))
        if strategy not in DIVERSIFYING_STRATEGIES or lambda_norm is None:
            continue
        grouped.setdefault((str(strategy), round(lambda_norm, 6)), []).append(row)

    rows: list[dict[str, object]] = []
    for (strategy, lambda_norm), group in sorted(grouped.items()):
        fcp_deltas = numeric_values(group, 'DeltaStrategyTopK_FCP')
        if not fcp_deltas:
            continue
        fcp_group = [
            row for row in group if float_or_none(row.get('DeltaStrategyTopK_FCP')) is not None
        ]
        distractor_deltas = numeric_values(
            group,
            'DeltaStrategyTopK_DistractorRate',
        )
        balanced_fcp = _lambda_balanced_mean(
            group,
            'DeltaStrategyTopK_FCP',
            dimensions=('EmbeddingModel', 'ExperimentFamily'),
        )
        balanced_safe = _lambda_balanced_fraction(
            fcp_group,
            predicate=lambda row: (
                (value := float_or_none(row.get('DeltaStrategyTopK_FCP'))) is not None
                and value >= 0.0
            ),
            dimensions=('EmbeddingModel', 'ExperimentFamily'),
        )
        balanced_distractor = _lambda_balanced_mean(
            group,
            'DeltaStrategyTopK_DistractorRate',
            dimensions=('EmbeddingModel', 'ExperimentFamily'),
        )
        out: dict[str, object] = {
            'strategy': strategy,
            'lam': _mean_or_none(
                [
                    value
                    for value in (float_or_none(row.get('lam')) for row in group)
                    if value is not None
                ]
            ),
            'lambda_norm': lambda_norm,
            'CellRows': len(fcp_deltas),
            'ModelCount': len({str(row.get('EmbeddingModel') or '') for row in group}),
            'FamilyCount': len({str(row.get('ExperimentFamily') or '') for row in group}),
            'MeanDeltaStrategyTopK_FCP': balanced_fcp,
            'CellMeanDeltaStrategyTopK_FCP': statistics.fmean(fcp_deltas),
            'CellMedianDeltaStrategyTopK_FCP': statistics.median(fcp_deltas),
            'CellQ25DeltaStrategyTopK_FCP': quantile(sorted(fcp_deltas), 0.25),
            'CellQ75DeltaStrategyTopK_FCP': quantile(sorted(fcp_deltas), 0.75),
            'CellSafeLambdaFraction': sum(value >= 0.0 for value in fcp_deltas)
            / len(fcp_deltas),
            'BalancedSafeLambdaFraction': balanced_safe,
            'MeanDeltaStrategyTopK_DistractorRate': balanced_distractor,
            'CellMeanDeltaStrategyTopK_DistractorRate': (
                statistics.fmean(distractor_deltas) if distractor_deltas else None
            ),
            'CellQ25DeltaStrategyTopK_DistractorRate': (
                quantile(sorted(distractor_deltas), 0.25) if distractor_deltas else None
            ),
            'CellQ75DeltaStrategyTopK_DistractorRate': (
                quantile(sorted(distractor_deltas), 0.75) if distractor_deltas else None
            ),
        }
        rows.append(out)
    return rows


def lambda_robustness_summary_rows(
    lambda_safety_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Summarize full-grid safety overall and by model, family, and budget."""
    rows: list[dict[str, object]] = []
    for strategy in DIVERSIFYING_STRATEGIES:
        strategy_rows = [row for row in lambda_safety_rows if row.get('strategy') == strategy]
        if not strategy_rows:
            continue
        scopes: list[tuple[str, str, list[Mapping[str, object]], tuple[str, ...], str]] = [
            (
                'overall',
                'all',
                strategy_rows,
                ('EmbeddingModel', 'ExperimentFamily'),
                'model-and-family-balanced',
            )
        ]
        for model in sorted({str(row.get('EmbeddingModel') or 'unknown') for row in strategy_rows}):
            scopes.append(
                (
                    'embedding_model',
                    model,
                    [row for row in strategy_rows if str(row.get('EmbeddingModel') or 'unknown') == model],
                    ('ExperimentFamily',),
                    'family-balanced',
                )
            )
        for family in sorted({str(row.get('ExperimentFamily') or 'unknown') for row in strategy_rows}):
            scopes.append(
                (
                    'experiment_family',
                    family,
                    [row for row in strategy_rows if str(row.get('ExperimentFamily') or 'unknown') == family],
                    ('EmbeddingModel',),
                    'model-balanced',
                )
            )
        for k in sorted(
            {
                k_value
                for k_value in (int_or_none(row.get('k')) for row in strategy_rows)
                if k_value is not None
            }
        ):
            scopes.append(
                (
                    'budget',
                    str(k),
                    [row for row in strategy_rows if int_or_none(row.get('k')) == k],
                    ('EmbeddingModel', 'ExperimentFamily'),
                    'model-and-family-balanced',
                )
            )

        for scope, scope_value, scoped_rows, dimensions, weighting in scopes:
            safe_values = numeric_values(scoped_rows, 'SafeLambdaFraction')
            worst_values = numeric_values(scoped_rows, 'WorstDeltaStrategyTopK_FCP')
            range_values = numeric_values(scoped_rows, 'DeltaStrategyTopK_FCPRange')
            if not safe_values or not worst_values:
                continue
            rows.append(
                {
                    'Scope': scope,
                    'ScopeValue': scope_value,
                    'strategy': strategy,
                    'Rows': len(scoped_rows),
                    'ModelCount': len(
                        {str(row.get('EmbeddingModel') or 'unknown') for row in scoped_rows}
                    ),
                    'FamilyCount': len(
                        {str(row.get('ExperimentFamily') or 'unknown') for row in scoped_rows}
                    ),
                    'BudgetCount': len(
                        {int_or_none(row.get('k')) for row in scoped_rows if int_or_none(row.get('k')) is not None}
                    ),
                    'LambdaGridPoints': max(
                        int_or_none(row.get('LambdaCount')) or 0 for row in scoped_rows
                    ),
                    'MeanWeighting': weighting,
                    'MeanSafeLambdaFraction': _lambda_balanced_mean(
                        scoped_rows,
                        'SafeLambdaFraction',
                        dimensions=dimensions,
                    ),
                    'MedianSafeLambdaFraction': statistics.median(safe_values),
                    'AllGridSafeRows': sum(value >= 1.0 - 1e-12 for value in safe_values),
                    'AllGridSafeRate': sum(value >= 1.0 - 1e-12 for value in safe_values)
                    / len(safe_values),
                    'MeanWorstDeltaStrategyTopK_FCP': _lambda_balanced_mean(
                        scoped_rows,
                        'WorstDeltaStrategyTopK_FCP',
                        dimensions=dimensions,
                    ),
                    'MedianWorstDeltaStrategyTopK_FCP': statistics.median(worst_values),
                    'MinWorstDeltaStrategyTopK_FCP': min(worst_values),
                    'MeanFCPRange': _lambda_balanced_mean(
                        scoped_rows,
                        'DeltaStrategyTopK_FCPRange',
                        dimensions=dimensions,
                    )
                    if range_values
                    else None,
                }
            )
    return rows


def _lambda_balanced_mean(
    rows: Sequence[Mapping[str, object]],
    column: str,
    *,
    dimensions: tuple[str, ...],
) -> float | None:
    grouped: dict[tuple[str, ...], list[float]] = {}
    for row in rows:
        value = float_or_none(row.get(column))
        if value is None:
            continue
        key = tuple(str(row.get(dimension) or 'unknown') for dimension in dimensions)
        grouped.setdefault(key, []).append(value)
    means = [statistics.fmean(values) for values in grouped.values() if values]
    return statistics.fmean(means) if means else None


def _lambda_balanced_fraction(
    rows: Sequence[Mapping[str, object]],
    *,
    predicate: Callable[[Mapping[str, object]], bool],
    dimensions: tuple[str, ...],
) -> float | None:
    grouped: dict[tuple[str, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(str(row.get(dimension) or 'unknown') for dimension in dimensions)
        grouped.setdefault(key, []).append(row)
    fractions: list[float] = []
    for group in grouped.values():
        fractions.append(sum(bool(predicate(row)) for row in group) / len(group))
    return statistics.fmean(fractions) if fractions else None


def embedding_model_summary_rows(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    geometry_rows: Sequence[Mapping[str, object]],
    low_budget_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    manifest_by_exp = {str(row.get('Experiment')): row for row in manifest_rows}
    geometry_by_exp = {str(row.get('Experiment')): row for row in geometry_rows}
    low_budget_by_exp = {str(row.get('Experiment')): row for row in low_budget_rows}
    run_rows: list[Mapping[str, object]] = []
    for experiment, manifest in manifest_by_exp.items():
        geometry = geometry_by_exp.get(experiment, {})
        low_budget = low_budget_by_exp.get(experiment, {})
        run_rows.append(
            {
                'EmbeddingModel': manifest.get('EmbeddingModel'),
                'ExperimentFamily': manifest.get('ExperimentFamily'),
                'ExperimentFamilyLabel': manifest.get('ExperimentFamilyLabel'),
                'EmbeddingDimension': manifest.get('EmbeddingDimension'),
                'GeometryPassRate': geometry.get('GeometryPassRate'),
                'GeometryQueries': geometry.get('GeometryQueries'),
                'GeometryPassQueries': geometry.get('GeometryPassQueries'),
                'TopK_FCP': low_budget.get('TopK_FCP'),
                'MMR_FCP': low_budget.get('MMR_FCP'),
                'FacLoc_FCP': low_budget.get('FacLoc_FCP'),
                'Delta_FacLoc_MMR_FCP': low_budget.get('Delta_FacLoc_MMR_FCP'),
                'Delta_FacLoc_TopK_FCP': low_budget.get('Delta_FacLoc_TopK_FCP'),
                'OnlyPassGeometry': manifest.get('OnlyPassGeometry'),
            }
        )

    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in run_rows:
        model = str(row.get('EmbeddingModel') or 'unknown')
        grouped.setdefault(model, []).append(row)

    rows: list[dict[str, object]] = []
    for model, group in sorted(grouped.items()):
        out: dict[str, object] = {'EmbeddingModel': model, 'Runs': len(group)}
        out.update(numeric_stats(numeric_values(group, 'EmbeddingDimension'), 'EmbeddingDimension'))
        out.update(numeric_stats(numeric_values(group, 'GeometryPassRate'), 'GeometryPassRate'))
        out.update(numeric_stats(numeric_values(group, 'GeometryQueries'), 'GeometryQueries'))
        out.update(
            numeric_stats(numeric_values(group, 'GeometryPassQueries'), 'GeometryPassQueries')
        )
        out.update(numeric_stats(numeric_values(group, 'TopK_FCP'), 'TopK_FCP'))
        out.update(numeric_stats(numeric_values(group, 'MMR_FCP'), 'MMR_FCP'))
        out.update(numeric_stats(numeric_values(group, 'FacLoc_FCP'), 'FacLoc_FCP'))
        out.update(
            numeric_stats(numeric_values(group, 'Delta_FacLoc_MMR_FCP'), 'Delta_FacLoc_MMR_FCP')
        )
        out.update(
            numeric_stats(
                numeric_values(group, 'Delta_FacLoc_TopK_FCP'),
                'Delta_FacLoc_TopK_FCP',
            )
        )
        for column in (
            'GeometryPassRate',
            'TopK_FCP',
            'MMR_FCP',
            'FacLoc_FCP',
            'Delta_FacLoc_MMR_FCP',
            'Delta_FacLoc_TopK_FCP',
        ):
            out[f'{column}_mean'] = family_balanced_mean(group, column)
        out['PassFilterRuns'] = sum(row.get('OnlyPassGeometry') is True for row in group)
        rows.append(out)
    return rows
