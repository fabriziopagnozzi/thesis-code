from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import cast

from experiments.medical_dataset_gen.analysis.analysis_constants import (
    DELTA_METRIC_LABELS,
    DIVERSIFYING_STRATEGIES,
    FCP_TIE_EPSILON,
    METRIC_LABELS,
    STRATEGIES,
    DeltaMetricLabel,
)
from experiments.medical_dataset_gen.analysis.helpers import (
    boundary_rate,
    delta_outcome,
    float_or_none,
    int_or_none,
    numeric_stats,
    numeric_values,
    sorted_rows,
    strategy_label,
    subtract,
    winner_for_metric,
)
from experiments.medical_dataset_gen.analysis.models import BudgetCategory
from experiments.medical_dataset_gen.analysis.report_config import (
    BUDGET_CATEGORIES,
    BUDGET_CATEGORY_LABELS,
    DELTA_METRIC_PLOT_SPECS,
)


def comparison_by_k_rows(strategy_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], dict[str, Mapping[str, object]]] = {}
    for row in strategy_rows:
        experiment = str(row.get('Experiment') or '')
        k = int_or_none(row.get('k'))
        strategy = row.get('strategy')
        if not experiment or k is None or strategy not in STRATEGIES:
            continue
        grouped.setdefault((experiment, k), {})[cast(str, strategy)] = row

    rows: list[dict[str, object]] = []
    for (experiment, k), by_strategy in sorted(grouped.items()):
        first = next(iter(by_strategy.values()))
        out = {
            'Experiment': experiment,
            'ShortExperiment': first.get('ShortExperiment'),
            'Distribution': first.get('Distribution'),
            'ShortDistribution': first.get('ShortDistribution'),
            'ExperimentFamily': first.get('ExperimentFamily'),
            'ExperimentFamilyLabel': first.get('ExperimentFamilyLabel'),
            'RunLabel': first.get('RunLabel'),
            'EmbeddingModel': first.get('EmbeddingModel'),
            'EmbeddingDimension': first.get('EmbeddingDimension'),
            'OnlyPassGeometry': first.get('OnlyPassGeometry'),
            'QueryScope': first.get('QueryScope'),
            'k': k,
            'SelectionSource': first.get('SelectionSource'),
        }
        for strategy in STRATEGIES:
            row = by_strategy.get(strategy)
            label = strategy_label(strategy)
            out[f'{label}_lambda'] = row.get('lam') if row else None
            out[f'{label}_lambda_norm'] = row.get('lambda_norm') if row else None
            out[f'{label}_n_queries'] = row.get('n_queries') if row else None
            for metric, metric_label in METRIC_LABELS.items():
                out[f'{label}_{metric_label}'] = row.get(metric) if row else None

        for metric_label in DELTA_METRIC_LABELS:
            fac_loc = float_or_none(out.get(f'FacLoc_{metric_label}'))
            mmr = float_or_none(out.get(f'MMR_{metric_label}'))
            top_k = float_or_none(out.get(f'TopK_{metric_label}'))
            out[f'Delta_FacLoc_MMR_{metric_label}'] = subtract(fac_loc, mmr)
            out[f'Delta_FacLoc_TopK_{metric_label}'] = subtract(fac_loc, top_k)
            out[f'Delta_MMR_TopK_{metric_label}'] = subtract(mmr, top_k)

        out['FacLocVsMMR_FCPOutcome'] = delta_outcome(
            float_or_none(out.get('Delta_FacLoc_MMR_FCP')),
            epsilon=FCP_TIE_EPSILON,
        )
        out['FacLocVsMMR_AllFacetCleanRateOutcome'] = delta_outcome(
            float_or_none(out.get('Delta_FacLoc_MMR_AllFacetCleanRate')),
            epsilon=FCP_TIE_EPSILON,
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
    for spec in DELTA_METRIC_PLOT_SPECS:
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
    metric_order = {spec.metric_label: index for index, spec in enumerate(DELTA_METRIC_PLOT_SPECS)}
    for spec in DELTA_METRIC_PLOT_SPECS:
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
    metric_order = {spec.metric_label: index for index, spec in enumerate(DELTA_METRIC_PLOT_SPECS)}
    budget_order = {category: index for index, category in enumerate(BUDGET_CATEGORIES)}

    for spec in DELTA_METRIC_PLOT_SPECS:
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
        'TieEpsilon': FCP_TIE_EPSILON,
    }
    _add_outcome_percentages(
        out,
        rows=len(complete_rows),
        facloc_better=sum(delta > FCP_TIE_EPSILON for delta in deltas_fm),
        facloc_tied=sum(abs(delta) <= FCP_TIE_EPSILON for delta in deltas_fm),
        facloc_worse=sum(delta < -FCP_TIE_EPSILON for delta in deltas_fm),
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
    complete_rows = [row for row in rows if float_or_none(row.get(delta_fm_col)) is not None]
    deltas_fm = numeric_values(complete_rows, delta_fm_col)
    deltas_ft = numeric_values(complete_rows, delta_ft_col)
    deltas_mt = numeric_values(complete_rows, delta_mt_col)
    facloc_better = sum(delta > FCP_TIE_EPSILON for delta in deltas_fm)
    facloc_tied = sum(abs(delta) <= FCP_TIE_EPSILON for delta in deltas_fm)
    facloc_worse = sum(delta < -FCP_TIE_EPSILON for delta in deltas_fm)
    return {
        'Metric': metric_title,
        'MetricLabel': metric,
        'BudgetCategory': budget_category,
        'BudgetView': budget_view,
        'Rows': len(complete_rows),
        'FacLocBetterRows': facloc_better,
        'FacLocTiedRows': facloc_tied,
        'FacLocWorseRows': facloc_worse,
        'FacLocTopKBetterRows': sum(delta > 0.0 for delta in deltas_ft),
        'MMRTopKBetterRows': sum(delta > 0.0 for delta in deltas_mt),
        'FacLocBetterPct': _fraction_or_none(facloc_better, len(complete_rows)),
        'FacLocTiedPct': _fraction_or_none(facloc_tied, len(complete_rows)),
        'FacLocWorsePct': _fraction_or_none(facloc_worse, len(complete_rows)),
        'FacLocTopKBetterPct': _fraction_or_none(
            sum(delta > 0.0 for delta in deltas_ft),
            len(complete_rows),
        ),
        'MMRTopKBetterPct': _fraction_or_none(
            sum(delta > 0.0 for delta in deltas_mt),
            len(complete_rows),
        ),
        'MeanDeltaFacLocMMR': statistics.fmean(deltas_fm) if deltas_fm else None,
        'MedianDeltaFacLocMMR': statistics.median(deltas_fm) if deltas_fm else None,
        'MeanDeltaFacLocTopK': statistics.fmean(deltas_ft) if deltas_ft else None,
        'MeanDeltaMMRTopK': statistics.fmean(deltas_mt) if deltas_mt else None,
        'TieEpsilon': FCP_TIE_EPSILON,
    }


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
            'low_budget': candidates[0],
            'medium_budget': candidates[len(candidates) // 2],
            'high_budget': candidates[-1],
        }
        completeness = 'complete strategies' if complete_rows else 'available rows'
        for category in BUDGET_CATEGORIES:
            selected = selected_by_category[category]
            out = dict(selected)
            out['BudgetCategory'] = category
            out['BudgetCategoryLabel'] = BUDGET_CATEGORY_LABELS[category]
            out['BudgetCategoryRule'] = _budget_category_rule(
                category, len(candidates), completeness
            )
            if category == 'low_budget':
                out['LowBudgetRule'] = (
                    'smallest k with all strategies' if complete_rows else 'smallest k'
                )
            rows.append(out)
    return rows


def _budget_category_rule(
    category: BudgetCategory,
    k_count: int,
    completeness: str,
) -> str:
    if category == 'low_budget':
        return f'lowest k among {completeness}'
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
        run_rows.append({
            'EmbeddingModel': manifest.get('EmbeddingModel'),
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
        })

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
        out['PassFilterRuns'] = sum(row.get('OnlyPassGeometry') is True for row in group)
        rows.append(out)
    return rows
