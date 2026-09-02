"""Model-first summaries for the fixed embedding-model crossing."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence

from experiments.medical_dataset_gen.reports.helpers import float_or_none
from experiments.medical_dataset_gen.reports.report_config import (
    BUDGET_CATEGORIES,
    REPORT_METRIC_SPECS,
    embedding_model_display_label,
)

type ReportRow = Mapping[str, object]


def model_grid_coverage_rows(
    manifest_rows: Sequence[ReportRow],
    *,
    embedding_models: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Audit that every model materializes the same distribution/wording cells."""
    keys_by_model: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in manifest_rows:
        model = str(row.get('EmbeddingModel') or '')
        distribution = str(row.get('Distribution') or '')
        wording = str(row.get('WordingConfig') or '')
        if model and distribution and wording:
            keys_by_model[model].add((distribution, wording))
    expected = set().union(*(keys_by_model.get(model, set()) for model in embedding_models))
    coverage: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    for model in embedding_models:
        observed = keys_by_model.get(model, set())
        missing = sorted(expected.difference(observed))
        coverage.append(
            {
                'EmbeddingModel': model,
                'EmbeddingModelLabel': embedding_model_display_label(model),
                'ExpectedCells': len(expected),
                'ObservedCells': len(observed),
                'MissingCells': len(missing),
                'Complete': not missing and bool(expected),
            }
        )
        missing_rows.extend(
            {
                'EmbeddingModel': model,
                'Distribution': distribution,
                'WordingConfig': wording,
            }
            for distribution, wording in missing
        )
    return coverage, missing_rows


def embedding_metric_summary_rows(
    comparison_rows: Sequence[ReportRow],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    models = sorted(
        {
            str(row.get('EmbeddingModel') or '')
            for row in comparison_rows
            if row.get('EmbeddingModel')
        }
    )
    families = sorted(
        {
            str(row.get('ExperimentFamilyLabel') or '')
            for row in comparison_rows
            if row.get('ExperimentFamilyLabel')
        }
    )
    for model in models:
        model_rows = [row for row in comparison_rows if row.get('EmbeddingModel') == model]
        scopes: list[tuple[str, str, list[ReportRow], tuple[str, ...]]] = [
            ('overall', 'all', model_rows, ('ExperimentFamilyLabel', 'BudgetCategory'))
        ]
        scopes.extend(
            (
                'experiment_family',
                family,
                [row for row in model_rows if row.get('ExperimentFamilyLabel') == family],
                ('BudgetCategory',),
            )
            for family in families
        )
        scopes.extend(
            (
                'budget',
                budget,
                [row for row in model_rows if row.get('BudgetCategory') == budget],
                ('ExperimentFamilyLabel',),
            )
            for budget in BUDGET_CATEGORIES
        )
        scopes.extend(
            (
                'family_budget',
                f'{family}|{budget}',
                [
                    row
                    for row in model_rows
                    if row.get('ExperimentFamilyLabel') == family
                    and row.get('BudgetCategory') == budget
                ],
                (),
            )
            for family in families
            for budget in BUDGET_CATEGORIES
        )
        for scope, scope_value, scoped_rows, balance_dimensions in scopes:
            if not scoped_rows:
                continue
            for spec in REPORT_METRIC_SPECS:
                metric = spec.metric_label
                out: dict[str, object] = {
                    'EmbeddingModel': model,
                    'EmbeddingModelLabel': embedding_model_display_label(model),
                    'MetricLabel': metric,
                    'Scope': scope,
                    'ScopeValue': scope_value,
                    'Rows': len(scoped_rows),
                }
                if scope in {'experiment_family', 'family_budget'}:
                    out['ExperimentFamilyLabel'] = scope_value.split('|', 1)[0]
                if scope in {'budget', 'family_budget'}:
                    out['BudgetCategory'] = scope_value.rsplit('|', 1)[-1]
                for column, output in (
                    (f'TopK_{metric}', 'MeanTopK'),
                    (f'MMR_{metric}', 'MeanMMR'),
                    (f'FacLoc_{metric}', 'MeanFacLoc'),
                    (f'Delta_FacLoc_MMR_{metric}', 'MeanDeltaFacLocMMR'),
                    (f'Delta_FacLoc_TopK_{metric}', 'MeanDeltaFacLocTopK'),
                    (f'Delta_MMR_TopK_{metric}', 'MeanDeltaMMRTopK'),
                ):
                    out[output] = _balanced_mean(
                        scoped_rows,
                        column,
                        dimensions=balance_dimensions,
                    )
                rows.append(out)
    return rows


def embedding_metric_range_rows(
    summary_rows: Sequence[ReportRow],
    *,
    complete_crossing: bool,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[ReportRow]] = defaultdict(list)
    for row in summary_rows:
        grouped[
            (
                str(row.get('MetricLabel') or ''),
                str(row.get('Scope') or ''),
                str(row.get('ScopeValue') or ''),
            )
        ].append(row)
    output: list[dict[str, object]] = []
    for (metric, scope, scope_value), rows in sorted(grouped.items()):
        out: dict[str, object] = {
            'MetricLabel': metric,
            'Scope': scope,
            'ScopeValue': scope_value,
            'ModelCount': len(rows),
            'CompleteCrossing': complete_crossing,
        }
        if rows:
            out['ExperimentFamilyLabel'] = rows[0].get('ExperimentFamilyLabel')
            out['BudgetCategory'] = rows[0].get('BudgetCategory')
        for column in (
            'MeanTopK',
            'MeanMMR',
            'MeanFacLoc',
            'MeanDeltaFacLocMMR',
            'MeanDeltaFacLocTopK',
            'MeanDeltaMMRTopK',
        ):
            values = [
                value for row in rows if (value := float_or_none(row.get(column))) is not None
            ]
            out[column] = statistics.fmean(values) if values else None
            out[f'{column}MinModel'] = min(values) if values and complete_crossing else None
            out[f'{column}MaxModel'] = max(values) if values and complete_crossing else None
        output.append(out)
    return output


def embedding_geometry_summary_rows(
    geometry_rows: Sequence[ReportRow],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    primary = [row for row in geometry_rows if row.get('AnalysisTier') != 'interaction']
    models = sorted(
        {str(row.get('EmbeddingModel') or '') for row in primary if row.get('EmbeddingModel')}
    )
    overall_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []
    for model in models:
        model_rows = [row for row in primary if row.get('EmbeddingModel') == model]
        overall_rows.append(_geometry_summary(model=model, rows=model_rows, family=None))
        families = sorted(
            {
                str(row.get('ExperimentFamilyLabel') or '')
                for row in model_rows
                if row.get('ExperimentFamilyLabel')
            }
        )
        family_rows.extend(
            _geometry_summary(
                model=model,
                rows=[row for row in model_rows if row.get('ExperimentFamilyLabel') == family],
                family=family,
            )
            for family in families
        )
    return overall_rows, family_rows


def lambda_curve_by_embedding_model_rows(
    lambda_grid_rows: Sequence[ReportRow],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, float], list[ReportRow]] = defaultdict(list)
    for row in lambda_grid_rows:
        model = str(row.get('EmbeddingModel') or '')
        strategy = str(row.get('strategy') or '')
        lambda_norm = float_or_none(row.get('lambda_norm'))
        if model and strategy in {'mmr', 'fac_loc'} and lambda_norm is not None:
            grouped[(model, strategy, round(lambda_norm, 6))].append(row)
    output: list[dict[str, object]] = []
    for (model, strategy, lambda_norm), rows in sorted(grouped.items()):
        output.append(
            {
                'EmbeddingModel': model,
                'EmbeddingModelLabel': embedding_model_display_label(model),
                'strategy': strategy,
                'lambda_norm': lambda_norm,
                'MeanDeltaStrategyTopK_FCP': _balanced_mean(
                    rows,
                    'DeltaStrategyTopK_FCP',
                    dimensions=('ExperimentFamily',),
                ),
                'SafeFraction': _balanced_fraction(
                    rows,
                    column='DeltaStrategyTopK_FCP',
                    dimensions=('ExperimentFamily',),
                ),
                'MeanDeltaStrategyTopK_DistractorRate': _balanced_mean(
                    rows,
                    'DeltaStrategyTopK_DistractorRate',
                    dimensions=('ExperimentFamily',),
                ),
                'CellRows': len(rows),
            }
        )
    return output


def _geometry_summary(
    *,
    model: str,
    rows: Sequence[ReportRow],
    family: str | None,
) -> dict[str, object]:
    out: dict[str, object] = {
        'EmbeddingModel': model,
        'EmbeddingModelLabel': embedding_model_display_label(model),
        'Rows': len(rows),
        'CoverageStressRate': _balanced_mean(
            rows,
            'CoverageStressRate',
            dimensions=() if family is not None else ('ExperimentFamilyLabel',),
        ),
        'GoldNearMissMargin': _balanced_mean(
            rows,
            'GoldNearMissMargin',
            dimensions=() if family is not None else ('ExperimentFamilyLabel',),
        ),
        'GoldBackgroundMargin': _balanced_mean(
            rows,
            'GoldBackgroundMargin',
            dimensions=() if family is not None else ('ExperimentFamilyLabel',),
        ),
    }
    if family is not None:
        out['ExperimentFamilyLabel'] = family
    return out


def _balanced_mean(
    rows: Sequence[ReportRow],
    column: str,
    *,
    dimensions: tuple[str, ...],
) -> float | None:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = float_or_none(row.get(column))
        if value is None:
            continue
        key = tuple(str(row.get(dimension) or 'unknown') for dimension in dimensions)
        grouped[key].append(value)
    means = [statistics.fmean(values) for values in grouped.values() if values]
    return statistics.fmean(means) if means else None


def _balanced_fraction(
    rows: Sequence[ReportRow],
    *,
    column: str,
    dimensions: tuple[str, ...],
) -> float | None:
    grouped: dict[tuple[str, ...], list[bool]] = defaultdict(list)
    for row in rows:
        value = float_or_none(row.get(column))
        if value is None:
            continue
        key = tuple(str(row.get(dimension) or 'unknown') for dimension in dimensions)
        grouped[key].append(value >= 0.0)
    fractions = [sum(values) / len(values) for values in grouped.values() if values]
    return statistics.fmean(fractions) if fractions else None
