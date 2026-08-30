"""Embedding-model completeness and model-first report summaries."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import cast

from experiments.medical_dataset_gen.reports.analysis_constants import (
    DIVERSIFYING_STRATEGIES,
    DeltaMetricLabel,
    practical_effect_threshold,
)
from experiments.medical_dataset_gen.reports.helpers import (
    float_or_none,
    ordered_embedding_models,
    quantile,
)
from experiments.medical_dataset_gen.reports.models import BudgetCategory
from experiments.medical_dataset_gen.reports.report_config import (
    BUDGET_CATEGORIES,
    BUDGET_CATEGORY_LABELS,
    REPORT_METRIC_SPECS,
    embedding_model_display_label,
)
from experiments.medical_dataset_gen.suites.core import SuiteManifest, SuiteManifestCell

type ReportRow = Mapping[str, object]
type MutableReportRow = dict[str, object]


def model_grid_coverage_rows(
    *,
    manifest: SuiteManifest,
    completed_experiments: set[str],
    embedding_models: Sequence[str],
    scope_cell_ids: set[str] | None = None,
) -> tuple[list[MutableReportRow], list[MutableReportRow], bool]:
    """Audit whether every selected model has its complete declared cell grid."""
    cells_by_model: dict[str, list[SuiteManifestCell]] = defaultdict(list)
    for cell in manifest.cells:
        if scope_cell_ids is not None and cell.cell_id not in scope_cell_ids:
            continue
        model = _resolve_manifest_model(cell, embedding_models)
        if model is not None:
            cells_by_model[model].append(cell)

    summary_rows: list[MutableReportRow] = []
    missing_rows: list[MutableReportRow] = []
    for model in ordered_embedding_models(embedding_models):
        cells = cells_by_model.get(model, [])
        completed = [cell for cell in cells if cell.name in completed_experiments]
        missing = [cell for cell in cells if cell.name not in completed_experiments]
        summary_rows.append(
            {
                'EmbeddingModel': model,
                'EmbeddingModelLabel': embedding_model_display_label(model),
                'ExpectedCells': len(cells),
                'CompletedCells': len(completed),
                'MissingCells': len(missing),
                'CoverageRate': len(completed) / len(cells) if cells else None,
                'Complete': bool(cells) and not missing,
            }
        )
        missing_rows.extend(
            {
                'EmbeddingModel': model,
                'EmbeddingModelLabel': embedding_model_display_label(model),
                'CellId': cell.cell_id,
                'Experiment': cell.name,
                'Distribution': cell.distribution_id,
                'RunProfile': cell.run_profile_id,
            }
            for cell in missing
        )
    complete = bool(summary_rows) and all(row['Complete'] is True for row in summary_rows)
    return summary_rows, missing_rows, complete


def embedding_metric_summary_rows(
    budget_rows: Sequence[ReportRow],
) -> list[MutableReportRow]:
    """Summarize held-out metrics inside each model before any model pooling."""
    rows: list[MutableReportRow] = []
    models = ordered_embedding_models(
        str(row.get('EmbeddingModel') or '') for row in budget_rows if row.get('EmbeddingModel')
    )
    for spec in REPORT_METRIC_SPECS:
        for model in models:
            model_rows = [row for row in budget_rows if row.get('EmbeddingModel') == model]
            rows.append(
                _embedding_metric_row(
                    metric=spec.metric_label,
                    metric_title=spec.title_label,
                    model=model,
                    scope='overall',
                    rows=model_rows,
                    strata=('ExperimentFamilyLabel', 'BudgetCategory'),
                )
            )
            for budget in BUDGET_CATEGORIES:
                scoped = [row for row in model_rows if row.get('BudgetCategory') == budget]
                rows.append(
                    _embedding_metric_row(
                        metric=spec.metric_label,
                        metric_title=spec.title_label,
                        model=model,
                        scope='budget',
                        rows=scoped,
                        strata=('ExperimentFamilyLabel',),
                        budget=budget,
                    )
                )
            families = sorted(
                {str(row.get('ExperimentFamilyLabel') or 'Unknown') for row in model_rows}
            )
            for family in families:
                family_rows = [
                    row
                    for row in model_rows
                    if str(row.get('ExperimentFamilyLabel') or 'Unknown') == family
                ]
                rows.append(
                    _embedding_metric_row(
                        metric=spec.metric_label,
                        metric_title=spec.title_label,
                        model=model,
                        scope='family',
                        rows=family_rows,
                        strata=('BudgetCategory',),
                        family=family,
                    )
                )
                for budget in BUDGET_CATEGORIES:
                    scoped = [row for row in family_rows if row.get('BudgetCategory') == budget]
                    rows.append(
                        _embedding_metric_row(
                            metric=spec.metric_label,
                            metric_title=spec.title_label,
                            model=model,
                            scope='family_budget',
                            rows=scoped,
                            strata=(),
                            family=family,
                            budget=budget,
                        )
                    )
    return rows


def embedding_metric_range_rows(
    model_rows: Sequence[ReportRow],
    *,
    embedding_models: Sequence[str],
    complete_crossing: bool = True,
) -> list[MutableReportRow]:
    """Pool equal-weight model means and retain their min--max envelope."""
    grouped: dict[tuple[str, str, str, str], list[ReportRow]] = defaultdict(list)
    for row in model_rows:
        grouped[
            (
                str(row.get('MetricLabel') or ''),
                str(row.get('Scope') or ''),
                str(row.get('ExperimentFamilyLabel') or ''),
                str(row.get('BudgetCategory') or ''),
            )
        ].append(row)

    expected_models = set(embedding_models)
    output: list[MutableReportRow] = []
    value_fields = (
        'MeanTopK',
        'MeanMMR',
        'MeanFacLoc',
        'MeanDeltaFacLocMMR',
        'MeanDeltaFacLocTopK',
        'MeanDeltaMMRTopK',
    )
    for (metric, scope, family, budget), rows in sorted(grouped.items()):
        present_models = {str(row.get('EmbeddingModel') or '') for row in rows}
        complete = complete_crossing and present_models == expected_models
        out: MutableReportRow = {
            'MetricLabel': metric,
            'Metric': rows[0].get('Metric'),
            'Scope': scope,
            'ExperimentFamilyLabel': family or None,
            'BudgetCategory': budget or None,
            'BudgetCategoryLabel': rows[0].get('BudgetCategoryLabel'),
            'ExpectedModels': len(expected_models),
            'ObservedModels': len(present_models),
            'CompleteModelCrossing': complete,
        }
        for field in value_fields:
            values = [
                (str(row.get('EmbeddingModel') or ''), value)
                for row in rows
                if (value := float_or_none(row.get(field))) is not None
            ]
            if complete and len(values) == len(expected_models):
                minimum_model, minimum = min(values, key=lambda item: item[1])
                maximum_model, maximum = max(values, key=lambda item: item[1])
                out[field] = statistics.fmean(value for _, value in values)
                out[f'{field}MinModel'] = minimum
                out[f'{field}MaxModel'] = maximum
                out[f'{field}MinModelName'] = minimum_model
                out[f'{field}MaxModelName'] = maximum_model
            else:
                out[field] = None
                out[f'{field}MinModel'] = None
                out[f'{field}MaxModel'] = None
                out[f'{field}MinModelName'] = None
                out[f'{field}MaxModelName'] = None
        delta_values = [
            value
            for row in rows
            if (value := float_or_none(row.get('MeanDeltaFacLocMMR'))) is not None
        ]
        threshold = practical_effect_threshold(cast(DeltaMetricLabel, metric))
        out['PositiveModelCount'] = sum(value > 0.0 for value in delta_values) if complete else None
        out['PracticallyPositiveModelCount'] = (
            sum(value > threshold for value in delta_values) if complete else None
        )
        output.append(out)
    return output


def embedding_geometry_summary_rows(
    geometry_rows: Sequence[ReportRow],
) -> tuple[list[MutableReportRow], list[MutableReportRow]]:
    """Describe construct-audit outcomes within each representation space."""
    eligible = [
        row
        for row in geometry_rows
        if row.get('IncludeInFamilySummary') is True
        and str(row.get('ExperimentFamily') or '') != 'interaction'
    ]
    models = ordered_embedding_models(
        str(row.get('EmbeddingModel') or '') for row in eligible if row.get('EmbeddingModel')
    )
    metrics = (
        'GeometryPassRate',
        'QueryToGoldMeanMean',
        'QueryToNearMissMeanMean',
        'QueryToBackgroundOutlierMeanMean',
        'GoldMinusNearMissSimilarityMarginMean',
        'GoldMinusBackgroundOutlierSimilarityMarginMean',
    )
    model_rows: list[MutableReportRow] = []
    family_rows: list[MutableReportRow] = []
    for model in models:
        scoped = [row for row in eligible if row.get('EmbeddingModel') == model]
        out: MutableReportRow = {
            'EmbeddingModel': model,
            'EmbeddingModelLabel': embedding_model_display_label(model),
            'Cells': len(scoped),
            'Families': len({str(row.get('ExperimentFamilyLabel') or 'Unknown') for row in scoped}),
        }
        for metric in metrics:
            out[metric] = _balanced_mean(scoped, metric, ('ExperimentFamilyLabel',))
        for failure, target in (
            ('FailMissingFacetRate', 'FacetCompletenessPassRate'),
            ('FailWeakPrimaryAxisDominanceRate', 'PrimaryAxisStressPassRate'),
            (
                'FailExcessStressHorizonFacetCoverageRate',
                'EarlyFacetCoverageStressPassRate',
            ),
        ):
            mean_failure = _balanced_mean(scoped, failure, ('ExperimentFamilyLabel',))
            out[target] = None if mean_failure is None else 1.0 - mean_failure
        pass_rates = _values(scoped, 'GeometryPassRate')
        out['GeometryPassRateMinCell'] = min(pass_rates) if pass_rates else None
        out['GeometryPassRateMaxCell'] = max(pass_rates) if pass_rates else None
        model_rows.append(out)

        for family in sorted(
            {str(row.get('ExperimentFamilyLabel') or 'Unknown') for row in scoped}
        ):
            group = [
                row
                for row in scoped
                if str(row.get('ExperimentFamilyLabel') or 'Unknown') == family
            ]
            family_out: MutableReportRow = {
                'EmbeddingModel': model,
                'EmbeddingModelLabel': embedding_model_display_label(model),
                'ExperimentFamilyLabel': family,
                'Cells': len(group),
            }
            for metric in metrics:
                family_out[metric] = _mean(group, metric)
            family_rows.append(family_out)
    return model_rows, family_rows


def lambda_curve_by_embedding_model_rows(
    lambda_grid_rows: Sequence[ReportRow],
) -> list[MutableReportRow]:
    """Retain per-model validation sensitivity curves before suite pooling."""
    grouped: dict[tuple[str, str, float], list[ReportRow]] = defaultdict(list)
    for row in lambda_grid_rows:
        model = str(row.get('EmbeddingModel') or '')
        strategy = str(row.get('strategy') or '')
        lambda_norm = float_or_none(row.get('lambda_norm'))
        if model and strategy in DIVERSIFYING_STRATEGIES and lambda_norm is not None:
            grouped[(model, strategy, round(lambda_norm, 6))].append(row)

    output: list[MutableReportRow] = []
    for (model, strategy, lambda_norm), rows in sorted(grouped.items()):
        fcp_values = _values(rows, 'DeltaStrategyTopK_FCP')
        distractor_values = _values(rows, 'DeltaStrategyTopK_DistractorRate')
        if not fcp_values:
            continue
        output.append(
            {
                'EmbeddingModel': model,
                'EmbeddingModelLabel': embedding_model_display_label(model),
                'strategy': strategy,
                'lambda_norm': lambda_norm,
                'lam': _mean(rows, 'lam'),
                'CellRows': len(fcp_values),
                'FamilyCount': len({str(row.get('ExperimentFamily') or 'unknown') for row in rows}),
                'MeanDeltaStrategyTopK_FCP': _balanced_mean(
                    rows, 'DeltaStrategyTopK_FCP', ('ExperimentFamily',)
                ),
                'CellQ25DeltaStrategyTopK_FCP': quantile(sorted(fcp_values), 0.25),
                'CellQ75DeltaStrategyTopK_FCP': quantile(sorted(fcp_values), 0.75),
                'CellSafeLambdaFraction': sum(value >= 0.0 for value in fcp_values)
                / len(fcp_values),
                'MeanDeltaStrategyTopK_DistractorRate': _balanced_mean(
                    rows, 'DeltaStrategyTopK_DistractorRate', ('ExperimentFamily',)
                )
                if distractor_values
                else None,
            }
        )
    return output


def _embedding_metric_row(
    *,
    metric: str,
    metric_title: str,
    model: str,
    scope: str,
    rows: Sequence[ReportRow],
    strata: tuple[str, ...],
    family: str | None = None,
    budget: BudgetCategory | None = None,
) -> MutableReportRow:
    return {
        'MetricLabel': metric,
        'Metric': metric_title,
        'Scope': scope,
        'ExperimentFamilyLabel': family,
        'BudgetCategory': budget,
        'BudgetCategoryLabel': BUDGET_CATEGORY_LABELS.get(budget) if budget else None,
        'EmbeddingModel': model,
        'EmbeddingModelLabel': embedding_model_display_label(model),
        'Rows': len(rows),
        'MeanTopK': _balanced_mean(rows, f'TopK_{metric}', strata),
        'MeanMMR': _balanced_mean(rows, f'MMR_{metric}', strata),
        'MeanFacLoc': _balanced_mean(rows, f'FacLoc_{metric}', strata),
        'MeanDeltaFacLocMMR': _balanced_mean(rows, f'Delta_FacLoc_MMR_{metric}', strata),
        'MeanDeltaFacLocTopK': _balanced_mean(rows, f'Delta_FacLoc_TopK_{metric}', strata),
        'MeanDeltaMMRTopK': _balanced_mean(rows, f'Delta_MMR_TopK_{metric}', strata),
    }


def _balanced_mean(
    rows: Sequence[ReportRow],
    column: str,
    dimensions: tuple[str, ...],
) -> float | None:
    if not dimensions:
        return _mean(rows, column)
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = float_or_none(row.get(column))
        if value is None:
            continue
        key = tuple(str(row.get(dimension) or 'Unknown') for dimension in dimensions)
        grouped[key].append(value)
    means = [statistics.fmean(values) for values in grouped.values() if values]
    return statistics.fmean(means) if means else None


def _mean(rows: Sequence[ReportRow], column: str) -> float | None:
    values = _values(rows, column)
    return statistics.fmean(values) if values else None


def _values(rows: Sequence[ReportRow], column: str) -> list[float]:
    return [value for row in rows if (value := float_or_none(row.get(column))) is not None]


def _resolve_manifest_model(
    cell: SuiteManifestCell,
    embedding_models: Sequence[str],
) -> str | None:
    raw = cell.run_profile_factors.get('embedding', '')
    exact = [model for model in embedding_models if model == raw]
    if exact:
        return exact[0]
    suffix = [model for model in embedding_models if model.rsplit('/', 1)[-1] == raw]
    return suffix[0] if len(suffix) == 1 else None
