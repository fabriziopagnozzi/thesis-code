"""Cross-wording-configuration aggregate figures."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from experiments.medical_dataset_gen.analysis.analysis_constants import (
    DeltaMetricLabel,
    practical_effect_threshold,
)
from experiments.medical_dataset_gen.analysis.helpers import float_or_none, short_model_label
from experiments.medical_dataset_gen.analysis.models import BudgetCategory, PlotFormat
from experiments.medical_dataset_gen.analysis.report_config import (
    AGGREGATE_PLOT_EXCLUDED_FAMILY_LABELS,
    BUDGET_CATEGORIES,
    BUDGET_CATEGORY_LABELS,
    EMBEDDING_MODEL_FACETED_PLOT_MODELS,
    REPORT_METRIC_LABEL_SET,
    REPORT_METRIC_LABELS,
    REPORT_METRIC_SPECS,
)


@dataclass(frozen=True)
class DeltaPanelSpec:
    title: str
    value_field: str
    pct_field: str
    cmap: str
    colorbar_label: str


@dataclass(frozen=True)
class SummaryAxisSpec:
    row_keys: tuple[str, ...]
    column_keys: tuple[str, ...]
    row_tick_labels: tuple[str, ...]
    column_tick_labels: tuple[str, ...]
    row_field: str
    column_field: str


def plot_config_fcp_budget_delta_heatmaps(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    """Compact FCP view: wording configuration by retrieval budget."""
    summary_rows = _metric_config_budget_rows(rows, metric_filter='FCP')
    if not _has_multiple_configs(summary_rows):
        return []
    configs = _ordered_configs(summary_rows)
    budget_labels = _budget_labels()
    axis = SummaryAxisSpec(
        row_keys=tuple(configs),
        column_keys=tuple(budget_labels),
        row_tick_labels=tuple(_config_label_for_key(summary_rows, config) for config in configs),
        column_tick_labels=tuple(budget_labels),
        row_field='WordingConfig',
        column_field='BudgetCategoryLabel',
    )
    return _plot_delta_heatmap_panels(
        plt=plt,
        rows=summary_rows,
        axis=axis,
        title='FCP deltas by wording configuration and retrieval budget',
        footnote=(
            'Each cell aggregates all matching experiment families and embedding models. '
            'Top line = mean FCP delta; bottom line = win rate for the comparison.'
        ),
        output_path=output_dir / f'cross_config_fcp_budget_delta_heatmaps.{plot_format}',
        figsize=(15.0, max(4.8, 0.46 * len(configs) + 2.2)),
    )


def plot_config_metric_delta_heatmap_low_budget(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    """Compact metric view: metric by wording configuration at low budget."""
    summary_rows = [
        row
        for row in _metric_config_budget_rows(rows)
        if row.get('BudgetCategory') == 'low_budget'
        and str(row.get('MetricLabel') or '') in REPORT_METRIC_LABEL_SET
    ]
    if not _has_multiple_configs(summary_rows):
        return []
    metric_labels = _ordered_metric_labels(summary_rows)
    configs = _ordered_configs(summary_rows)
    axis = SummaryAxisSpec(
        row_keys=tuple(metric_labels),
        column_keys=tuple(configs),
        row_tick_labels=tuple(
            _metric_title_from_rows(summary_rows, metric) for metric in metric_labels
        ),
        column_tick_labels=tuple(_config_label_for_key(summary_rows, config) for config in configs),
        row_field='MetricLabel',
        column_field='WordingConfig',
    )
    return _plot_delta_heatmap_panels(
        plt=plt,
        rows=summary_rows,
        axis=axis,
        title='Low-budget metric deltas by wording configuration',
        footnote=(
            'Each cell aggregates all matching experiment families and embedding models. '
            'Top line = mean delta; bottom line = win rate for the comparison.'
        ),
        output_path=output_dir / f'cross_config_metric_delta_heatmap_low_budget.{plot_format}',
        figsize=(15.0, max(5.2, 0.58 * len(metric_labels) + 2.4)),
    )


def plot_config_fcp_family_budget_delta_heatmaps(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    """Detailed FCP view: wording configuration x family by budget."""
    summary_rows = _metric_config_family_budget_rows(rows, metric_filter='FCP')
    if not _has_multiple_configs(summary_rows):
        return []
    axis = _config_family_budget_axis(summary_rows)
    return _plot_delta_heatmap_panels(
        plt=plt,
        rows=summary_rows,
        axis=axis,
        title='FCP deltas by wording configuration, experiment family, and retrieval budget',
        footnote=(
            'Rows are grouped by wording configuration, then experiment family. '
            'Top line = mean FCP delta; bottom line = win rate for the comparison.'
        ),
        output_path=output_dir / f'cross_config_fcp_family_budget_delta_heatmaps.{plot_format}',
        figsize=(15.0, max(8.0, 0.31 * len(axis.row_keys) + 2.4)),
        row_group_boundaries=_row_group_boundaries(summary_rows, axis.row_keys, 'WordingConfig'),
    )


def plot_config_fcp_family_budget_delta_heatmaps_by_embedding_model(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    paths: list[Path] = []
    for model in _ordered_embedding_models(rows):
        model_rows = [row for row in rows if row.get('EmbeddingModel') == model]
        summary_rows = _metric_config_family_budget_rows(model_rows, metric_filter='FCP')
        if not _has_multiple_configs(summary_rows):
            continue
        axis = _config_family_budget_axis(summary_rows)
        paths.extend(
            _plot_delta_heatmap_panels(
                plt=plt,
                rows=summary_rows,
                axis=axis,
                title=(
                    'FCP deltas by wording configuration, experiment family, '
                    f'and retrieval budget for {short_model_label(model)}'
                ),
                footnote=(
                    'Rows are grouped by wording configuration, then experiment family. '
                    'Top line = mean FCP delta; bottom line = win rate for the comparison.'
                ),
                output_path=(
                    output_dir
                    / (
                        'cross_config_fcp_family_budget_delta_heatmaps_by_emb_model_'
                        f'{_filename_token(model)}.{plot_format}'
                    )
                ),
                figsize=(15.0, max(8.0, 0.31 * len(axis.row_keys) + 2.4)),
                row_group_boundaries=_row_group_boundaries(
                    summary_rows,
                    axis.row_keys,
                    'WordingConfig',
                ),
            )
        )
    return paths


def plot_config_metric_family_delta_heatmap_low_budget(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    """Detailed metric view: wording configuration x metric by family."""
    summary_rows = [
        row
        for row in _metric_config_family_budget_rows(rows)
        if row.get('BudgetCategory') == 'low_budget'
        and str(row.get('MetricLabel') or '') in REPORT_METRIC_LABEL_SET
    ]
    if not _has_multiple_configs(summary_rows):
        return []
    axis = _config_metric_family_axis(summary_rows)
    return _plot_delta_heatmap_panels(
        plt=plt,
        rows=summary_rows,
        axis=axis,
        title='Low-budget metric deltas by wording configuration and experiment family',
        footnote=(
            'Rows are grouped by wording configuration, then metric. '
            'Top line = mean delta; bottom line = win rate for the comparison.'
        ),
        output_path=output_dir / f'cross_config_metric_family_delta_heatmap_low_budget.{plot_format}',
        figsize=(15.0, max(9.5, 0.28 * len(axis.row_keys) + 2.4)),
        row_group_boundaries=_row_group_boundaries(summary_rows, axis.row_keys, 'WordingConfig'),
    )


def plot_config_metric_family_delta_heatmap_low_budget_by_embedding_model(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    paths: list[Path] = []
    for model in _ordered_embedding_models(rows):
        model_rows = [row for row in rows if row.get('EmbeddingModel') == model]
        summary_rows = [
            row
            for row in _metric_config_family_budget_rows(model_rows)
            if row.get('BudgetCategory') == 'low_budget'
            and str(row.get('MetricLabel') or '') in REPORT_METRIC_LABEL_SET
        ]
        if not _has_multiple_configs(summary_rows):
            continue
        axis = _config_metric_family_axis(summary_rows)
        paths.extend(
            _plot_delta_heatmap_panels(
                plt=plt,
                rows=summary_rows,
                axis=axis,
                title=(
                    'Low-budget metric deltas by wording configuration and '
                    f'experiment family for {short_model_label(model)}'
                ),
                footnote=(
                    'Rows are grouped by wording configuration, then metric. '
                    'Top line = mean delta; bottom line = win rate for the comparison.'
                ),
                output_path=(
                    output_dir
                    / (
                        'cross_config_metric_family_delta_heatmap_low_budget_by_emb_model_'
                        f'{_filename_token(model)}.{plot_format}'
                    )
                ),
                figsize=(15.0, max(9.5, 0.28 * len(axis.row_keys) + 2.4)),
                row_group_boundaries=_row_group_boundaries(
                    summary_rows,
                    axis.row_keys,
                    'WordingConfig',
                ),
            )
        )
    return paths


def _plot_delta_heatmap_panels(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    axis: SummaryAxisSpec,
    title: str,
    footnote: str,
    output_path: Path,
    figsize: tuple[float, float],
    row_group_boundaries: Sequence[int] = (),
) -> list[Path]:
    specs = _delta_panel_specs()
    matrices, values = _delta_matrices(
        rows=rows,
        axis=axis,
        specs=specs,
    )
    if not values:
        return []

    fig, axes_obj = plt.subplots(ncols=3, figsize=figsize, sharey=True)  # type: ignore[attr-defined]
    axes = cast(Sequence[Any], axes_obj)
    try:
        for ax, spec, matrix in zip(axes, specs, matrices, strict=True):
            image = ax.imshow(
                _matrix_with_nan(matrix),
                cmap=spec.cmap,
                norm=_symmetric_delta_norm(values),
                aspect='auto',
            )
            ax.set_title(spec.title)
            ax.set_xticks(range(len(axis.column_tick_labels)))
            ax.set_xticklabels(axis.column_tick_labels, rotation=35, ha='right')
            ax.set_yticks(range(len(axis.row_tick_labels)))
            ax.set_yticklabels(axis.row_tick_labels, fontsize=7)
            for boundary in row_group_boundaries:
                ax.axhline(boundary - 0.5, color='#111827', linewidth=0.8, alpha=0.55)
            for y_index, row_key in enumerate(axis.row_keys):
                for x_index, column_key in enumerate(axis.column_keys):
                    source_row = _find_summary_row(
                        rows,
                        axis.row_field,
                        row_key,
                        axis.column_field,
                        column_key,
                    )
                    value = matrix[y_index][x_index]
                    if value is None:
                        continue
                    pct = (
                        float_or_none(source_row.get(spec.pct_field))
                        if source_row is not None
                        else None
                    )
                    ax.text(
                        x_index,
                        y_index,
                        f'{value:+.3f}\n{pct:.0%}' if pct is not None else f'{value:+.3f}',
                        ha='center',
                        va='center',
                        fontsize=6.5,
                        color=_heatmap_text_color(value, values),
                    )
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(spec.colorbar_label)
        fig.suptitle(title, y=0.985)
        fig.text(0.5, 0.018, footnote, ha='center', va='bottom', fontsize=8, color='#303030')
        fig.tight_layout(rect=(0, 0.055, 1, 0.955))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
        return [output_path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def _metric_config_budget_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    metric_filter: DeltaMetricLabel | None = None,
) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for spec in REPORT_METRIC_SPECS:
        if metric_filter is not None and spec.metric_label != metric_filter:
            continue
        grouped: dict[tuple[str, BudgetCategory, str], list[Mapping[str, object]]] = {}
        for row in rows:
            config = str(row.get('WordingConfig') or '')
            category_value = str(row.get('BudgetCategory') or '')
            if not config or category_value not in BUDGET_CATEGORIES:
                continue
            category = cast(BudgetCategory, category_value)
            budget_label = str(row.get('BudgetCategoryLabel') or BUDGET_CATEGORY_LABELS[category])
            grouped.setdefault((config, category, budget_label), []).append(row)
        for (config, category, budget_label), group in grouped.items():
            first = group[0]
            summary_rows.append(
                _metric_summary_row(
                    metric=spec.metric_label,
                    metric_title=spec.title_label,
                    rows=group,
                    extra={
                        'WordingConfig': config,
                        'WordingConfigLabel': first.get('WordingConfigLabel'),
                        'QueryMode': first.get('QueryMode'),
                        'FocusMode': first.get('FocusMode'),
                        'ChunkTextMode': first.get('ChunkTextMode'),
                        'BudgetCategory': category,
                        'BudgetCategoryLabel': budget_label,
                    },
                )
            )
    summary_rows.sort(key=_config_budget_sort_key)
    return summary_rows


def _metric_config_family_budget_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    metric_filter: DeltaMetricLabel | None = None,
) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for spec in REPORT_METRIC_SPECS:
        if metric_filter is not None and spec.metric_label != metric_filter:
            continue
        grouped: dict[tuple[str, str, BudgetCategory, str], list[Mapping[str, object]]] = {}
        for row in rows:
            config = str(row.get('WordingConfig') or '')
            family = str(row.get('ExperimentFamilyLabel') or 'Unknown')
            category_value = str(row.get('BudgetCategory') or '')
            if (
                not config
                or category_value not in BUDGET_CATEGORIES
                or family in AGGREGATE_PLOT_EXCLUDED_FAMILY_LABELS
            ):
                continue
            category = cast(BudgetCategory, category_value)
            budget_label = str(row.get('BudgetCategoryLabel') or BUDGET_CATEGORY_LABELS[category])
            grouped.setdefault((config, family, category, budget_label), []).append(row)
        for (config, family, category, budget_label), group in grouped.items():
            first = group[0]
            row_key = _composite_key(config, family)
            metric_key = _composite_key(config, spec.metric_label)
            summary_rows.append(
                _metric_summary_row(
                    metric=spec.metric_label,
                    metric_title=spec.title_label,
                    rows=group,
                    extra={
                        'WordingConfig': config,
                        'WordingConfigLabel': first.get('WordingConfigLabel'),
                        'QueryMode': first.get('QueryMode'),
                        'FocusMode': first.get('FocusMode'),
                        'ChunkTextMode': first.get('ChunkTextMode'),
                        'ExperimentFamilyLabel': family,
                        'BudgetCategory': category,
                        'BudgetCategoryLabel': budget_label,
                        'ConfigFamilyKey': row_key,
                        'ConfigMetricKey': metric_key,
                    },
                )
            )
    summary_rows.sort(key=_config_family_budget_sort_key)
    return summary_rows


def _metric_summary_row(
    *,
    metric: DeltaMetricLabel,
    metric_title: str,
    rows: Sequence[Mapping[str, object]],
    extra: Mapping[str, object],
) -> dict[str, object]:
    delta_fm_col = f'Delta_FacLoc_MMR_{metric}'
    delta_ft_col = f'Delta_FacLoc_TopK_{metric}'
    delta_mt_col = f'Delta_MMR_TopK_{metric}'
    complete_rows = [row for row in rows if float_or_none(row.get(delta_fm_col)) is not None]
    deltas_fm = _numeric_values(complete_rows, delta_fm_col)
    deltas_ft = _numeric_values(complete_rows, delta_ft_col)
    deltas_mt = _numeric_values(complete_rows, delta_mt_col)
    threshold = practical_effect_threshold(metric)
    row_count = len(complete_rows)
    facloc_better = sum(delta > threshold for delta in deltas_fm)
    facloc_tied = sum(abs(delta) <= threshold for delta in deltas_fm)
    facloc_worse = sum(delta < -threshold for delta in deltas_fm)
    facloc_topk_better = sum(delta > 0.0 for delta in deltas_ft)
    mmr_topk_better = sum(delta > 0.0 for delta in deltas_mt)
    return {
        **dict(extra),
        'Metric': metric_title,
        'MetricLabel': metric,
        'Rows': row_count,
        'FacLocBetterPct': _fraction_or_none(facloc_better, row_count),
        'FacLocTiedPct': _fraction_or_none(facloc_tied, row_count),
        'FacLocWorsePct': _fraction_or_none(facloc_worse, row_count),
        'FacLocTopKBetterPct': _fraction_or_none(facloc_topk_better, row_count),
        'MMRTopKBetterPct': _fraction_or_none(mmr_topk_better, row_count),
        'MeanDeltaFacLocMMR': statistics.fmean(deltas_fm) if deltas_fm else None,
        'MeanDeltaFacLocTopK': statistics.fmean(deltas_ft) if deltas_ft else None,
        'MeanDeltaMMRTopK': statistics.fmean(deltas_mt) if deltas_mt else None,
    }


def _config_family_budget_axis(rows: Sequence[Mapping[str, object]]) -> SummaryAxisSpec:
    configs = _ordered_configs(rows)
    families = _ordered_families(rows)
    row_keys = tuple(
        _composite_key(config, family)
        for config in configs
        for family in families
        if _has_summary_row(rows, 'WordingConfig', config, 'ExperimentFamilyLabel', family)
    )
    labels = tuple(
        f'{_config_label_for_key(rows, config)} | {family}'
        for config in configs
        for family in families
        if _composite_key(config, family) in row_keys
    )
    return SummaryAxisSpec(
        row_keys=row_keys,
        column_keys=tuple(_budget_labels()),
        row_tick_labels=labels,
        column_tick_labels=tuple(_budget_labels()),
        row_field='ConfigFamilyKey',
        column_field='BudgetCategoryLabel',
    )


def _config_metric_family_axis(rows: Sequence[Mapping[str, object]]) -> SummaryAxisSpec:
    configs = _ordered_configs(rows)
    metrics = _ordered_metric_labels(rows)
    families = _ordered_families(rows)
    row_keys = tuple(
        _composite_key(config, metric)
        for config in configs
        for metric in metrics
        if _has_summary_row(rows, 'WordingConfig', config, 'MetricLabel', metric)
    )
    labels = tuple(
        f'{_config_label_for_key(rows, config)} | {_metric_title_from_rows(rows, metric)}'
        for config in configs
        for metric in metrics
        if _composite_key(config, metric) in row_keys
    )
    return SummaryAxisSpec(
        row_keys=row_keys,
        column_keys=tuple(families),
        row_tick_labels=labels,
        column_tick_labels=tuple(families),
        row_field='ConfigMetricKey',
        column_field='ExperimentFamilyLabel',
    )


def _delta_panel_specs() -> tuple[DeltaPanelSpec, ...]:
    return (
        DeltaPanelSpec(
            title='Mean FacLoc - MMR',
            value_field='MeanDeltaFacLocMMR',
            pct_field='FacLocBetterPct',
            cmap='RdBu',
            colorbar_label='Mean FacLoc - MMR delta',
        ),
        DeltaPanelSpec(
            title='Mean FacLoc - top-k',
            value_field='MeanDeltaFacLocTopK',
            pct_field='FacLocTopKBetterPct',
            cmap='RdYlGn',
            colorbar_label='Mean FacLoc - top-k delta',
        ),
        DeltaPanelSpec(
            title='Mean MMR - top-k',
            value_field='MeanDeltaMMRTopK',
            pct_field='MMRTopKBetterPct',
            cmap='RdYlGn',
            colorbar_label='Mean MMR - top-k delta',
        ),
    )


def _delta_matrices(
    *,
    rows: Sequence[Mapping[str, object]],
    axis: SummaryAxisSpec,
    specs: Sequence[DeltaPanelSpec],
) -> tuple[list[list[list[float | None]]], list[float]]:
    matrices: list[list[list[float | None]]] = []
    values: list[float] = []
    for spec in specs:
        matrix = _summary_matrix(
            rows=rows,
            row_keys=axis.row_keys,
            column_keys=axis.column_keys,
            row_field=axis.row_field,
            column_field=axis.column_field,
            value_field=spec.value_field,
        )
        matrices.append(matrix)
        values.extend(value for row in matrix for value in row if value is not None)
    return matrices, values


def _summary_matrix(
    *,
    rows: Sequence[Mapping[str, object]],
    row_keys: Sequence[str],
    column_keys: Sequence[str],
    row_field: str,
    column_field: str,
    value_field: str,
) -> list[list[float | None]]:
    by_key: dict[tuple[str, str], float | None] = {}
    for row in rows:
        row_key = str(row.get(row_field) or '')
        column_key = str(row.get(column_field) or '')
        if row_key and column_key:
            by_key[(row_key, column_key)] = float_or_none(row.get(value_field))
    return [
        [by_key.get((row_key, column_key)) for column_key in column_keys] for row_key in row_keys
    ]


def _find_summary_row(
    rows: Sequence[Mapping[str, object]],
    first_field: str,
    first_value: str,
    second_field: str,
    second_value: str,
) -> Mapping[str, object] | None:
    for row in rows:
        if str(row.get(first_field) or '') == first_value and str(row.get(second_field) or '') == second_value:
            return row
    return None


def _has_summary_row(
    rows: Sequence[Mapping[str, object]],
    first_field: str,
    first_value: str,
    second_field: str,
    second_value: str,
) -> bool:
    return _find_summary_row(rows, first_field, first_value, second_field, second_value) is not None


def _ordered_configs(rows: Sequence[Mapping[str, object]]) -> list[str]:
    configs = {str(row.get('WordingConfig') or '') for row in rows if row.get('WordingConfig')}
    return sorted(configs, key=_config_sort_key)


def _ordered_families(rows: Sequence[Mapping[str, object]]) -> list[str]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        family = str(row.get('ExperimentFamilyLabel') or '')
        value = float_or_none(row.get('MeanDeltaFacLocMMR'))
        if not family or value is None:
            continue
        grouped.setdefault(family, []).append(value)
    return [
        family
        for family, _mean in sorted(
            ((family, statistics.fmean(values)) for family, values in grouped.items() if values),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def _ordered_metric_labels(rows: Sequence[Mapping[str, object]]) -> list[str]:
    present = {str(row.get('MetricLabel') or '') for row in rows if row.get('MetricLabel')}
    order = {metric: index for index, metric in enumerate(REPORT_METRIC_LABELS)}
    return sorted(present, key=lambda metric: order.get(metric, len(order)))


def _ordered_embedding_models(rows: Sequence[Mapping[str, object]]) -> list[str]:
    available = {str(row.get('EmbeddingModel') or '') for row in rows if row.get('EmbeddingModel')}
    if EMBEDDING_MODEL_FACETED_PLOT_MODELS:
        return [model for model in EMBEDDING_MODEL_FACETED_PLOT_MODELS if model in available]
    return sorted(available)


def _row_group_boundaries(
    rows: Sequence[Mapping[str, object]],
    row_keys: Sequence[str],
    group_field: str,
) -> list[int]:
    if not row_keys:
        return []
    groups = {
        key: str(row.get(group_field) or '')
        for key in row_keys
        for row in rows
        if key in {str(row.get('ConfigFamilyKey') or ''), str(row.get('ConfigMetricKey') or '')}
    }
    boundaries: list[int] = []
    previous = groups.get(row_keys[0], '')
    for index, key in enumerate(row_keys[1:], start=1):
        current = groups.get(key, '')
        if current != previous:
            boundaries.append(index)
            previous = current
    return boundaries


def _budget_labels() -> list[str]:
    return [BUDGET_CATEGORY_LABELS[category] for category in BUDGET_CATEGORIES]


def _metric_title_from_rows(rows: Sequence[Mapping[str, object]], metric_label: str) -> str:
    for row in rows:
        if row.get('MetricLabel') == metric_label and row.get('Metric'):
            return str(row['Metric'])
    return metric_label


def _config_label_for_key(rows: Sequence[Mapping[str, object]], config: str) -> str:
    for row in rows:
        if row.get('WordingConfig') == config and row.get('WordingConfigLabel'):
            return str(row['WordingConfigLabel'])
    return config


def _config_sort_key(config: str) -> tuple[int, int, int, str]:
    parts = _config_parts(config)
    query_order = {'biased': 0, 'unbiased': 1}
    focus_order = {'list': 0, 'natural': 1}
    chunk_order = {'simple': 0, 'hardened': 1}
    return (
        query_order.get(parts[0], 99),
        focus_order.get(parts[1], 99),
        chunk_order.get(parts[2], 99),
        config,
    )


def _config_parts(config: str) -> tuple[str, str, str]:
    parts = config.split('_')
    if len(parts) >= 5 and parts[1] == 'q' and parts[3] == 'f':
        return (parts[0], parts[2], parts[4])
    return ('unknown', 'unknown', 'unknown')


def _config_budget_sort_key(row: Mapping[str, object]) -> tuple[tuple[int, int, int, str], int]:
    budget_order = {category: index for index, category in enumerate(BUDGET_CATEGORIES)}
    return (
        _config_sort_key(str(row.get('WordingConfig') or '')),
        budget_order.get(cast(BudgetCategory, row.get('BudgetCategory')), 99),
    )


def _config_family_budget_sort_key(
    row: Mapping[str, object],
) -> tuple[tuple[int, int, int, str], str, int, int]:
    budget_order = {category: index for index, category in enumerate(BUDGET_CATEGORIES)}
    metric_order = {metric: index for index, metric in enumerate(REPORT_METRIC_LABELS)}
    return (
        _config_sort_key(str(row.get('WordingConfig') or '')),
        str(row.get('ExperimentFamilyLabel') or ''),
        budget_order.get(cast(BudgetCategory, row.get('BudgetCategory')), 99),
        metric_order.get(cast(DeltaMetricLabel, row.get('MetricLabel')), 99),
    )


def _composite_key(left: str, right: str) -> str:
    return f'{left}::{right}'


def _numeric_values(rows: Sequence[Mapping[str, object]], column: str) -> list[float]:
    return [
        value for value in (float_or_none(row.get(column)) for row in rows) if value is not None
    ]


def _fraction_or_none(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _has_multiple_configs(rows: Sequence[Mapping[str, object]]) -> bool:
    return len({row.get('WordingConfig') for row in rows if row.get('WordingConfig')}) > 1


def _matrix_with_nan(matrix: Sequence[Sequence[float | None]]) -> list[list[float]]:
    return [[value if value is not None else math.nan for value in row] for row in matrix]


def _symmetric_delta_norm(values: Sequence[float]) -> Any:
    from matplotlib.colors import Normalize, TwoSlopeNorm

    max_abs = max((abs(value) for value in values), default=0.0)
    if max_abs <= 0:
        return Normalize(vmin=-1.0, vmax=1.0)
    return TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)


def _heatmap_text_color(value: float, values: Sequence[float]) -> str:
    max_abs = max((abs(item) for item in values), default=0.0)
    if max_abs <= 0:
        return '#202020'
    return '#FFFFFF' if abs(value) / max_abs >= 0.58 else '#202020'


def _filename_token(value: str) -> str:
    return (
        short_model_label(value)
        .lower()
        .replace('/', '_')
        .replace('-', '_')
        .replace('.', '_')
    )
