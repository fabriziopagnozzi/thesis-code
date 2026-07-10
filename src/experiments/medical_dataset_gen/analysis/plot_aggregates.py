from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from experiments.medical_dataset_gen.analysis.analysis_constants import DELTA_METRIC_LABELS
from experiments.medical_dataset_gen.analysis.helpers import _float_or_none
from experiments.medical_dataset_gen.analysis.models import (
    BudgetCategory,
    DeltaMetricPlotSpec,
    PlotFormat,
)
from experiments.medical_dataset_gen.analysis.plot_diagnostics import (
    _add_family_legend,
    _annotate_horizontal_values,
    _family_color_for_row,
    _family_grouped_rows,
)
from experiments.medical_dataset_gen.analysis.report_config import (
    AGGREGATE_METRIC_ORDER,
    AGGREGATE_PLOT_EXCLUDED_FAMILY_LABELS,
    BUDGET_CATEGORIES,
    BUDGET_CATEGORY_LABELS,
)


def _plot_metric_budget_outcomes(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    budget_order = {
        'all_k': 0,
        **{category: index + 1 for index, category in enumerate(BUDGET_CATEGORIES)},
    }
    plot_rows = [
        row
        for row in rows
        if _float_or_none(row.get('FacLocBetterPct')) is not None
        and _float_or_none(row.get('FacLocTiedPct')) is not None
        and _float_or_none(row.get('FacLocWorsePct')) is not None
    ]
    plot_rows = sorted(
        plot_rows,
        key=lambda row: (
            _metric_plot_order(str(row.get('MetricLabel') or row.get('Metric') or '')),
            budget_order.get(str(row.get('BudgetCategory') or ''), 99),
        ),
    )
    if not plot_rows:
        return []

    labels = [f'{row.get("Metric")} | {row.get("BudgetView")}' for row in plot_rows]
    better = [_float_or_none(row.get('FacLocBetterPct')) or 0.0 for row in plot_rows]
    tied = [_float_or_none(row.get('FacLocTiedPct')) or 0.0 for row in plot_rows]
    worse = [_float_or_none(row.get('FacLocWorsePct')) or 0.0 for row in plot_rows]
    mean_delta = [_float_or_none(row.get('MeanDeltaFacLocMMR')) for row in plot_rows]
    positions = list(range(len(plot_rows)))
    fig_height = max(5.5, 0.31 * len(plot_rows) + 1.6)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))  # type: ignore[attr-defined]
    try:
        ax.barh(positions, better, color='#287C8E', label='FacLoc > MMR')
        ax.barh(positions, tied, left=better, color='#B8BEC8', label='Tie')
        left_worse = [first + second for first, second in zip(better, tied, strict=True)]
        ax.barh(positions, worse, left=left_worse, color='#C44E52', label='FacLoc < MMR')
        ax.set_title('FacLoc-vs-MMR outcome profile by metric and retrieval budget')
        ax.set_xlabel('Share of experiment-k rows')
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1.17)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.2)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
        for position, parts in enumerate(zip(better, tied, worse, strict=True)):
            start = 0.0
            for width in parts:
                if width >= 0.075:
                    ax.text(
                        start + width / 2,
                        position,
                        f'{width:.0%}',
                        ha='center',
                        va='center',
                        fontsize=7,
                        color='#202020',
                    )
                start += width
            if mean_delta[position] is not None:
                ax.text(
                    1.02,
                    position,
                    f'{mean_delta[position]:+.3f}',
                    ha='left',
                    va='center',
                    fontsize=7,
                    color='#202020',
                )
        ax.text(1.02, -0.75, 'Mean F-M', ha='left', va='center', fontsize=8)
        fig.tight_layout()
        path = output_dir / f'aggregate_metric_budget_outcomes.{plot_format}'
        fig.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def _plot_metric_family_delta_heatmap(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    plot_rows = [
        row
        for row in rows
        if _float_or_none(row.get('MeanDeltaFacLocMMR')) is not None
        and str(row.get('MetricLabel') or '') in DELTA_METRIC_LABELS
        and _is_core_aggregate_family_row(row)
    ]
    if not plot_rows:
        return []

    metric_labels = _ordered_unique(
        [str(row.get('MetricLabel')) for row in plot_rows],
        key=_metric_plot_order,
    )
    families = _ordered_families_for_summary_rows(plot_rows)
    matrix = _summary_matrix(
        rows=plot_rows,
        row_keys=metric_labels,
        column_keys=families,
        row_field='MetricLabel',
        column_field='ExperimentFamilyLabel',
        value_field='MeanDeltaFacLocMMR',
    )
    values = [value for row in matrix for value in row if value is not None]
    if not values:
        return []

    fig, ax = plt.subplots(figsize=(10.5, 5.6))  # type: ignore[attr-defined]
    try:
        image = ax.imshow(
            _matrix_with_nan(matrix),
            cmap='RdBu',
            norm=_symmetric_delta_norm(values),
            aspect='auto',
        )
        ax.set_title('Mean FacLoc - MMR delta by metric and experiment family')
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(families, rotation=35, ha='right')
        ax.set_yticks(range(len(metric_labels)))
        ax.set_yticklabels(
            [_metric_title_from_rows(plot_rows, metric_label) for metric_label in metric_labels]
        )
        for y_index, metric_label in enumerate(metric_labels):
            for x_index, family in enumerate(families):
                row = _find_summary_row(
                    plot_rows,
                    MetricLabel=metric_label,
                    ExperimentFamilyLabel=family,
                )
                value = matrix[y_index][x_index]
                pct = _float_or_none(row.get('FacLocBetterPct')) if row is not None else None
                if value is None:
                    continue
                ax.text(
                    x_index,
                    y_index,
                    f'{value:+.3f}\n{pct:.0%}' if pct is not None else f'{value:+.3f}',
                    ha='center',
                    va='center',
                    fontsize=7,
                    color=_heatmap_text_color(value, values),
                )
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label('Mean FacLoc - MMR delta')
        fig.tight_layout()
        path = output_dir / f'aggregate_metric_family_delta_heatmap.{plot_format}'
        fig.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def _plot_fcp_family_budget_heatmaps(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    plot_rows = [
        row
        for row in rows
        if row.get('MetricLabel') == 'FCP'
        and str(row.get('BudgetCategory') or '') in BUDGET_CATEGORIES
        and _is_core_aggregate_family_row(row)
    ]
    if not plot_rows:
        return []
    families = _ordered_families_for_summary_rows(plot_rows)
    budget_labels = [BUDGET_CATEGORY_LABELS[category] for category in BUDGET_CATEGORIES]
    budget_label_by_category = {
        category: BUDGET_CATEGORY_LABELS[category] for category in BUDGET_CATEGORIES
    }
    matrices = [
        (
            'Mean FacLoc - MMR FCP',
            'MeanDeltaFacLocMMR',
            'FacLocBetterPct',
        ),
        (
            'Mean FacLoc - top-k FCP',
            'MeanDeltaFacLocTopK',
            'FacLocTopKBetterPct',
        ),
    ]
    all_values: list[float] = []
    matrix_values: list[list[list[float | None]]] = []
    for _title, value_field, _pct_field in matrices:
        matrix = _summary_matrix(
            rows=plot_rows,
            row_keys=families,
            column_keys=[budget_label_by_category[category] for category in BUDGET_CATEGORIES],
            row_field='ExperimentFamilyLabel',
            column_field='BudgetCategoryLabel',
            value_field=value_field,
        )
        matrix_values.append(matrix)
        all_values.extend(value for row in matrix for value in row if value is not None)
    if not all_values:
        return []

    fig, axes_obj = plt.subplots(ncols=2, figsize=(11.5, 6.2), sharey=True)  # type: ignore[attr-defined]
    axes = cast(Sequence[Any], axes_obj)
    try:
        for ax, (title, value_field, pct_field), matrix in zip(
            axes,
            matrices,
            matrix_values,
            strict=True,
        ):
            image = ax.imshow(
                _matrix_with_nan(matrix),
                cmap='RdBu',
                norm=_symmetric_delta_norm(all_values),
                aspect='auto',
            )
            ax.set_title(title)
            ax.set_xticks(range(len(budget_labels)))
            ax.set_xticklabels(budget_labels)
            ax.set_yticks(range(len(families)))
            ax.set_yticklabels(families)
            for y_index, family in enumerate(families):
                for x_index, category in enumerate(BUDGET_CATEGORIES):
                    budget_label = budget_label_by_category[category]
                    row = _find_summary_row(
                        plot_rows,
                        ExperimentFamilyLabel=family,
                        BudgetCategoryLabel=budget_label,
                    )
                    value = _float_or_none(row.get(value_field)) if row is not None else None
                    pct = _float_or_none(row.get(pct_field)) if row is not None else None
                    if value is None:
                        continue
                    ax.text(
                        x_index,
                        y_index,
                        f'{value:+.3f}\n{pct:.0%}' if pct is not None else f'{value:+.3f}',
                        ha='center',
                        va='center',
                        fontsize=7,
                        color=_heatmap_text_color(value, all_values),
                    )
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Mean FCP delta')
        fig.suptitle('FCP aggregation by experiment family and retrieval budget', y=0.98)
        fig.text(
            0.5,
            0.02,
            'Cell format: top line = mean FCP delta. Bottom line = FacLoc win rate '
            'within that family-budget group: left panel uses FacLoc - MMR FCP > 0.05; '
            'right panel uses FacLoc - top-k FCP > 0.',
            ha='center',
            va='bottom',
            fontsize=8,
            color='#303030',
        )
        fig.tight_layout(rect=(0, 0.07, 1, 0.95))
        path = output_dir / f'fcp_family_budget_delta_heatmaps.{plot_format}'
        fig.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def _plot_budget_delta_columns(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    category: BudgetCategory,
    spec: DeltaMetricPlotSpec,
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    value_columns = (
        f'Delta_FacLoc_MMR_{spec.metric_label}',
        f'Delta_FacLoc_TopK_{spec.metric_label}',
    )
    plot_rows = [
        row
        for row in rows
        if all(_float_or_none(row.get(column)) is not None for column in value_columns)
    ]
    if not plot_rows:
        return []
    plot_rows = _family_grouped_rows(plot_rows, value_columns[0])
    labels = [str(row.get('ShortExperiment') or row.get('Experiment')) for row in plot_rows]
    colors = [_family_color_for_row(row) for row in plot_rows]
    category_label = BUDGET_CATEGORY_LABELS[category]
    series = [
        (
            f'FacLoc - MMR {spec.title_label}',
            [_float_or_none(row.get(value_columns[0])) or 0.0 for row in plot_rows],
        ),
        (
            f'FacLoc - top-k {spec.title_label}',
            [_float_or_none(row.get(value_columns[1])) or 0.0 for row in plot_rows],
        ),
    ]
    fig_height = max(5.0, 0.28 * len(labels) + 1.6)
    fig, axes_obj = plt.subplots(  # type: ignore[attr-defined]
        ncols=2,
        sharey=True,
        figsize=(15.0, fig_height),
    )
    axes = cast(Sequence[Any], axes_obj)
    try:
        for ax, (title, values) in zip(axes, series, strict=True):
            _draw_family_delta_bars(ax=ax, labels=labels, values=values, colors=colors)
            ax.set_title(title)
            ax.set_xlabel(f'{category_label} delta')
        _add_family_legend(fig=fig, rows=plot_rows)
        fig.suptitle(f'{category_label} {spec.title_label} deltas by experiment', y=0.995)
        fig.tight_layout(rect=(0, 0.03, 1, 0.985))
        path = output_dir / f'{spec.filename_token}_{category}_deltas_by_experiment.{plot_format}'
        fig.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def _metric_plot_order(metric_label: str) -> int:
    order = {label: index for index, label in enumerate(AGGREGATE_METRIC_ORDER)}
    return order.get(metric_label, len(order))


def _is_core_aggregate_family_row(row: Mapping[str, object]) -> bool:
    family = str(row.get('ExperimentFamilyLabel') or '')
    return family not in AGGREGATE_PLOT_EXCLUDED_FAMILY_LABELS


def _ordered_unique(values: Iterable[str], *, key: object | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    if key is None:
        return sorted(out)
    key_func = cast(Any, key)
    return sorted(out, key=key_func)


def _ordered_families_for_summary_rows(rows: Sequence[Mapping[str, object]]) -> list[str]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        family = str(row.get('ExperimentFamilyLabel') or '')
        value = _float_or_none(row.get('MeanDeltaFacLocMMR'))
        if not family or value is None:
            continue
        grouped.setdefault(family, []).append(value)
    return [
        family
        for family, _mean_value in sorted(
            ((family, statistics.fmean(values)) for family, values in grouped.items() if values),
            key=lambda item: (-item[1], item[0]),
        )
    ]


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
            by_key[(row_key, column_key)] = _float_or_none(row.get(value_field))
    return [
        [by_key.get((row_key, column_key)) for column_key in column_keys] for row_key in row_keys
    ]


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


def _metric_title_from_rows(rows: Sequence[Mapping[str, object]], metric_label: str) -> str:
    for row in rows:
        if row.get('MetricLabel') == metric_label and row.get('Metric'):
            return str(row.get('Metric'))
    return metric_label


def _find_summary_row(
    rows: Sequence[Mapping[str, object]],
    **fields: str,
) -> Mapping[str, object] | None:
    for row in rows:
        if all(str(row.get(field) or '') == value for field, value in fields.items()):
            return row
    return None


def _draw_family_delta_bars(
    *,
    ax: Any,
    labels: Sequence[str],
    values: Sequence[float],
    colors: Sequence[str],
) -> None:
    positions = list(range(len(labels)))
    ax.barh(positions, values, color=colors)
    ax.axvline(0.0, color='#303030', linewidth=0.9)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.25)
    _annotate_horizontal_values(ax=ax, values=values)
