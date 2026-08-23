from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from experiments.medical_dataset_gen.reports.analysis_constants import (
    EXPERIMENT_FAMILIES,
    EXPERIMENT_FAMILY_COLORS,
    EXPERIMENT_FAMILY_LABELS,
    DeltaMetricLabel,
    ExperimentFamilyId,
    practical_effect_threshold,
)
from experiments.medical_dataset_gen.reports.helpers import (
    experiment_plot_label,
    float_or_none,
    ordered_embedding_models_for_rows,
    short_model_label,
)
from experiments.medical_dataset_gen.reports.models import (
    BudgetCategory,
    DeltaMetricPlotSpec,
    PlotFormat,
)
from experiments.medical_dataset_gen.reports.plot_diagnostics import (
    annotate_horizontal_values,
    family_color_for_row,
)
from experiments.medical_dataset_gen.reports.plot_rendering import (
    set_axis_title,
    set_figure_title,
    title_aware_layout_top,
)
from experiments.medical_dataset_gen.reports.report_config import (
    AGGREGATE_PLOT_EXCLUDED_FAMILY_LABELS,
    BUDGET_CATEGORIES,
    BUDGET_CATEGORY_LABELS,
    DELTA_HEATMAP_ABS_SCALE_BY_VALUE_FIELD,
    DELTA_HEATMAP_DEFAULT_ABS_SCALE,
    REPORT_METRIC_LABEL_SET,
    REPORT_METRIC_LABELS,
    REPORT_METRIC_SPECS,
)


@dataclass(frozen=True)
class DeltaHeatmapSpec:
    title: str
    value_field: str
    raw_value_field: str
    pct_field: str
    cmap: str
    colorbar_label: str


def plot_metric_budget_outcomes(
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
        if float_or_none(row.get('FacLocBetterPct')) is not None
        and float_or_none(row.get('FacLocTiedPct')) is not None
        and float_or_none(row.get('FacLocWorsePct')) is not None
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
    better = [float_or_none(row.get('FacLocBetterPct')) or 0.0 for row in plot_rows]
    tied = [float_or_none(row.get('FacLocTiedPct')) or 0.0 for row in plot_rows]
    worse = [float_or_none(row.get('FacLocWorsePct')) or 0.0 for row in plot_rows]
    mean_delta = [float_or_none(row.get('MeanDeltaFacLocMMR')) for row in plot_rows]
    positions = list(range(len(plot_rows)))
    fig_height = max(5.5, 0.31 * len(plot_rows) + 1.6)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))  # type: ignore[attr-defined]
    try:
        ax.barh(positions, better, color='#287C8E', label='FacLoc > MMR')
        ax.barh(positions, tied, left=better, color='#B8BEC8', label='Tie')
        left_worse = [first + second for first, second in zip(better, tied, strict=True)]
        ax.barh(positions, worse, left=left_worse, color='#C44E52', label='FacLoc < MMR')
        set_axis_title(
            axis=ax,
            title='FacLoc-vs-MMR outcome profile by metric and retrieval budget',
        )
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


def plot_metric_family_delta_heatmap_low_budget(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    plot_rows = [
        row
        for row in rows
        if float_or_none(row.get('MeanDeltaFacLocMMR')) is not None
        and row.get('BudgetCategory') == 'low_budget'
        and str(row.get('MetricLabel') or '') in REPORT_METRIC_LABEL_SET
        and _is_core_aggregate_family_row(row)
    ]
    if not plot_rows:
        return []

    metric_labels = _ordered_unique(
        [str(row.get('MetricLabel')) for row in plot_rows],
        key=_metric_plot_order,
    )
    families = _ordered_families_for_summary_rows(plot_rows)
    specs = _delta_heatmap_specs(metric_suffix='')
    matrices, values = _delta_heatmap_matrices(
        rows=plot_rows,
        row_keys=metric_labels,
        column_keys=families,
        row_field='MetricLabel',
        column_field='ExperimentFamilyLabel',
        specs=specs,
    )
    if not values:
        return []

    fig, axes_obj = plt.subplots(ncols=3, figsize=(15.0, 5.8), sharey=True)  # type: ignore[attr-defined]
    axes = cast(Sequence[Any], axes_obj)
    try:
        _draw_delta_heatmap_row(
            fig=fig,
            axes=axes,
            rows=plot_rows,
            row_keys=metric_labels,
            column_keys=families,
            row_field='MetricLabel',
            column_field='ExperimentFamilyLabel',
            row_tick_labels=[
                _metric_title_from_rows(plot_rows, metric_label) for metric_label in metric_labels
            ],
            column_tick_labels=families,
            specs=specs,
            matrices=matrices,
            values=values,
            show_y_tick_labels=True,
        )
        set_figure_title(
            figure=fig,
            title='Low-budget metric deltas by experiment family',
            y=0.98,
        )
        fig.text(
            0.5,
            0.02,
            'Cell lines: raw mean for the first strategy; bold improvement delta; win rate. '
            'FacLoc - MMR uses the metric-specific practical effect threshold, top-k '
            'comparisons use > 0.',
            ha='center',
            va='bottom',
            fontsize=8,
            color='#303030',
        )
        fig.tight_layout(rect=(0, 0.08, 1, title_aware_layout_top(titled_top=0.93)))
        path = output_dir / f'metric_family_delta_heatmap_low_budget.{plot_format}'
        fig.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def plot_metric_family_delta_heatmap_by_embedding_model(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    plot_rows = [
        row
        for row in _metric_family_budget_rows_by_embedding(rows)
        if row.get('BudgetCategory') == 'low_budget'
        and float_or_none(row.get('MeanDeltaFacLocMMR')) is not None
        and str(row.get('MetricLabel') or '') in REPORT_METRIC_LABEL_SET
        and _is_core_aggregate_family_row(row)
    ]
    if not plot_rows:
        return []

    models = _ordered_embedding_models(plot_rows)
    metric_labels = _ordered_unique(
        [str(row.get('MetricLabel')) for row in plot_rows],
        key=_metric_plot_order,
    )
    families = _ordered_families_for_summary_rows(plot_rows)
    specs = _delta_heatmap_specs(metric_suffix='')
    matrices_by_model: list[list[list[list[float | None]]]] = []
    all_values: list[float] = []
    for model in models:
        model_rows = [row for row in plot_rows if row.get('EmbeddingModel') == model]
        matrices, values = _delta_heatmap_matrices(
            rows=model_rows,
            row_keys=metric_labels,
            column_keys=families,
            row_field='MetricLabel',
            column_field='ExperimentFamilyLabel',
            specs=specs,
        )
        matrices_by_model.append(matrices)
        all_values.extend(values)
    if not all_values:
        return []

    fig_height = _embedding_heatmap_fig_height(
        model_count=len(models),
        matrix_row_count=len(metric_labels),
    )
    fig, axes_obj = plt.subplots(  # type: ignore[attr-defined]
        nrows=len(models),
        ncols=3,
        figsize=(15.0, fig_height),
        squeeze=False,
        sharex=True,
        sharey=True,
        gridspec_kw={'hspace': 0.86},
    )
    axes_grid = cast(Sequence[Sequence[Any]], axes_obj)
    try:
        row_title_axes: list[tuple[Sequence[Any], str]] = []
        for row_index, (model, matrices) in enumerate(zip(models, matrices_by_model, strict=True)):
            axes = axes_grid[row_index]
            model_rows = [row for row in plot_rows if row.get('EmbeddingModel') == model]
            _draw_delta_heatmap_row(
                fig=fig,
                axes=axes,
                rows=model_rows,
                row_keys=metric_labels,
                column_keys=families,
                row_field='MetricLabel',
                column_field='ExperimentFamilyLabel',
                row_tick_labels=[
                    _metric_title_from_rows(plot_rows, metric_label)
                    for metric_label in metric_labels
                ],
                column_tick_labels=families,
                specs=specs,
                matrices=matrices,
                values=all_values,
                show_y_tick_labels=True,
                show_titles=row_index == 0,
            )
            row_title_axes.append((axes, short_model_label(model)))
        set_figure_title(
            figure=fig,
            title='Low-budget metric deltas by experiment family and embedding model',
            y=1.04,
        )
        fig.subplots_adjust(
            left=0.08,
            right=0.98,
            bottom=0.22,
            top=title_aware_layout_top(titled_top=0.82, untitled_top=0.94),
            hspace=0.86,
            wspace=0.38,
        )
        for axes, row_title in row_title_axes:
            _add_embedding_heatmap_row_title(fig=fig, axes=axes, title=row_title)
        path = output_dir / f'metric_family_delta_heatmap_by_emb_model.{plot_format}'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def plot_metric_family_delta_heatmap_low_budget_best_embedding_model(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    summary_rows = [
        row
        for row in _metric_family_budget_rows_by_embedding(rows)
        if row.get('BudgetCategory') == 'low_budget'
        and float_or_none(row.get('MeanDeltaFacLocMMR')) is not None
        and str(row.get('MetricLabel') or '') in REPORT_METRIC_LABEL_SET
        and _is_core_aggregate_family_row(row)
    ]
    if not summary_rows:
        return []

    candidate_models = set(_ordered_embedding_models(summary_rows))
    selection_rows = [
        row
        for row in summary_rows
        if row.get('EmbeddingModel') in candidate_models
        and row.get('MetricLabel') == 'FCP'
        and row.get('ExperimentFamilyLabel') == EXPERIMENT_FAMILY_LABELS['balanced_clean']
    ]
    if not selection_rows:
        return []

    best_row = max(
        selection_rows,
        key=lambda row: _float_sort_key(row.get('MeanDeltaFacLocMMR')),
    )
    best_model = str(best_row.get('EmbeddingModel') or '')
    plot_rows = [row for row in summary_rows if row.get('EmbeddingModel') == best_model]
    if not plot_rows:
        return []

    metric_labels = _ordered_unique(
        [str(row.get('MetricLabel')) for row in plot_rows],
        key=_metric_plot_order,
    )
    families = _ordered_families_for_summary_rows(plot_rows)
    specs = _delta_heatmap_specs(metric_suffix='')
    matrices, values = _delta_heatmap_matrices(
        rows=plot_rows,
        row_keys=metric_labels,
        column_keys=families,
        row_field='MetricLabel',
        column_field='ExperimentFamilyLabel',
        specs=specs,
    )
    if not values:
        return []

    fig, axes_obj = plt.subplots(ncols=3, figsize=(15.0, 5.8), sharey=True)  # type: ignore[attr-defined]
    axes = cast(Sequence[Any], axes_obj)
    try:
        _draw_delta_heatmap_row(
            fig=fig,
            axes=axes,
            rows=plot_rows,
            row_keys=metric_labels,
            column_keys=families,
            row_field='MetricLabel',
            column_field='ExperimentFamilyLabel',
            row_tick_labels=[
                _metric_title_from_rows(plot_rows, metric_label) for metric_label in metric_labels
            ],
            column_tick_labels=families,
            specs=specs,
            matrices=matrices,
            values=values,
            show_y_tick_labels=True,
        )
        set_figure_title(
            figure=fig,
            title=f'Low-budget metric deltas by experiment family for {short_model_label(best_model)}',
            y=0.98,
        )
        fig.text(
            0.5,
            0.02,
            'Embedding model selected by the Balanced clean / FCP cell. Cell lines: raw mean '
            'for the first strategy; bold improvement delta; win rate. FacLoc - MMR uses the '
            'metric-specific practical effect threshold, top-k comparisons use > 0.',
            ha='center',
            va='bottom',
            fontsize=8,
            color='#303030',
        )
        fig.tight_layout(rect=(0, 0.08, 1, title_aware_layout_top(titled_top=0.93)))
        path = output_dir / f'metric_family_delta_heatmap_low_budget_best_emb_model.{plot_format}'
        fig.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def plot_fcp_family_budget_heatmaps(
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
    specs = _delta_heatmap_specs(metric_suffix=' FCP')
    matrix_values, values = _delta_heatmap_matrices(
        rows=plot_rows,
        row_keys=families,
        column_keys=[budget_label_by_category[category] for category in BUDGET_CATEGORIES],
        row_field='ExperimentFamilyLabel',
        column_field='BudgetCategoryLabel',
        specs=specs,
    )
    if not values:
        return []

    fig, axes_obj = plt.subplots(ncols=3, figsize=(15.0, 6.2), sharey=True)  # type: ignore[attr-defined]
    axes = cast(Sequence[Any], axes_obj)
    try:
        _draw_delta_heatmap_row(
            fig=fig,
            axes=axes,
            rows=plot_rows,
            row_keys=families,
            column_keys=[budget_label_by_category[category] for category in BUDGET_CATEGORIES],
            row_field='ExperimentFamilyLabel',
            column_field='BudgetCategoryLabel',
            row_tick_labels=families,
            column_tick_labels=budget_labels,
            specs=specs,
            matrices=matrix_values,
            values=values,
            show_y_tick_labels=True,
        )
        set_figure_title(
            figure=fig,
            title='FCP aggregation by experiment family and retrieval budget',
            y=0.98,
        )
        fig.text(
            0.5,
            0.02,
            'Cell lines: raw mean for the first strategy; bold FCP improvement delta; win rate. '
            'Left panel uses FacLoc - MMR FCP > 0.05; top-k comparison panels use > 0.',
            ha='center',
            va='bottom',
            fontsize=8,
            color='#303030',
        )
        fig.tight_layout(rect=(0, 0.07, 1, title_aware_layout_top(titled_top=0.95)))
        path = output_dir / f'fcp_family_budget_delta_heatmaps.{plot_format}'
        fig.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def plot_fcp_family_budget_heatmaps_by_embedding_model(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    plot_rows = [
        row
        for row in _metric_family_budget_rows_by_embedding(rows)
        if row.get('MetricLabel') == 'FCP'
        and str(row.get('BudgetCategory') or '') in BUDGET_CATEGORIES
        and _is_core_aggregate_family_row(row)
    ]
    if not plot_rows:
        return []

    models = _ordered_embedding_models(plot_rows)
    families = _ordered_families_for_summary_rows(plot_rows)
    budget_labels = [BUDGET_CATEGORY_LABELS[category] for category in BUDGET_CATEGORIES]
    budget_label_by_category = {
        category: BUDGET_CATEGORY_LABELS[category] for category in BUDGET_CATEGORIES
    }
    specs = _delta_heatmap_specs(metric_suffix=' FCP')
    matrices_by_model: list[list[list[list[float | None]]]] = []
    all_values: list[float] = []
    column_keys = [budget_label_by_category[category] for category in BUDGET_CATEGORIES]
    for model in models:
        model_rows = [row for row in plot_rows if row.get('EmbeddingModel') == model]
        matrices, values = _delta_heatmap_matrices(
            rows=model_rows,
            row_keys=families,
            column_keys=column_keys,
            row_field='ExperimentFamilyLabel',
            column_field='BudgetCategoryLabel',
            specs=specs,
        )
        matrices_by_model.append(matrices)
        all_values.extend(values)
    if not all_values:
        return []

    fig_height = _embedding_heatmap_fig_height(
        model_count=len(models),
        matrix_row_count=len(families),
    )
    fig, axes_obj = plt.subplots(  # type: ignore[attr-defined]
        nrows=len(models),
        ncols=3,
        figsize=(15.0, fig_height),
        squeeze=False,
        sharex=True,
        sharey=True,
        gridspec_kw={'hspace': 0.86},
    )
    axes_grid = cast(Sequence[Sequence[Any]], axes_obj)
    try:
        row_title_axes: list[tuple[Sequence[Any], str]] = []
        for row_index, (model, matrices) in enumerate(zip(models, matrices_by_model, strict=True)):
            axes = axes_grid[row_index]
            model_rows = [row for row in plot_rows if row.get('EmbeddingModel') == model]
            _draw_delta_heatmap_row(
                fig=fig,
                axes=axes,
                rows=model_rows,
                row_keys=families,
                column_keys=column_keys,
                row_field='ExperimentFamilyLabel',
                column_field='BudgetCategoryLabel',
                row_tick_labels=families,
                column_tick_labels=budget_labels,
                specs=specs,
                matrices=matrices,
                values=all_values,
                show_y_tick_labels=True,
                show_titles=row_index == 0,
            )
            row_title_axes.append((axes, short_model_label(model)))
        set_figure_title(
            figure=fig,
            title='FCP aggregation by experiment family, retrieval budget, and embedding model',
            y=1.04,
        )
        fig.subplots_adjust(
            left=0.08,
            right=0.98,
            bottom=0.22,
            top=title_aware_layout_top(titled_top=0.82, untitled_top=0.94),
            hspace=0.86,
            wspace=0.38,
        )
        for axes, row_title in row_title_axes:
            _add_embedding_heatmap_row_title(fig=fig, axes=axes, title=row_title)
        path = output_dir / f'fcp_family_budget_delta_heatmaps_by_emb_model.{plot_format}'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def plot_budget_delta_columns(
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
        if all(float_or_none(row.get(column)) is not None for column in value_columns)
    ]
    if not plot_rows:
        return []
    plot_rows = _budget_delta_grouped_rows(plot_rows, value_columns[0])
    labels = [experiment_plot_label(row) for row in plot_rows]
    parent_keys = [_parent_experiment_key(row) for row in plot_rows]
    colors = [family_color_for_row(row) for row in plot_rows]
    category_label = BUDGET_CATEGORY_LABELS[category]
    series = [
        (
            f'FacLoc - MMR {spec.title_label}',
            [float_or_none(row.get(value_columns[0])) or 0.0 for row in plot_rows],
        ),
        (
            f'FacLoc - top-k {spec.title_label}',
            [float_or_none(row.get(value_columns[1])) or 0.0 for row in plot_rows],
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
            _draw_family_delta_bars(
                ax=ax,
                labels=labels,
                values=values,
                colors=colors,
                parent_keys=parent_keys,
            )
            set_axis_title(axis=ax, title=title)
            ax.set_xlabel(f'{category_label} delta')
        _add_budget_delta_family_legend(fig=fig, rows=plot_rows)
        set_figure_title(
            figure=fig,
            title=f'{category_label} {spec.title_label} deltas by experiment',
            y=0.995,
        )
        fig.text(
            0.5,
            0.018,
            'Alternating light blue and light grey bands group rows from the same parent experiment, with different embedding models. Notice: qwen3-0.6B seems the best model.',
            ha='center',
            va='bottom',
            fontsize=11,
            color='#303030',
        )
        fig.tight_layout(
            # The remaining top margin accommodates the family legend, not a title.
            rect=(0, 0.075, 1, title_aware_layout_top(titled_top=0.91)),
        )
        path = output_dir / f'{spec.filename_token}_{category}_deltas_by_experiment.{plot_format}'
        fig.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def _delta_heatmap_specs(*, metric_suffix: str) -> tuple[DeltaHeatmapSpec, ...]:
    return (
        DeltaHeatmapSpec(
            title=f'Mean FacLoc - MMR{metric_suffix}',
            value_field='MeanDeltaFacLocMMR',
            raw_value_field='MeanFacLoc',
            pct_field='FacLocBetterPct',
            cmap='RdBu',
            colorbar_label='Mean improvement delta',
        ),
        DeltaHeatmapSpec(
            title=f'Mean FacLoc - top-k{metric_suffix}',
            value_field='MeanDeltaFacLocTopK',
            raw_value_field='MeanFacLoc',
            pct_field='FacLocTopKBetterPct',
            cmap='RdYlGn',
            colorbar_label='Mean improvement delta',
        ),
        DeltaHeatmapSpec(
            title=f'Mean MMR - top-k{metric_suffix}',
            value_field='MeanDeltaMMRTopK',
            raw_value_field='MeanMMR',
            pct_field='MMRTopKBetterPct',
            cmap='RdYlGn',
            colorbar_label='Mean improvement delta',
        ),
    )


def _delta_heatmap_matrices(
    *,
    rows: Sequence[Mapping[str, object]],
    row_keys: Sequence[str],
    column_keys: Sequence[str],
    row_field: str,
    column_field: str,
    specs: Sequence[DeltaHeatmapSpec],
) -> tuple[list[list[list[float | None]]], list[float]]:
    matrices: list[list[list[float | None]]] = []
    values: list[float] = []
    for spec in specs:
        matrix = _summary_matrix(
            rows=rows,
            row_keys=row_keys,
            column_keys=column_keys,
            row_field=row_field,
            column_field=column_field,
            value_field=spec.value_field,
        )
        matrices.append(matrix)
        values.extend(value for row in matrix for value in row if value is not None)
    return matrices, values


def _draw_delta_heatmap_row(
    *,
    fig: Any,
    axes: Sequence[Any],
    rows: Sequence[Mapping[str, object]],
    row_keys: Sequence[str],
    column_keys: Sequence[str],
    row_field: str,
    column_field: str,
    row_tick_labels: Sequence[str],
    column_tick_labels: Sequence[str],
    specs: Sequence[DeltaHeatmapSpec],
    matrices: Sequence[Sequence[Sequence[float | None]]],
    values: Sequence[float],
    show_y_tick_labels: bool,
    show_titles: bool = True,
) -> None:
    for ax, spec, matrix in zip(axes, specs, matrices, strict=True):
        image = ax.imshow(
            _matrix_with_nan(matrix),
            cmap=spec.cmap,
            norm=_symmetric_delta_norm(spec.value_field),
            aspect='auto',
        )
        if show_titles:
            set_axis_title(axis=ax, title=spec.title)
        ax.set_xticks(range(len(column_tick_labels)))
        ax.set_xticklabels(column_tick_labels, rotation=35, ha='right')
        ax.set_yticks(range(len(row_tick_labels)))
        ax.set_yticklabels(row_tick_labels if show_y_tick_labels else [''] * len(row_tick_labels))
        for y_index, row_key in enumerate(row_keys):
            for x_index, column_key in enumerate(column_keys):
                source_row = _find_summary_row(
                    rows,
                    **{row_field: row_key, column_field: column_key},
                )
                value = matrix[y_index][x_index]
                pct = (
                    float_or_none(source_row.get(spec.pct_field))
                    if source_row is not None
                    else None
                )
                if value is None:
                    continue
                raw_value = (
                    float_or_none(source_row.get(spec.raw_value_field))
                    if source_row is not None
                    else None
                )
                _annotate_delta_heatmap_cell(
                    ax=ax,
                    x_index=x_index,
                    y_index=y_index,
                    raw_value=raw_value,
                    delta=value,
                    pct=pct,
                    value_field=spec.value_field,
                )
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, extend='both')
        cbar.set_label(spec.colorbar_label)


def _annotate_delta_heatmap_cell(
    *,
    ax: Any,
    x_index: int,
    y_index: int,
    raw_value: float | None,
    delta: float,
    pct: float | None,
    value_field: str,
) -> None:
    color = _heatmap_text_color(delta, value_field)
    if raw_value is not None:
        ax.text(
            x_index,
            y_index - 0.27,
            f'{raw_value:.3f}',
            ha='center',
            va='center',
            fontsize=5.8,
            color=color,
        )
    ax.text(
        x_index,
        y_index,
        f'{delta:+.3f}',
        ha='center',
        va='center',
        fontsize=6.6,
        fontweight='bold',
        color=color,
    )
    if pct is not None:
        ax.text(
            x_index,
            y_index + 0.27,
            f'{pct:.0%}',
            ha='center',
            va='center',
            fontsize=5.8,
            color=color,
        )


def _add_embedding_heatmap_row_title(*, fig: Any, axes: Sequence[Any], title: str) -> None:
    positions = [ax.get_position() for ax in axes]
    left = min(position.x0 for position in positions)
    right = max(position.x1 for position in positions)
    top = max(position.y1 for position in positions)
    fig.text(
        (left + right) / 2,
        top + 0.045,
        title,
        ha='center',
        va='bottom',
        fontsize=9.5,
        fontweight='bold',
    )


def _metric_family_budget_rows_by_embedding(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    metric_order = {spec.metric_label: index for index, spec in enumerate(REPORT_METRIC_SPECS)}
    budget_order = {category: index for index, category in enumerate(BUDGET_CATEGORIES)}
    for spec in REPORT_METRIC_SPECS:
        grouped: dict[tuple[str, str, BudgetCategory, str], list[Mapping[str, object]]] = {}
        for row in rows:
            embedding_model = str(row.get('EmbeddingModel') or '')
            family = str(row.get('ExperimentFamilyLabel') or 'Unknown')
            category_value = str(row.get('BudgetCategory') or '')
            if not embedding_model or category_value not in BUDGET_CATEGORIES:
                continue
            category = cast(BudgetCategory, category_value)
            budget_label = str(row.get('BudgetCategoryLabel') or BUDGET_CATEGORY_LABELS[category])
            grouped.setdefault((embedding_model, family, category, budget_label), []).append(row)

        for (embedding_model, family, category, budget_label), group in grouped.items():
            out = _metric_family_budget_row_by_embedding(
                metric=spec.metric_label,
                metric_title=spec.title_label,
                rows=group,
                embedding_model=embedding_model,
                family=family,
                budget_category=category,
                budget_label=budget_label,
            )
            out['_MetricSort'] = metric_order[spec.metric_label]
            out['_BudgetSort'] = budget_order[category]
            summary_rows.append(out)

    summary_rows.sort(
        key=lambda row: (
            str(row.get('EmbeddingModel') or ''),
            cast(int, row['_MetricSort']),
            cast(int, row['_BudgetSort']),
            str(row.get('ExperimentFamilyLabel') or ''),
        )
    )
    for row in summary_rows:
        row.pop('_MetricSort', None)
        row.pop('_BudgetSort', None)
    return summary_rows


def _metric_family_budget_row_by_embedding(
    *,
    metric: DeltaMetricLabel,
    metric_title: str,
    rows: Sequence[Mapping[str, object]],
    embedding_model: str,
    family: str,
    budget_category: BudgetCategory,
    budget_label: str,
) -> dict[str, object]:
    delta_fm_col = f'Delta_FacLoc_MMR_{metric}'
    delta_ft_col = f'Delta_FacLoc_TopK_{metric}'
    delta_mt_col = f'Delta_MMR_TopK_{metric}'
    topk_col = f'TopK_{metric}'
    mmr_col = f'MMR_{metric}'
    facloc_col = f'FacLoc_{metric}'
    complete_rows = [row for row in rows if float_or_none(row.get(delta_fm_col)) is not None]
    deltas_fm = _numeric_values(complete_rows, delta_fm_col)
    deltas_ft = _numeric_values(complete_rows, delta_ft_col)
    deltas_mt = _numeric_values(complete_rows, delta_mt_col)
    topk_values = _numeric_values(complete_rows, topk_col)
    mmr_values = _numeric_values(complete_rows, mmr_col)
    facloc_values = _numeric_values(complete_rows, facloc_col)
    threshold = practical_effect_threshold(metric)
    row_count = len(complete_rows)
    facloc_better = sum(delta > threshold for delta in deltas_fm)
    facloc_tied = sum(abs(delta) <= threshold for delta in deltas_fm)
    facloc_worse = sum(delta < -threshold for delta in deltas_fm)
    facloc_topk_better = sum(delta > 0.0 for delta in deltas_ft)
    mmr_topk_better = sum(delta > 0.0 for delta in deltas_mt)
    return {
        'EmbeddingModel': embedding_model,
        'Metric': metric_title,
        'MetricLabel': metric,
        'ExperimentFamilyLabel': family,
        'BudgetCategory': budget_category,
        'BudgetCategoryLabel': budget_label,
        'Rows': row_count,
        'FacLocBetterRows': facloc_better,
        'FacLocTiedRows': facloc_tied,
        'FacLocWorseRows': facloc_worse,
        'FacLocTopKBetterRows': facloc_topk_better,
        'MMRTopKBetterRows': mmr_topk_better,
        'FacLocBetterPct': _fraction_or_none(facloc_better, row_count),
        'FacLocTiedPct': _fraction_or_none(facloc_tied, row_count),
        'FacLocWorsePct': _fraction_or_none(facloc_worse, row_count),
        'FacLocTopKBetterPct': _fraction_or_none(facloc_topk_better, row_count),
        'MMRTopKBetterPct': _fraction_or_none(mmr_topk_better, row_count),
        'MeanTopK': statistics.fmean(topk_values) if topk_values else None,
        'MeanMMR': statistics.fmean(mmr_values) if mmr_values else None,
        'MeanFacLoc': statistics.fmean(facloc_values) if facloc_values else None,
        'MeanDeltaFacLocMMR': statistics.fmean(deltas_fm) if deltas_fm else None,
        'MeanDeltaFacLocTopK': statistics.fmean(deltas_ft) if deltas_ft else None,
        'MeanDeltaMMRTopK': statistics.fmean(deltas_mt) if deltas_mt else None,
    }


def _numeric_values(rows: Sequence[Mapping[str, object]], column: str) -> list[float]:
    return [
        value for value in (float_or_none(row.get(column)) for row in rows) if value is not None
    ]


def _float_sort_key(value: object) -> float:
    return float_value if (float_value := float_or_none(value)) is not None else -math.inf


def _fraction_or_none(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _ordered_embedding_models(rows: Sequence[Mapping[str, object]]) -> list[str]:
    return ordered_embedding_models_for_rows(rows)


def _embedding_heatmap_fig_height(*, model_count: int, matrix_row_count: int) -> float:
    # Scale by both dimensions so faceted heatmaps keep readable cell height.
    row_block_height = 0.52 * max(matrix_row_count, 1) + 0.85
    return max(5.2, model_count * row_block_height + 0.9)


def _metric_plot_order(metric_label: str) -> int:
    order = {label: index for index, label in enumerate(REPORT_METRIC_LABELS)}
    return order.get(metric_label, len(order))


def _budget_delta_grouped_rows(
    rows: Sequence[Mapping[str, object]],
    value_column: str,
) -> list[Mapping[str, object]]:
    family_groups: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        family_groups.setdefault(_family_group_key(row), []).append(row)

    ordered_rows: list[Mapping[str, object]] = []
    for _family_mean, family_key in sorted(
        (
            (_mean_row_value(family_rows, value_column), family_key)
            for family_key, family_rows in family_groups.items()
        ),
        key=lambda item: (item[0], item[1]),
    ):
        parent_groups: dict[str, list[Mapping[str, object]]] = {}
        for row in family_groups[family_key]:
            parent_groups.setdefault(_parent_experiment_key(row), []).append(row)

        for _parent_mean, parent_key in sorted(
            (
                (_mean_row_value(parent_rows, value_column), parent_key)
                for parent_key, parent_rows in parent_groups.items()
            ),
            key=lambda item: (item[0], item[1]),
        ):
            ordered_rows.extend(
                sorted(
                    parent_groups[parent_key],
                    key=lambda row: (
                        float_or_none(row.get(value_column)) or float('-inf'),
                        str(row.get('ShortExperiment') or row.get('Experiment') or ''),
                    ),
                )
            )
    return ordered_rows


def _family_group_key(row: Mapping[str, object]) -> str:
    return str(row.get('ExperimentFamily') or row.get('ExperimentFamilyLabel') or 'unknown')


def _parent_experiment_key(row: Mapping[str, object]) -> str:
    experiment = str(row.get('Experiment') or row.get('ShortExperiment') or '')
    if '/' not in experiment:
        return experiment
    return experiment.split('/', 1)[0]


def _mean_row_value(rows: Sequence[Mapping[str, object]], value_column: str) -> float:
    values = [
        value
        for value in (float_or_none(row.get(value_column)) for row in rows)
        if value is not None
    ]
    return statistics.fmean(values) if values else float('-inf')


def _parent_group_spans(parent_keys: Sequence[str]) -> list[tuple[int, int]]:
    if not parent_keys:
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    current_parent = parent_keys[0]
    for index, parent_key in enumerate(parent_keys[1:], start=1):
        if parent_key == current_parent:
            continue
        spans.append((start, index - 1))
        start = index
        current_parent = parent_key
    spans.append((start, len(parent_keys) - 1))
    return spans


def _add_budget_delta_family_legend(*, fig: Any, rows: Sequence[Mapping[str, object]]) -> None:
    from matplotlib.patches import Patch

    present = {
        cast(ExperimentFamilyId, row.get('ExperimentFamily'))
        for row in rows
        if isinstance(row.get('ExperimentFamily'), str)
        and row.get('ExperimentFamily') in EXPERIMENT_FAMILIES
    }
    if not present:
        return
    ordered: list[ExperimentFamilyId] = [
        family_id for family_id in EXPERIMENT_FAMILIES if family_id in present
    ]
    handles = [
        Patch(
            facecolor=EXPERIMENT_FAMILY_COLORS[family_id],
            label=EXPERIMENT_FAMILY_LABELS[family_id],
        )
        for family_id in ordered
    ]
    fig.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.965),
        ncol=min(4, len(handles)),
        frameon=False,
        fontsize=11,
    )


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
        value = float_or_none(row.get('MeanDeltaFacLocMMR'))
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
            by_key[(row_key, column_key)] = float_or_none(row.get(value_field))
    return [
        [by_key.get((row_key, column_key)) for column_key in column_keys] for row_key in row_keys
    ]


def _matrix_with_nan(matrix: Sequence[Sequence[float | None]]) -> list[list[float]]:
    return [[value if value is not None else math.nan for value in row] for row in matrix]


def _symmetric_delta_norm(value_field: str) -> Any:
    from matplotlib.colors import TwoSlopeNorm

    max_abs = _heatmap_abs_scale(value_field)
    return TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)


def _heatmap_text_color(value: float, value_field: str) -> str:
    max_abs = _heatmap_abs_scale(value_field)
    return '#FFFFFF' if abs(value) / max_abs >= 0.58 else '#202020'


def _heatmap_abs_scale(value_field: str) -> float:
    return DELTA_HEATMAP_ABS_SCALE_BY_VALUE_FIELD.get(
        value_field,
        DELTA_HEATMAP_DEFAULT_ABS_SCALE,
    )


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
    parent_keys: Sequence[str],
) -> None:
    positions = list(range(len(labels)))
    _draw_parent_group_guides(ax=ax, parent_keys=parent_keys)
    ax.barh(positions, values, color=colors, zorder=2)
    ax.axvline(0.0, color='#303030', linewidth=0.9, zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.25, zorder=1)
    annotate_horizontal_values(ax=ax, values=values)


def _draw_parent_group_guides(*, ax: Any, parent_keys: Sequence[str]) -> None:
    for span_index, (start, end) in enumerate(_parent_group_spans(parent_keys)):
        # band_color = '#EAF3FB' if span_index % 2 == 0 else '#F3F4F6'
        band_color = '#F3F4F6' if span_index % 2 == 0 else '#EAF3FB'

        ax.axhspan(
            start - 0.5,
            end + 0.5,
            facecolor=band_color,
            alpha=0.75,
            linewidth=0,
            zorder=0,
        )
        if start > 0:
            ax.axhline(start - 0.5, color='#D3DEE8', linewidth=0.7, zorder=1)
