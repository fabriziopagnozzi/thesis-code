"""Cross-wording-configuration aggregate figures."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from experiments.medical_dataset_gen.reports.analysis_constants import (
    EXPERIMENT_FAMILIES,
    EXPERIMENT_FAMILY_LABELS,
    DeltaMetricLabel,
    practical_effect_threshold,
)
from experiments.medical_dataset_gen.reports.helpers import float_or_none, short_model_label
from experiments.medical_dataset_gen.reports.models import BudgetCategory, PlotFormat
from experiments.medical_dataset_gen.reports.report_config import (
    AGGREGATE_PLOT_EXCLUDED_FAMILY_LABELS,
    BUDGET_CATEGORIES,
    BUDGET_CATEGORY_LABELS,
    DELTA_HEATMAP_ABS_SCALE_BY_VALUE_FIELD,
    DELTA_HEATMAP_DEFAULT_ABS_SCALE,
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


@dataclass(frozen=True)
class HeatmapPanelRow:
    label: str
    rows: Sequence[Mapping[str, object]]
    axis: SummaryAxisSpec


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
        row_tick_labels=tuple(
            _compact_config_label_for_key(summary_rows, config) for config in configs
        ),
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


def plot_config_fcp_budget_delta_heatmaps_by_distribution(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    """FCP wording-configuration budget view faceted by experiment family."""
    summary_rows = _metric_config_family_budget_rows(rows, metric_filter='FCP')
    if not _has_multiple_configs(summary_rows):
        return []
    configs = _ordered_configs(summary_rows)
    budget_labels = _budget_labels()
    panel_rows: list[HeatmapPanelRow] = []
    for family in _ordered_distribution_family_labels(summary_rows):
        family_rows = [row for row in summary_rows if row.get('ExperimentFamilyLabel') == family]
        if not _has_multiple_configs(family_rows):
            continue
        panel_rows.append(
            HeatmapPanelRow(
                label=family,
                rows=family_rows,
                axis=SummaryAxisSpec(
                    row_keys=tuple(configs),
                    column_keys=tuple(budget_labels),
                    row_tick_labels=tuple(
                        _compact_config_label_for_key(summary_rows, config) for config in configs
                    ),
                    column_tick_labels=tuple(budget_labels),
                    row_field='WordingConfig',
                    column_field='BudgetCategoryLabel',
                ),
            )
        )
    if not panel_rows:
        return []
    return _plot_delta_heatmap_panel_grid(
        plt=plt,
        panel_rows=panel_rows,
        title='FCP deltas by wording configuration and retrieval budget within each family',
        footnote=(
            'Each row filters to one experiment family. Cell format: top line = mean FCP delta; '
            'bottom line = win rate for the comparison.'
        ),
        output_path=output_dir
        / f'cross_config_fcp_budget_delta_heatmaps_by_distribution.{plot_format}',
        figsize=(
            16.5,
            _panel_grid_fig_height(panel_rows, min_cell_height=0.48, min_height=13.0),
        ),
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
        column_tick_labels=tuple(
            _config_label_for_key(summary_rows, config) for config in configs
        ),
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


def plot_config_metric_delta_heatmap_low_budget_by_distribution(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    """Low-budget metric wording-configuration view faceted by experiment family."""
    summary_rows = [
        row
        for row in _metric_config_family_budget_rows(rows)
        if row.get('BudgetCategory') == 'low_budget'
        and str(row.get('MetricLabel') or '') in REPORT_METRIC_LABEL_SET
    ]
    if not _has_multiple_configs(summary_rows):
        return []
    metric_labels = _ordered_metric_labels(summary_rows)
    configs = _ordered_configs(summary_rows)
    panel_rows: list[HeatmapPanelRow] = []
    for family in _ordered_distribution_family_labels(summary_rows):
        family_rows = [row for row in summary_rows if row.get('ExperimentFamilyLabel') == family]
        if not _has_multiple_configs(family_rows):
            continue
        panel_rows.append(
            HeatmapPanelRow(
                label=family,
                rows=family_rows,
                axis=SummaryAxisSpec(
                    row_keys=tuple(metric_labels),
                    column_keys=tuple(configs),
                    row_tick_labels=tuple(
                        _metric_title_from_rows(summary_rows, metric) for metric in metric_labels
                    ),
                    column_tick_labels=tuple(
                        _config_label_for_key(summary_rows, config) for config in configs
                    ),
                    row_field='MetricLabel',
                    column_field='WordingConfig',
                ),
            )
        )
    if not panel_rows:
        return []
    return _plot_delta_heatmap_panel_grid(
        plt=plt,
        panel_rows=panel_rows,
        title='Low-budget metric deltas by wording configuration within each family',
        footnote=(
            'Each row filters to one experiment family. Cell format: top line = mean delta; '
            'bottom line = win rate for the comparison.'
        ),
        output_path=output_dir
        / f'cross_config_metric_delta_heatmap_low_budget_by_distribution.{plot_format}',
        figsize=(
            16.5,
            _panel_grid_fig_height(panel_rows, min_cell_height=0.40, min_height=12.0),
        ),
    )


def plot_config_metric_delta_heatmap_low_budget_by_distribution_embedding_model(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    """Low-budget metric wording-configuration view faceted by family and embedding model."""
    summary_rows = [
        row
        for row in _metric_config_family_budget_rows_by_embedding(rows)
        if row.get('BudgetCategory') == 'low_budget'
        and str(row.get('MetricLabel') or '') in REPORT_METRIC_LABEL_SET
    ]
    if not _has_multiple_configs(summary_rows):
        return []
    metric_labels = _ordered_metric_labels(summary_rows)
    configs = _ordered_configs(summary_rows)
    paths: list[Path] = []
    for family in _ordered_distribution_family_labels(summary_rows):
        family_rows = [row for row in summary_rows if row.get('ExperimentFamilyLabel') == family]
        if not _has_multiple_configs(family_rows):
            continue
        panel_rows: list[HeatmapPanelRow] = []
        for model in _ordered_embedding_models(family_rows):
            model_rows = [row for row in family_rows if row.get('EmbeddingModel') == model]
            if not _has_multiple_configs(model_rows):
                continue
            panel_rows.append(
                HeatmapPanelRow(
                    label=short_model_label(model),
                    rows=model_rows,
                    axis=SummaryAxisSpec(
                        row_keys=tuple(metric_labels),
                        column_keys=tuple(configs),
                        row_tick_labels=tuple(
                            _metric_title_from_rows(summary_rows, metric)
                            for metric in metric_labels
                        ),
                        column_tick_labels=tuple(
                            _config_label_for_key(summary_rows, config)
                            for config in configs
                        ),
                        row_field='MetricLabel',
                        column_field='WordingConfig',
                    ),
                )
            )
        if not panel_rows:
            continue
        paths.extend(
            _plot_delta_heatmap_panel_grid(
                plt=plt,
                panel_rows=panel_rows,
                title=(
                    'Low-budget metric deltas by wording configuration in '
                    f'{family}, by embedding model'
                ),
                footnote=(
                    'Each row filters to one embedding model within the experiment family. '
                    'Cell format: top line = mean delta; bottom line = win rate for the '
                    'comparison.'
                ),
                output_path=output_dir
                / (
                    'cross_config_metric_delta_heatmap_low_budget_'
                    f'{_label_filename_token(family)}_by_emb_model.{plot_format}'
                ),
                figsize=(
                    16.5,
                    _panel_grid_fig_height(panel_rows, min_cell_height=0.42, min_height=8.0),
                ),
            )
        )
    return paths


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
    panel_rows = _configuration_family_budget_panel_rows(summary_rows)
    return _plot_delta_heatmap_panel_grid(
        plt=plt,
        panel_rows=panel_rows,
        title='FCP deltas by wording configuration, experiment family, and retrieval budget',
        footnote=(
            'Each row is one wording configuration. '
            'Top line = mean FCP delta; bottom line = win rate for the comparison.'
        ),
        output_path=output_dir / f'cross_config_fcp_family_budget_delta_heatmaps.{plot_format}',
        figsize=(
            16.5,
            _panel_grid_fig_height(panel_rows, min_cell_height=0.46, min_height=12.0),
        ),
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
        panel_rows = _configuration_family_budget_panel_rows(summary_rows)
        paths.extend(
            _plot_delta_heatmap_panel_grid(
                plt=plt,
                panel_rows=panel_rows,
                title=(
                    'FCP deltas by wording configuration, experiment family, '
                    f'and retrieval budget for {short_model_label(model)}'
                ),
                footnote=(
                    'Each row is one wording configuration. '
                    'Top line = mean FCP delta; bottom line = win rate for the comparison.'
                ),
                output_path=(
                    output_dir
                    / (
                        'cross_config_fcp_family_budget_delta_heatmaps_by_emb_model_'
                        f'{_filename_token(model)}.{plot_format}'
                    )
                ),
                figsize=(
                    16.5,
                    _panel_grid_fig_height(panel_rows, min_cell_height=0.46, min_height=12.0),
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
    panel_rows = _configuration_metric_family_panel_rows(summary_rows)
    return _plot_delta_heatmap_panel_grid(
        plt=plt,
        panel_rows=panel_rows,
        title='Low-budget metric deltas by wording configuration and experiment family',
        footnote=(
            'Each row is one wording configuration. '
            'Top line = mean delta; bottom line = win rate for the comparison.'
        ),
        output_path=output_dir
        / f'cross_config_metric_family_delta_heatmap_low_budget.{plot_format}',
        figsize=(
            16.5,
            _panel_grid_fig_height(panel_rows, min_cell_height=0.44, min_height=13.0),
        ),
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
        panel_rows = _configuration_metric_family_panel_rows(summary_rows)
        paths.extend(
            _plot_delta_heatmap_panel_grid(
                plt=plt,
                panel_rows=panel_rows,
                title=(
                    'Low-budget metric deltas by wording configuration and '
                    f'experiment family for {short_model_label(model)}'
                ),
                footnote=(
                    'Each row is one wording configuration. '
                    'Top line = mean delta; bottom line = win rate for the comparison.'
                ),
                output_path=(
                    output_dir
                    / (
                        'cross_config_metric_family_delta_heatmap_low_budget_by_emb_model_'
                        f'{_filename_token(model)}.{plot_format}'
                    )
                ),
                figsize=(
                    16.5,
                    _panel_grid_fig_height(panel_rows, min_cell_height=0.44, min_height=13.0),
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
            image = _draw_delta_heatmap_axis(
                fig=fig,
                ax=ax,
                rows=rows,
                axis=axis,
                spec=spec,
                matrix=matrix,
                show_title=True,
                show_x_tick_labels=True,
                show_y_tick_labels=True,
            )
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, extend='both')
            cbar.set_label(spec.colorbar_label)
        fig.suptitle(title, y=0.985)
        fig.text(0.5, 0.018, footnote, ha='center', va='bottom', fontsize=8, color='#303030')
        fig.tight_layout(rect=(0, 0.055, 1, 0.955))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')
        return [output_path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def _plot_delta_heatmap_panel_grid(
    *,
    plt: object,
    panel_rows: Sequence[HeatmapPanelRow],
    title: str,
    footnote: str,
    output_path: Path,
    figsize: tuple[float, float],
) -> list[Path]:
    specs = _delta_panel_specs()
    active_panel_rows: list[HeatmapPanelRow] = []
    panel_matrices: list[tuple[list[list[list[float | None]]], list[float]]] = []
    all_values: list[float] = []
    for panel_row in panel_rows:
        matrices, values = _delta_matrices(rows=panel_row.rows, axis=panel_row.axis, specs=specs)
        if not values:
            continue
        active_panel_rows.append(panel_row)
        panel_matrices.append((matrices, values))
        all_values.extend(values)
    if not all_values:
        return []

    nrows = len(panel_matrices)
    fig, axes_obj = plt.subplots(
        nrows=nrows,
        ncols=3,
        figsize=figsize,
        sharex='col',
        squeeze=False,
        gridspec_kw={'wspace': 0.38},
    )  # type: ignore[attr-defined]
    axes = cast(Sequence[Sequence[Any]], axes_obj)
    try:
        for row_index, (panel_row, (matrices, _values)) in enumerate(
            zip(active_panel_rows, panel_matrices, strict=True)
        ):
            for column_index, (spec, matrix) in enumerate(zip(specs, matrices, strict=True)):
                image = _draw_delta_heatmap_axis(
                    fig=fig,
                    ax=axes[row_index][column_index],
                    rows=panel_row.rows,
                    axis=panel_row.axis,
                    spec=spec,
                    matrix=matrix,
                    show_title=False,
                    show_x_tick_labels=row_index == nrows - 1,
                    show_y_tick_labels=column_index == 0,
                )
                cbar = fig.colorbar(
                    image, ax=axes[row_index][column_index], fraction=0.046, pad=0.04, extend='both'
                )
                cbar.set_label(spec.colorbar_label, fontsize=7)
                cbar.ax.tick_params(labelsize=7)
        fig.subplots_adjust(
            left=0.10,
            right=0.985,
            bottom=_panel_grid_bottom_margin(active_panel_rows),
            top=0.88,
            hspace=_panel_grid_hspace(nrows),
        )
        _add_panel_column_titles(fig=fig, axes=axes[0], specs=specs)
        for row_index, panel_row in enumerate(active_panel_rows):
            _add_panel_row_title(fig=fig, axes=axes[row_index], title=panel_row.label)
        fig.suptitle(title, y=0.995)
        fig.text(0.5, 0.016, footnote, ha='center', va='bottom', fontsize=8, color='#303030')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')
        return [output_path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def _panel_grid_fig_height(
    panel_rows: Sequence[HeatmapPanelRow],
    *,
    min_cell_height: float,
    min_height: float,
) -> float:
    title_gap_height = 0.50
    body_height = sum(
        max(1, len(panel_row.axis.row_keys)) * min_cell_height + title_gap_height
        for panel_row in panel_rows
    )
    return max(min_height, body_height + 1.4)


def _panel_grid_hspace(nrows: int) -> float:
    if nrows <= 1:
        return 0.0
    if nrows == 2:
        return 0.20
    if nrows <= 4:
        return 0.40
    if nrows <= 6:
        return 0.54
    return 0.65


def _panel_grid_bottom_margin(panel_rows: Sequence[HeatmapPanelRow]) -> float:
    if panel_rows and any(' / ' in label for label in panel_rows[-1].axis.column_tick_labels):
        return 0.20
    return 0.12


def _add_panel_row_title(*, fig: Any, axes: Sequence[Any], title: str) -> None:
    positions = [ax.get_position() for ax in axes]
    left = min(position.x0 for position in positions)
    right = max(position.x1 for position in positions)
    top = max(position.y1 for position in positions)
    fig.text(
        (left + right) / 2,
        top + 0.018,
        title,
        ha='center',
        va='bottom',
        fontsize=9.5,
        fontweight='bold',
    )


def _add_panel_column_titles(
    *,
    fig: Any,
    axes: Sequence[Any],
    specs: Sequence[DeltaPanelSpec],
) -> None:
    top = max(ax.get_position().y1 for ax in axes)
    for ax, spec in zip(axes, specs, strict=True):
        position = ax.get_position()
        fig.text(
            (position.x0 + position.x1) / 2,
            top + 0.055,
            spec.title,
            ha='center',
            va='bottom',
            fontsize=10,
        )


def _draw_delta_heatmap_axis(
    *,
    fig: Any,
    ax: Any,
    rows: Sequence[Mapping[str, object]],
    axis: SummaryAxisSpec,
    spec: DeltaPanelSpec,
    matrix: Sequence[Sequence[float | None]],
    show_title: bool,
    show_x_tick_labels: bool,
    show_y_tick_labels: bool,
) -> Any:
    _ = fig
    image = ax.imshow(
        _matrix_with_nan(matrix),
        cmap=spec.cmap,
        norm=_symmetric_delta_norm(spec.value_field),
        aspect='auto',
    )
    if show_title:
        ax.set_title(spec.title)
    ax.set_xticks(range(len(axis.column_tick_labels)))
    ax.set_xticklabels(
        axis.column_tick_labels if show_x_tick_labels else [''] * len(axis.column_tick_labels),
        rotation=_tick_label_rotation(axis.column_tick_labels),
        ha=_tick_label_alignment(axis.column_tick_labels),
    )
    ax.set_yticks(range(len(axis.row_tick_labels)))
    ax.set_yticklabels(
        axis.row_tick_labels if show_y_tick_labels else [''] * len(axis.row_tick_labels),
        fontsize=7,
    )
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
            pct = float_or_none(source_row.get(spec.pct_field)) if source_row is not None else None
            ax.text(
                x_index,
                y_index,
                f'{value:+.3f}\n{pct:.0%}' if pct is not None else f'{value:+.3f}',
                ha='center',
                va='center',
                fontsize=6.5,
                color=_heatmap_text_color(value, spec.value_field),
            )
    return image


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


def _metric_config_family_budget_rows_by_embedding(
    rows: Sequence[Mapping[str, object]],
    *,
    metric_filter: DeltaMetricLabel | None = None,
) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for spec in REPORT_METRIC_SPECS:
        if metric_filter is not None and spec.metric_label != metric_filter:
            continue
        grouped: dict[tuple[str, str, str, BudgetCategory, str], list[Mapping[str, object]]] = {}
        for row in rows:
            embedding_model = str(row.get('EmbeddingModel') or '')
            config = str(row.get('WordingConfig') or '')
            family = str(row.get('ExperimentFamilyLabel') or 'Unknown')
            category_value = str(row.get('BudgetCategory') or '')
            if (
                not embedding_model
                or not config
                or category_value not in BUDGET_CATEGORIES
                or family in AGGREGATE_PLOT_EXCLUDED_FAMILY_LABELS
            ):
                continue
            category = cast(BudgetCategory, category_value)
            budget_label = str(row.get('BudgetCategoryLabel') or BUDGET_CATEGORY_LABELS[category])
            grouped.setdefault(
                (embedding_model, config, family, category, budget_label), []
            ).append(row)
        for (embedding_model, config, family, category, budget_label), group in grouped.items():
            first = group[0]
            summary_rows.append(
                _metric_summary_row(
                    metric=spec.metric_label,
                    metric_title=spec.title_label,
                    rows=group,
                    extra={
                        'EmbeddingModel': embedding_model,
                        'WordingConfig': config,
                        'WordingConfigLabel': first.get('WordingConfigLabel'),
                        'QueryMode': first.get('QueryMode'),
                        'FocusMode': first.get('FocusMode'),
                        'ChunkTextMode': first.get('ChunkTextMode'),
                        'ExperimentFamilyLabel': family,
                        'BudgetCategory': category,
                        'BudgetCategoryLabel': budget_label,
                    },
                )
            )
    summary_rows.sort(
        key=lambda row: (
            str(row.get('EmbeddingModel') or ''),
            _config_family_budget_sort_key(row),
        )
    )
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


def _configuration_family_budget_panel_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[HeatmapPanelRow]:
    configs = _ordered_configs(rows)
    families = _ordered_distribution_family_labels(rows)
    budget_labels = tuple(_budget_labels())
    return [
        HeatmapPanelRow(
            label=_config_label_for_key(rows, config),
            rows=[row for row in rows if row.get('WordingConfig') == config],
            axis=SummaryAxisSpec(
                row_keys=tuple(families),
                column_keys=budget_labels,
                row_tick_labels=tuple(families),
                column_tick_labels=budget_labels,
                row_field='ExperimentFamilyLabel',
                column_field='BudgetCategoryLabel',
            ),
        )
        for config in configs
    ]


def _configuration_metric_family_panel_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[HeatmapPanelRow]:
    configs = _ordered_configs(rows)
    metrics = _ordered_metric_labels(rows)
    families = _ordered_distribution_family_labels(rows)
    return [
        HeatmapPanelRow(
            label=_config_label_for_key(rows, config),
            rows=[row for row in rows if row.get('WordingConfig') == config],
            axis=SummaryAxisSpec(
                row_keys=tuple(metrics),
                column_keys=tuple(families),
                row_tick_labels=tuple(
                    _metric_title_from_rows(rows, metric) for metric in metrics
                ),
                column_tick_labels=tuple(families),
                row_field='MetricLabel',
                column_field='ExperimentFamilyLabel',
            ),
        )
        for config in configs
    ]


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
        if (
            str(row.get(first_field) or '') == first_value
            and str(row.get(second_field) or '') == second_value
        ):
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
    return [model for model in EMBEDDING_MODEL_FACETED_PLOT_MODELS if model in available]


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


def _compact_config_label_for_key(rows: Sequence[Mapping[str, object]], config: str) -> str:
    return _config_label_for_key(rows, config).replace(' / ', '\n')


def _tick_label_rotation(labels: Sequence[str]) -> int:
    _ = labels
    return 35


def _tick_label_alignment(labels: Sequence[str]) -> str:
    _ = labels
    return 'right'


def _ordered_distribution_family_labels(rows: Sequence[Mapping[str, object]]) -> list[str]:
    present = {
        str(row.get('ExperimentFamilyLabel') or '')
        for row in rows
        if row.get('ExperimentFamilyLabel')
    }
    excluded_labels = {
        *AGGREGATE_PLOT_EXCLUDED_FAMILY_LABELS,
        EXPERIMENT_FAMILY_LABELS['budget_sweep'],
        EXPERIMENT_FAMILY_LABELS['embedding_comparison'],
        EXPERIMENT_FAMILY_LABELS['unknown'],
    }
    ordered = [
        EXPERIMENT_FAMILY_LABELS[family_id]
        for family_id in EXPERIMENT_FAMILIES
        if EXPERIMENT_FAMILY_LABELS[family_id] in present
        and EXPERIMENT_FAMILY_LABELS[family_id] not in excluded_labels
    ]
    return ordered or _ordered_families(rows)


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


def _filename_token(value: str) -> str:
    return short_model_label(value).lower().replace('/', '_').replace('-', '_').replace('.', '_')


def _label_filename_token(value: str) -> str:
    return '_'.join(value.lower().replace('/', '_').replace('-', '_').replace('.', '_').split())
