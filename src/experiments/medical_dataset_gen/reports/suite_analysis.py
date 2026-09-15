"""Strict, manifest-driven summaries for the frozen v5 suites."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.medical_dataset_gen.query_geometry.geom_plots_configs import (
    CANDIDATE_POOL_FACET_COLORS,
)
from experiments.medical_dataset_gen.reports.plot_rendering import (
    set_axis_title,
    set_figure_title,
    title_aware_layout_top,
)
from experiments.medical_dataset_gen.suites.core import (
    ComparisonGroup,
    SuiteManifest,
    SuiteManifestCell,
)

_PRIMARY_METRICS = (
    'Delta_FacLoc_MMR_FCP',
    'Delta_FacLoc_MMR_FacetCoverage',
    'Delta_FacLoc_MMR_AllFacetCoverageRate',
    'Delta_FacLoc_MMR_AllFacetCleanRate',
    'Delta_FacLoc_MMR_Precision',
    'Delta_FacLoc_MMR_alpha_nDCG',
)

_PRIMARY_METRIC_LABELS: dict[str, str] = {
    'Delta_FacLoc_MMR_FCP': 'FCP',
    'Delta_FacLoc_MMR_FacetCoverage': 'Facet coverage',
    'Delta_FacLoc_MMR_AllFacetCoverageRate': 'All-facet coverage',
    'Delta_FacLoc_MMR_AllFacetCleanRate': 'All-facet clean rate',
    'Delta_FacLoc_MMR_Precision': 'Precision',
    'Delta_FacLoc_MMR_alpha_nDCG': 'alpha-nDCG',
}

# Suite response surfaces are included at text width and retain a 3-column by
# 2-row layout. Larger explicit typography keeps the six panels legible after
# the PDF is scaled in the thesis. The combined interaction figure stacks two
# such blocks so their axes remain directly comparable.
SUITE_FIGURE_SIZE_IN = (10.0, 6.6)
COMBINED_INTERACTION_FIGURE_SIZE_IN = (10.0, 12.4)
SUITE_AXIS_TITLE_SIZE = 15
SUITE_AXIS_LABEL_SIZE = 13
SUITE_TICK_LABEL_SIZE = 12
SUITE_LEGEND_SIZE = 13
SUITE_FIGURE_TITLE_SIZE = 18
SUITE_LINE_WIDTH = 2.6
SUITE_MARKER_SIZE = 6

_SUITE_COMPARISON_LABELS: dict[str, str] = {
    'scale_balanced': 'Balanced reference',
    'scale_dominance': 'Dominant facet',
    'scale_sparse_one': 'One sparse facet',
    'scale_near_miss': 'Near-miss-heavy',
    'scale_compact_background': 'Compact background',
}

_SUITE_FACTOR_LEVEL_LABELS: dict[tuple[str, str], str] = {
    ('dominance_level', 'control'): 'Balanced support',
    ('dominance_level', 'high'): 'High dominance',
    ('sparse_level', 'control'): 'Balanced support',
    ('sparse_level', 'severe'): 'One severe sparse facet',
}

_COMBINED_INTERACTION_STEM = 'stressor_interactions_by_objective'
_COMBINED_INTERACTION_SPECS: tuple[tuple[str, str, str, str, Mapping[str, str]], ...] = (
    (
        'Dominance \N{MULTIPLICATION SIGN} background topology',
        'interaction_dominance_background',
        'background_topology',
        'dominance_level',
        {
            'Balanced support': CANDIDATE_POOL_FACET_COLORS[0],
            'High dominance': CANDIDATE_POOL_FACET_COLORS[1],
        },
    ),
    (
        'Sparse support \N{MULTIPLICATION SIGN} near-miss load',
        'interaction_sparse_near_miss',
        'near_miss_mass',
        'sparse_level',
        {
            'Balanced support': CANDIDATE_POOL_FACET_COLORS[3],
            'One severe sparse facet': CANDIDATE_POOL_FACET_COLORS[2],
        },
    ),
)

_TOPOLOGY_FIGURE_SPECS: tuple[tuple[str, str, str, str, str | None], ...] = (
    (
        'background_topology_by_objective',
        'Background topology response',
        'background_topology',
        'background_topology',
        None,
    ),
)

_SUITE_FIGURES_WITHOUT_X_LABEL = frozenset(
    {
        'scale_by_dataset_size',
        'background_topology_by_objective',
        _COMBINED_INTERACTION_STEM,
    }
)


def matched_contrast_rows(
    *,
    manifest: SuiteManifest,
    comparison_rows: Sequence[Mapping[str, object]],
    enforce_strict: bool = False,
    scope_cell_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    """Emit only complete declared contrasts, never inferred name-based ones."""
    rows_by_name: dict[str, dict[int, Mapping[str, object]]] = defaultdict(dict)
    for row in comparison_rows:
        name, k = str(row.get('Experiment') or ''), row.get('k')
        if name and isinstance(k, int):
            rows_by_name[name][k] = row
    output: list[dict[str, object]] = []
    for group in manifest.comparison_groups:
        group_cells = _group_cells(manifest, group)
        if not group_cells:
            continue
        # ``--where`` may intentionally request an endpoint-only smoke run.
        # It is not an incomplete declared contrast unless every member of the
        # declared group was in that report's requested scope.
        if scope_cell_ids is not None and any(
            cell.cell_id not in scope_cell_ids for cell in group_cells
        ):
            continue
        missing_cells = [cell.cell_id for cell in group_cells if cell.name not in rows_by_name]
        if missing_cells:
            # A filtered smoke report can legitimately include a shared
            # reference cell from otherwise unselected contrasts.  Once two
            # members are present, a missing sibling is an incomplete cross.
            present_count = len(group_cells) - len(missing_cells)
            if enforce_strict and present_count >= 2:
                raise ValueError(f'{group.comparison_id}: missing cells {missing_cells}')
            continue
        _validate_declared_matching(group, group_cells)
        common_k = set.intersection(*(set(rows_by_name[cell.name]) for cell in group_cells))
        expected_k = set.union(*(set(rows_by_name[cell.name]) for cell in group_cells))
        if common_k != expected_k:
            if enforce_strict:
                raise ValueError(f'{group.comparison_id}: incomplete budgets')
            continue
        for k in sorted(common_k):
            for cell in group_cells:
                row = rows_by_name[cell.name][k]
                out: dict[str, object] = {
                    'Comparison': group.comparison_id,
                    'AnalysisBlock': group.analysis_block,
                    'CellId': cell.cell_id,
                    'Distribution': cell.distribution_id,
                    'RunProfile': cell.run_profile_id,
                    'ArtifactOrigin': cell.origin,
                    'EmbeddingModel': row.get('EmbeddingModel'),
                    'k': k,
                    'TuningPolicy': str(row.get('LambdaPolicy') or 'cell_tuned'),
                }
                for factor in group.varying_factors:
                    value = _factor_value(_factor_for_cell(group, cell, factor))
                    out[f'Factor_{factor}'] = value
                    out[f'IsReference_{factor}'] = value == _factor_value(
                        group.reference_levels.get(factor)
                    )
                    out[f'FactorOrder_{factor}'] = json.dumps(
                        [_factor_value(level) for level in group.factor_levels.get(factor, [])]
                    )
                for metric in _PRIMARY_METRICS:
                    out[metric] = row.get(metric)
                output.append(out)
    return output


def write_suite_factor_figures(
    *,
    output_dir: Path,
    contrast_rows: Sequence[Mapping[str, object]],
) -> list[Path]:
    """Render compact, manifest-factor-driven scale and topology response plots."""
    from matplotlib import pyplot as plt

    figure_dir = output_dir / 'figures' / 'suite'
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    # A raw line for every profile x k x comparison produces over one hundred
    # traces.  These figures instead show equal-weight means over those
    # evaluation conditions, retaining the declared manipulation as the line.
    written.extend(
        _write_aggregated_factor_figure(
            plt=plt,
            output_dir=figure_dir,
            rows=contrast_rows,
            stem='scale_by_dataset_size',
            title='Scale response by candidate-pool size',
            comparison_ids=tuple(_SUITE_COMPARISON_LABELS),
            factor='scale',
            line_key='comparison',
        )
    )
    for stem, title, comparison_id, factor, line_factor in _TOPOLOGY_FIGURE_SPECS:
        written.extend(
            _write_aggregated_factor_figure(
                plt=plt,
                output_dir=figure_dir,
                rows=contrast_rows,
                stem=stem,
                title=title,
                comparison_ids=(comparison_id,),
                factor=factor,
                line_key=line_factor,
            )
        )
    written.extend(
        _write_combined_interaction_figure(
            plt=plt,
            output_dir=figure_dir,
            rows=contrast_rows,
        )
    )
    return written


def _write_aggregated_factor_figure(
    *,
    plt: Any,
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    stem: str,
    title: str,
    comparison_ids: Sequence[str],
    factor: str,
    line_key: str | None,
) -> list[Path]:
    """Plot matched-factor response means without profile/budget line clutter."""
    prepared = _aggregated_factor_values(
        rows=rows,
        comparison_ids=comparison_ids,
        factor=factor,
        line_key=line_key,
    )
    if prepared is None:
        return []
    levels, line_values = prepared

    figure, axes = plt.subplots(2, 3, figsize=SUITE_FIGURE_SIZE_IN)
    try:
        _draw_aggregated_factor_axes(
            axes=list(axes.flat),
            levels=levels,
            line_values=line_values,
            factor=factor,
            show_x_label=stem not in _SUITE_FIGURES_WITHOUT_X_LABEL,
        )

        handles, labels = axes.flat[0].get_legend_handles_labels()
        if len(handles) > 1:
            legend_columns = min(len(handles), 3)
            figure.legend(
                handles,
                labels,
                loc='lower center',
                ncol=legend_columns,
                frameon=False,
                fontsize=SUITE_LEGEND_SIZE,
            )
            bottom = 0.20 if len(handles) > 3 else 0.14
        else:
            bottom = 0.08
        set_figure_title(
            figure=figure,
            title=f'{title} (equal-model mean; band: model range)',
            fontsize=SUITE_FIGURE_TITLE_SIZE,
        )
        figure.subplots_adjust(
            left=0.09,
            right=0.99,
            top=title_aware_layout_top(titled_top=0.88, untitled_top=0.96),
            bottom=bottom,
            wspace=0.34,
            hspace=0.42,
        )
        written: list[Path] = []
        for suffix in ('png', 'pdf'):
            path = output_dir / f'{stem}.{suffix}'
            figure.savefig(path, dpi=180 if suffix == 'png' else None)
            written.append(path)
        return written
    finally:
        plt.close(figure)


def _write_combined_interaction_figure(
    *,
    plt: Any,
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
) -> list[Path]:
    """Stack both factorial interaction surfaces in one thesis figure."""
    blocks: list[
        tuple[
            str,
            str,
            Mapping[str, str],
            list[str],
            dict[str, dict[str, dict[str, dict[str, list[float]]]]],
        ]
    ] = []
    for title, comparison_id, factor, line_factor, line_colors in _COMBINED_INTERACTION_SPECS:
        prepared = _aggregated_factor_values(
            rows=rows,
            comparison_ids=(comparison_id,),
            factor=factor,
            line_key=line_factor,
        )
        if prepared is None:
            return []
        levels, line_values = prepared
        blocks.append((title, factor, line_colors, levels, line_values))

    figure = plt.figure(figsize=COMBINED_INTERACTION_FIGURE_SIZE_IN)
    try:
        subfigures = list(figure.subfigures(2, 1, hspace=0.04))
        for subfigure, (title, factor, line_colors, levels, line_values) in zip(
            subfigures, blocks, strict=True
        ):
            axes = subfigure.subplots(2, 3)
            _draw_aggregated_factor_axes(
                axes=list(axes.flat),
                levels=levels,
                line_values=line_values,
                factor=factor,
                show_x_label=False,
                line_colors=line_colors,
            )
            handles, labels = axes.flat[0].get_legend_handles_labels()
            subfigure.suptitle(title, fontsize=SUITE_FIGURE_TITLE_SIZE - 2, y=0.995)
            subfigure.legend(
                handles,
                labels,
                loc='lower center',
                ncol=2,
                frameon=False,
                fontsize=SUITE_LEGEND_SIZE,
            )
            subfigure.subplots_adjust(
                left=0.09,
                right=0.99,
                top=0.88,
                bottom=0.15,
                wspace=0.34,
                hspace=0.42,
            )

        written: list[Path] = []
        for suffix in ('png', 'pdf'):
            path = output_dir / f'{_COMBINED_INTERACTION_STEM}.{suffix}'
            figure.savefig(path, dpi=180 if suffix == 'png' else None)
            written.append(path)
        return written
    finally:
        plt.close(figure)


def _aggregated_factor_values(
    *,
    rows: Sequence[Mapping[str, object]],
    comparison_ids: Sequence[str],
    factor: str,
    line_key: str | None,
) -> tuple[list[str], dict[str, dict[str, dict[str, dict[str, list[float]]]]]] | None:
    """Collect factor levels and model-stratified values for one response surface."""
    factor_column = f'Factor_{factor}'
    comparison_set = set(comparison_ids)
    selected_rows = [
        row
        for row in rows
        if str(row.get('Comparison') or '') in comparison_set
        and row.get(factor_column) not in (None, '')
    ]
    if not selected_rows:
        return None

    levels = _ordered_factor_levels(selected_rows, factor=factor)
    if not levels:
        return None
    line_values: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    for row in selected_rows:
        line_label = _suite_line_label(row=row, line_key=line_key)
        level = str(row[factor_column])
        model = str(row.get('EmbeddingModel') or 'unknown')
        for metric in _PRIMARY_METRICS:
            value = _number(row.get(metric))
            if value is not None:
                line_values[line_label][metric][level][model].append(value)
    return levels, line_values


def _draw_aggregated_factor_axes(
    *,
    axes: Sequence[Any],
    levels: Sequence[str],
    line_values: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]]],
    factor: str,
    show_x_label: bool,
    line_colors: Mapping[str, str] | None = None,
) -> None:
    """Draw one six-metric response surface into caller-owned axes."""
    for panel_index, (metric, axis) in enumerate(zip(_PRIMARY_METRICS, axes, strict=True)):
        for line_label in sorted(line_values):
            model_means = [
                [
                    statistics.fmean(values)
                    for values in line_values[line_label][metric].get(level, {}).values()
                    if values
                ]
                for level in levels
            ]
            means = [statistics.fmean(values) if values else None for values in model_means]
            if any(value is not None for value in means):
                positions = list(range(len(levels)))
                line = axis.plot(
                    positions,
                    means,
                    marker='o',
                    linewidth=SUITE_LINE_WIDTH,
                    markersize=SUITE_MARKER_SIZE,
                    label=line_label,
                    color=line_colors.get(line_label) if line_colors is not None else None,
                )[0]
                if all(values for values in model_means):
                    axis.fill_between(
                        positions,
                        [min(values) for values in model_means],
                        [max(values) for values in model_means],
                        color=line.get_color(),
                        alpha=0.14,
                        linewidth=0,
                    )
        axis.axhline(0.0, color='#666666', linewidth=0.8, zorder=0)
        axis.axhline(0.05, color='#999999', linewidth=0.5, linestyle=':', zorder=0)
        axis.axhline(-0.05, color='#999999', linewidth=0.5, linestyle=':', zorder=0)
        set_axis_title(
            axis=axis,
            title=_PRIMARY_METRIC_LABELS[metric],
            fontsize=SUITE_AXIS_TITLE_SIZE,
        )
        if show_x_label:
            axis.set_xlabel(_factor_axis_label(factor), fontsize=SUITE_AXIS_LABEL_SIZE)
        # One y-label per row avoids collisions in the narrow thesis column
        # while retaining the shared unit for all three panels.
        axis.set_ylabel(
            'FacLoc - MMR' if panel_index % 3 == 0 else '',
            fontsize=SUITE_AXIS_LABEL_SIZE,
        )
        axis.set_xticks(range(len(levels)), [_display_factor_level(level) for level in levels])
        axis.tick_params(axis='both', labelsize=SUITE_TICK_LABEL_SIZE)
        axis.grid(axis='y', alpha=0.2)


def _ordered_factor_levels(rows: Sequence[Mapping[str, object]], *, factor: str) -> list[str]:
    """Read the author-declared factor order, with a deterministic fallback."""
    order_column = f'FactorOrder_{factor}'
    raw_order = next((row.get(order_column) for row in rows if row.get(order_column)), None)
    if raw_order is not None:
        try:
            order = [str(value) for value in json.loads(str(raw_order))]
        except json.JSONDecodeError:
            order = []
        present = {str(row.get(f'Factor_{factor}')) for row in rows}
        ordered = [level for level in order if level in present]
        if ordered:
            return ordered
    return sorted({str(row[f'Factor_{factor}']) for row in rows})


def _suite_line_label(*, row: Mapping[str, object], line_key: str | None) -> str:
    if line_key is None:
        return 'Overall mean'
    if line_key == 'comparison':
        comparison = str(row.get('Comparison') or '')
        return _SUITE_COMPARISON_LABELS.get(comparison, _display_factor_level(comparison))
    level = str(row.get(f'Factor_{line_key}') or '')
    return _SUITE_FACTOR_LEVEL_LABELS.get(
        (line_key, level),
        _display_factor_level(level),
    )


def _factor_axis_label(factor: str) -> str:
    return {
        'scale': 'Candidate-pool size',
        'background_topology': 'Background topology',
    }.get(factor, _display_factor_level(factor))


def _display_factor_level(value: str) -> str:
    return value.replace('_', ' ').replace('x', '\N{MULTIPLICATION SIGN}').title()


def _group_cells(manifest: SuiteManifest, group: ComparisonGroup) -> list[SuiteManifestCell]:
    wanted_distributions = set(group.distribution_ids)
    selected = [cell for cell in manifest.cells if cell.distribution_id in wanted_distributions]
    # A comparison is valid only on run profiles shared by every member.
    profiles_by_distribution: dict[str, set[str]] = defaultdict(set)
    for cell in selected:
        profiles_by_distribution[cell.distribution_id].add(cell.run_profile_id)
    shared = set.intersection(
        *(profiles_by_distribution[identifier] for identifier in wanted_distributions)
    )
    selected = [cell for cell in selected if cell.run_profile_id in shared]
    return sorted(
        selected,
        key=lambda cell: (cell.run_profile_id, group.distribution_ids.index(cell.distribution_id)),
    )


def _validate_declared_matching(group: ComparisonGroup, cells: Sequence[SuiteManifestCell]) -> None:
    for factor in group.matching_factors:
        values = {_factor_value(cell.factors.get(factor)) for cell in cells}
        if len(values) != 1:
            raise ValueError(
                f'{group.comparison_id}: matching factor {factor!r} is confounded: {values}'
            )


def _factor_for_cell(group: ComparisonGroup, cell: SuiteManifestCell, factor: str) -> object:
    return cell.factors.get(factor, group.reference_levels.get(factor))


def _factor_value(value: object) -> str:
    import json

    return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)


def _number(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
