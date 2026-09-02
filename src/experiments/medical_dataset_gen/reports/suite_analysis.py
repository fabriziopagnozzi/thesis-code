"""Strict, manifest-driven summaries for native and migrated v5 suites."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

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

_INVALID_BACKGROUND_SHELLS = frozenset({'near', 'intermediate'})

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

_SUITE_COMPARISON_LABELS: dict[str, str] = {
    'scale_balanced': 'Balanced reference',
    'scale_dominance': 'Dominant facet',
    'scale_sparse_one': 'One sparse facet',
    'scale_near_miss': 'Near-miss-heavy',
    'scale_compact_background': 'Compact background',
}

_TOPOLOGY_FIGURE_SPECS: tuple[tuple[str, str, str, str, str | None], ...] = (
    (
        'background_topology_by_objective',
        'Background topology response',
        'background_topology',
        'background_topology',
        None,
    ),
    (
        'near_miss_topology_by_objective',
        'Near-miss topology response',
        'near_miss_topology',
        'near_miss_topology',
        None,
    ),
    (
        'dominance_background_interaction_by_objective',
        'Dominance x background-topology response',
        'interaction_dominance_background',
        'background_topology',
        'dominance_level',
    ),
    (
        'sparse_near_miss_interaction_by_objective',
        'Sparse support x near-miss-load response',
        'interaction_sparse_near_miss',
        'near_miss_mass',
        'sparse_level',
    ),
)

RESULTS_SUITE_FIGURE_STEMS: tuple[str, ...] = (
    'scale_by_dataset_size',
    'background_topology_by_objective',
    'dominance_background_interaction_by_objective',
    'sparse_near_miss_interaction_by_objective',
)


def report_eligible_manifest(manifest: SuiteManifest) -> tuple[SuiteManifest, set[str]]:
    """Exclude legacy background variants that violate the outlier definition.

    Background outliers must change condition, subgroup, and clinical axis.
    Earlier suite materializations additionally contain ``near`` and
    ``intermediate`` variants that do not meet that requirement.  Their stored
    artifacts remain available for audit, but they are not evidence for the
    benchmark and must not enter report summaries or contrasts.
    """
    excluded_distributions = {
        distribution.distribution_id
        for distribution in manifest.distributions
        if distribution.family_id == 'background_variant'
        and distribution.factors.get('background_shell') in _INVALID_BACKGROUND_SHELLS
    }
    if not excluded_distributions:
        return manifest, set()

    eligible_cells = [
        cell for cell in manifest.cells if cell.distribution_id not in excluded_distributions
    ]
    eligible_distributions = [
        distribution
        for distribution in manifest.distributions
        if distribution.distribution_id not in excluded_distributions
    ]
    comparison_groups = [
        _report_eligible_comparison_group(group, excluded_distributions)
        for group in manifest.comparison_groups
    ]
    reporting_manifest = manifest.model_copy(
        update={
            'cells': eligible_cells,
            'evaluations': eligible_cells,
            'distributions': eligible_distributions,
            'comparison_groups': comparison_groups,
        }
    )
    return reporting_manifest, excluded_distributions


def _report_eligible_comparison_group(
    group: ComparisonGroup,
    excluded_distributions: set[str],
) -> ComparisonGroup:
    if group.comparison_id != 'background_topology_shell':
        return group
    distribution_ids = [
        distribution_id
        for distribution_id in group.distribution_ids
        if distribution_id not in excluded_distributions
    ]
    return group.model_copy(
        update={
            'comparison_id': 'background_topology',
            'distribution_ids': distribution_ids,
            'varying_factors': ['background_topology'],
            'matching_factors': ['background_mass', 'near_miss_mass'],
            'factor_levels': {'background_topology': ['32x1', '16x2', '8x4', '4x8']},
            'reference_levels': {'background_topology': '32x1'},
            'owned_paths': {'background_topology': ['generation.chunk_pools.background_outliers']},
        }
    )


def suite_distribution_and_family_rows(
    comparison_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Aggregate profile → evaluation → distribution → block → family.

    ``IncludeInFamilySummary`` prevents the dense topology surface and the
    selected interactions from acquiring more influence merely because they
    have more variants.
    """
    causal = [row for row in comparison_rows if row.get('IncludeInCausalSummaries') is True]
    by_distribution: dict[tuple[str, str, str, str, str, int], list[Mapping[str, object]]] = (
        defaultdict(list)
    )
    for row in causal:
        k = row.get('k')
        if not isinstance(k, int):
            continue
        for block in _analysis_blocks(row):
            key = (
                str(row.get('ExperimentFamily') or 'unknown'),
                block,
                str(row.get('Distribution') or 'unknown'),
                str(row.get('ArtifactOrigin') or 'unknown'),
                str(row.get('EmbeddingModel') or 'unknown'),
                k,
            )
            by_distribution[key].append(row)
    distribution_rows: list[dict[str, object]] = []
    for (family, block, distribution, origin, model, k), rows in sorted(by_distribution.items()):
        out: dict[str, object] = {
            'ExperimentFamily': family,
            'ExperimentFamilyLabel': rows[0].get('ExperimentFamilyLabel'),
            'AnalysisBlock': block,
            'Distribution': distribution,
            'ArtifactOrigin': origin,
            'EmbeddingModel': model,
            'k': k,
            'Cells': len(rows),
            'IncludeInFamilySummary': all(
                row.get('IncludeInFamilySummary', True) is True for row in rows
            ),
        }
        _add_means(out, rows)
        distribution_rows.append(out)

    # Equal-weight distributions within a block.  The distribution report
    # remains complete (including scale and interactions); primary family
    # summaries are restricted below so those dense blocks cannot dominate.
    by_block: dict[tuple[str, str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in distribution_rows:
        by_block[
            (
                str(row['ExperimentFamily']),
                str(row['AnalysisBlock']),
                str(row['ArtifactOrigin']),
                str(row['EmbeddingModel']),
                int(cast(int, row['k'])),
            )
        ].append(row)
    block_rows: list[dict[str, object]] = []
    for (family, block, origin, model, k), rows in sorted(by_block.items()):
        out: dict[str, object] = {
            'ExperimentFamily': family,
            'ExperimentFamilyLabel': rows[0].get('ExperimentFamilyLabel'),
            'AnalysisBlock': block,
            'ArtifactOrigin': origin,
            'EmbeddingModel': model,
            'k': k,
            'Distributions': len(rows),
            'Aggregation': 'equal_distribution_weight',
            'IncludeInFamilySummary': all(
                row.get('IncludeInFamilySummary', False) is True for row in rows
            ),
        }
        _add_means(out, rows)
        block_rows.append(out)
    by_family: dict[tuple[str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in block_rows:
        if row.get('AnalysisBlock') == 'scale' or row.get('IncludeInFamilySummary') is not True:
            continue
        by_family[
            (
                str(row['ExperimentFamily']),
                str(row['ArtifactOrigin']),
                str(row['EmbeddingModel']),
                int(cast(int, row['k'])),
            )
        ].append(row)
    family_rows: list[dict[str, object]] = []
    for (family, origin, model, k), rows in sorted(by_family.items()):
        out = {
            'ExperimentFamily': family,
            'ExperimentFamilyLabel': rows[0].get('ExperimentFamilyLabel'),
            'ArtifactOrigin': origin,
            'EmbeddingModel': model,
            'k': k,
            'AnalysisBlocks': len(rows),
            'Aggregation': 'equal_analysis_block_weight',
        }
        _add_means(out, rows)
        family_rows.append(out)
    return distribution_rows, family_rows


def _analysis_blocks(row: Mapping[str, object]) -> tuple[str, ...]:
    raw = str(row.get('AnalysisBlocks') or '')
    blocks = tuple(block for block in raw.split('|') if block)
    return blocks or ('unblocked',)


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
        if row.get('IncludeInCausalSummaries') is not True:
            continue
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
            if enforce_strict and group.strict and present_count >= 2:
                raise ValueError(f'{group.comparison_id}: missing cells {missing_cells}')
            continue
        _validate_declared_matching(group, group_cells)
        common_k = set.intersection(*(set(rows_by_name[cell.name]) for cell in group_cells))
        expected_k = set.union(*(set(rows_by_name[cell.name]) for cell in group_cells))
        if common_k != expected_k:
            if enforce_strict and group.strict:
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
                for factor in group.all_varying_factors:
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


def analysis_series_rows(
    *,
    manifest: SuiteManifest,
    comparison_rows: Sequence[Mapping[str, object]],
    enforce_strict: bool = False,
    scope_cell_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    """Emit declared non-rectangular series at each point's explicit budget.

    This covers proportional scale diagonals, for which a normal contrast is
    unsuitable because the selected budget intentionally changes with the
    distribution and run profile.
    """
    rows_by_name_and_k: dict[tuple[str, int], Mapping[str, object]] = {}
    for row in comparison_rows:
        if row.get('IncludeInCausalSummaries') is not True:
            continue
        name, k = str(row.get('Experiment') or ''), row.get('k')
        if name and isinstance(k, int):
            rows_by_name_and_k[(name, k)] = row
    cells_by_id = {cell.cell_id: cell for cell in manifest.cells}
    output: list[dict[str, object]] = []
    for series in manifest.analysis_series:
        points = [
            (
                point,
                cells_by_id[f'{point.distribution_id}__{point.run_profile_id}'],
            )
            for point in series.points
        ]
        if scope_cell_ids is not None and any(
            cell.cell_id not in scope_cell_ids for _, cell in points
        ):
            continue
        missing = [
            cell.cell_id for point, cell in points if (cell.name, point.k) not in rows_by_name_and_k
        ]
        if missing:
            if enforce_strict and series.strict:
                raise ValueError(f'{series.series_id}: missing series points {missing}')
            continue
        factor_names = sorted({key for point, _ in points for key in point.factors})
        for point, cell in points:
            row = rows_by_name_and_k[(cell.name, point.k)]
            out: dict[str, object] = {
                'Series': series.series_id,
                'AnalysisBlock': series.analysis_block,
                'PointId': point.point_id,
                'IsReference': point.point_id == series.reference_point_id,
                'CellId': cell.cell_id,
                'Distribution': cell.distribution_id,
                'RunProfile': cell.run_profile_id,
                'ArtifactOrigin': cell.origin,
                'EmbeddingModel': row.get('EmbeddingModel'),
                'k': point.k,
                'TuningPolicy': str(row.get('LambdaPolicy') or 'cell_tuned'),
            }
            for factor in factor_names:
                out[f'Factor_{factor}'] = point.factors.get(factor)
            for metric in _PRIMARY_METRICS:
                out[metric] = row.get(metric)
            output.append(out)
    return output


def factor_interaction_rows(
    *, manifest: SuiteManifest, comparison_rows: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    """Response surfaces plus two-level difference-in-differences effects."""
    rows_by_name: dict[str, dict[int, Mapping[str, object]]] = defaultdict(dict)
    for row in comparison_rows:
        name, k = str(row.get('Experiment') or ''), row.get('k')
        if row.get('IncludeInCausalSummaries') is True and name and isinstance(k, int):
            rows_by_name[name][k] = row
    output: list[dict[str, object]] = []
    for group in manifest.comparison_groups:
        cells = _group_cells(manifest, group)
        factors = group.all_varying_factors
        if len(factors) != 2 or not cells:
            continue
        for profile in sorted({cell.run_profile_id for cell in cells}):
            profile_cells = [cell for cell in cells if cell.run_profile_id == profile]
            if any(cell.name not in rows_by_name for cell in profile_cells):
                continue
            common_k = (
                set.intersection(*(set(rows_by_name[cell.name]) for cell in profile_cells))
                if profile_cells
                else set()
            )
            for k in sorted(common_k):
                values: dict[tuple[str, str], Mapping[str, object]] = {}
                for cell in profile_cells:
                    key = tuple(
                        _factor_value(_factor_for_cell(group, cell, factor)) for factor in factors
                    )
                    values[cast(tuple[str, str], key)] = rows_by_name[cell.name][k]
                    surface = {
                        'Comparison': group.comparison_id,
                        'AnalysisBlock': group.analysis_block,
                        'InteractionType': 'response_surface',
                        'RunProfile': profile,
                        'EmbeddingModel': rows_by_name[cell.name][k].get('EmbeddingModel'),
                        'k': k,
                        f'Factor_{factors[0]}': key[0],
                        f'Factor_{factors[1]}': key[1],
                    }
                    for metric in _PRIMARY_METRICS:
                        surface[metric] = rows_by_name[cell.name][k].get(metric)
                    output.append(surface)
                levels_a = [
                    _factor_value(value) for value in group.factor_levels.get(factors[0], [])
                ]
                levels_b = [
                    _factor_value(value) for value in group.factor_levels.get(factors[1], [])
                ]
                if (
                    len(levels_a) != 2
                    or len(levels_b) != 2
                    or any((a, b) not in values for a in levels_a for b in levels_b)
                ):
                    continue
                did: dict[str, object] = {
                    'Comparison': group.comparison_id,
                    'AnalysisBlock': group.analysis_block,
                    'InteractionType': 'difference_in_differences',
                    'RunProfile': profile,
                    'EmbeddingModel': next(
                        (
                            value.get('EmbeddingModel')
                            for value in values.values()
                            if value.get('EmbeddingModel')
                        ),
                        None,
                    ),
                    'k': k,
                    f'Factor_{factors[0]}_low': levels_a[0],
                    f'Factor_{factors[0]}_high': levels_a[1],
                    f'Factor_{factors[1]}_low': levels_b[0],
                    f'Factor_{factors[1]}_high': levels_b[1],
                }
                for metric in _PRIMARY_METRICS:
                    corners = [
                        _number(values[(levels_a[1], levels_b[1])].get(metric)),
                        _number(values[(levels_a[1], levels_b[0])].get(metric)),
                        _number(values[(levels_a[0], levels_b[1])].get(metric)),
                        _number(values[(levels_a[0], levels_b[0])].get(metric)),
                    ]
                    if any(value is None for value in corners):
                        did[f'DiD_{metric}'] = None
                    else:
                        numeric_corners = cast(list[float], corners)
                        did[f'DiD_{metric}'] = (
                            numeric_corners[0]
                            - numeric_corners[1]
                            - numeric_corners[2]
                            + numeric_corners[3]
                        )
                output.append(did)
    return output


def crossing_rows(contrast_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Bracket zero and practical-margin crossings on declared factor orders."""
    grouped: dict[tuple[str, str, str, int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in contrast_rows:
        factor_columns = sorted(key for key in row if key.startswith('Factor_'))
        if len(factor_columns) != 1:
            continue
        grouped[
            (
                str(row['Comparison']),
                str(row['RunProfile']),
                str(row.get('EmbeddingModel') or 'unknown'),
                int(cast(int, row['k'])),
                factor_columns[0],
            )
        ].append(row)
    output: list[dict[str, object]] = []
    for (comparison, profile, model, k, factor), rows in grouped.items():
        order_column = f'FactorOrder_{factor.removeprefix("Factor_")}'
        raw_order = next((row.get(order_column) for row in rows if row.get(order_column)), '[]')
        try:
            order = list(json.loads(str(raw_order)))
        except json.JSONDecodeError:
            order = []
        rank = {str(level): index for index, level in enumerate(order)}
        ordered = sorted(rows, key=lambda row: rank.get(str(row[factor]), len(rank)))
        for metric in _PRIMARY_METRICS:
            for threshold in (0.0, -0.05, 0.05):
                for left, right in pairwise(ordered):
                    left_value, right_value = _number(left.get(metric)), _number(right.get(metric))
                    if left_value is None or right_value is None:
                        continue
                    if (left_value - threshold) * (right_value - threshold) <= 0:
                        output.append(
                            {
                                'Comparison': comparison,
                                'RunProfile': profile,
                                'EmbeddingModel': model,
                                'k': k,
                                'Factor': factor.removeprefix('Factor_'),
                                'Metric': metric.removeprefix('Delta_FacLoc_MMR_'),
                                'Threshold': threshold,
                                'LeftLevel': left[factor],
                                'RightLevel': right[factor],
                                'LeftDelta': left_value,
                                'RightDelta': right_value,
                            }
                        )
    return output


def write_suite_factor_figures(
    *,
    output_dir: Path,
    contrast_rows: Sequence[Mapping[str, object]],
    stems: Collection[str] | None = None,
) -> list[Path]:
    """Render compact, manifest-factor-driven scale and topology response plots."""
    from matplotlib import pyplot as plt

    figure_dir = output_dir / 'figures' / 'suite'
    figure_dir.mkdir(parents=True, exist_ok=True)
    for obsolete_stem in ('scale_by_objective', 'topology_by_objective'):
        for suffix in ('png', 'pdf'):
            (figure_dir / f'{obsolete_stem}.{suffix}').unlink(missing_ok=True)
    written: list[Path] = []
    # A raw line for every profile x k x comparison produces over one hundred
    # traces.  These figures instead show equal-weight means over those
    # evaluation conditions, retaining the declared manipulation as the line.
    selected_stems = set(stems) if stems is not None else None
    if selected_stems is None or 'scale_by_dataset_size' in selected_stems:
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
        if selected_stems is not None and stem not in selected_stems:
            continue
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
    factor_column = f'Factor_{factor}'
    comparison_set = set(comparison_ids)
    selected_rows = [
        row
        for row in rows
        if str(row.get('Comparison') or '') in comparison_set
        and row.get(factor_column) not in (None, '')
    ]
    if not selected_rows:
        return []

    levels = _ordered_factor_levels(selected_rows, factor=factor)
    if not levels:
        return []
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

    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    try:
        for metric, axis in zip(_PRIMARY_METRICS, axes.flat, strict=True):
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
                        linewidth=2.0,
                        label=line_label,
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
            set_axis_title(axis=axis, title=_PRIMARY_METRIC_LABELS[metric])
            axis.set_xlabel(_factor_axis_label(factor))
            axis.set_ylabel('FacLoc - MMR')
            axis.set_xticks(range(len(levels)), [_display_factor_level(level) for level in levels])
            axis.grid(axis='y', alpha=0.2)

        handles, labels = axes.flat[0].get_legend_handles_labels()
        if len(handles) > 1:
            figure.legend(
                handles,
                labels,
                loc='lower center',
                ncol=min(len(handles), 5),
                frameon=False,
                fontsize=9,
            )
            bottom = 0.14
        else:
            bottom = 0.08
        set_figure_title(
            figure=figure,
            title=f'{title} (mean across run profiles and K)',
            fontsize=16,
        )
        figure.subplots_adjust(
            left=0.08,
            right=0.98,
            top=title_aware_layout_top(titled_top=0.88, untitled_top=0.96),
            bottom=bottom,
            wspace=0.26,
            hspace=0.34,
        )
        written: list[Path] = []
        for suffix in ('png', 'pdf'):
            path = output_dir / f'{stem}.{suffix}'
            figure.savefig(path, dpi=180 if suffix == 'png' else None)
            written.append(path)
        return written
    finally:
        plt.close(figure)


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
    return _display_factor_level(str(row.get(f'Factor_{line_key}') or ''))


def _factor_axis_label(factor: str) -> str:
    return {
        'scale': 'Candidate-pool size',
        'background_topology': 'Background topology',
        'near_miss_topology': 'Near-miss topology',
    }.get(factor, _display_factor_level(factor))


def _display_factor_level(value: str) -> str:
    return value.replace('_', ' ').replace('x', '\N{MULTIPLICATION SIGN}').title()


def _group_cells(manifest: SuiteManifest, group: ComparisonGroup) -> list[SuiteManifestCell]:
    cells = [cell for cell in manifest.cells if cell.include_in_causal_summaries]
    if group.cells:
        wanted = set(group.cells)
        return [cell for cell in cells if cell.cell_id in wanted]
    wanted_distributions = set(group.distribution_ids)
    selected = [cell for cell in cells if cell.distribution_id in wanted_distributions]
    if group.run_profile_ids:
        selected = [cell for cell in selected if cell.run_profile_id in set(group.run_profile_ids)]
    else:
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


def _add_means(out: dict[str, object], rows: Sequence[Mapping[str, object]]) -> None:
    for metric in _PRIMARY_METRICS:
        values = [
            float(value)
            for row in rows
            if isinstance(
                (value := row.get(metric, row.get(f'{metric}_mean'))),
                int | float,
            )
        ]
        if values:
            out[f'{metric}_mean'] = statistics.fmean(values)


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
