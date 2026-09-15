from __future__ import annotations

import json
import re
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from experiments.medical_dataset_gen.reports.analysis_constants import (
    DISTRIBUTION_FAMILY_ABBREVIATIONS,
    EXPERIMENT_FAMILIES,
    ExperimentFamilyId,
    StrategyName,
)
from experiments.medical_dataset_gen.reports.helpers import (
    float_or_none,
    interaction_distribution_label,
    quantile,
    strategy_label,
)
from experiments.medical_dataset_gen.reports.models import PlotFormat
from experiments.medical_dataset_gen.reports.plot_rendering import set_axis_title
from experiments.medical_dataset_gen.reports.report_config import OBJECTIVE_COLORS

PRIMARY_GOLD_ROLE_STACKS: tuple[tuple[str, str, str], ...] = (
    ('Facet 1', 'DominantPrimaryGoldCountMean', '#2166AC'),
    ('Facet 2', 'OtherPrimaryGoldCountMean', '#4393C3'),
)

# Secondary and niche evidence is generated as one cluster per facet.  Keeping
# individual facets separate in this audit makes the intended multi-aspect
# composition visible instead of merging them into one broad gold segment.
FACET_GOLD_ROLE_STACKS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        'Facet',
        'SecondaryGoldCountMean',
        'secondary_gold',
        ('#92C5DE', '#D1E5F0'),
    ),
    (
        'Niche Facet',
        'NicheGoldCountMean',
        'niche_gold',
        ('#9970AB', '#C2A5CF'),
    ),
)

# Composition figures reserve a fixed vertical budget for each distribution.
# This keeps bar thickness comparable between the complete audit and small
# subsets such as the four interaction pools.
COMPOSITION_FIGURE_WIDTH_IN = 9
COMPOSITION_ROW_HEIGHT_IN = 0.25
COMPOSITION_FAMILY_GAP_HEIGHT_IN = 0.10
COMPOSITION_FIXED_HEIGHT_IN = 1.3
COMPOSITION_BAR_HEIGHT = 0.17
COMPOSITION_BAR_ALPHA = 0.70
BETWEEN_FAMILIES_OFFSET = 0.30
WITHIN_FAMILY_OFFSET = 0.25
COMPOSITION_LEGEND_FONT_SIZE = 10
COMPOSITION_LEGEND_COLUMNS = 4
COMPOSITION_LEGEND_ROW_HEIGHT_IN = 0.20
SCALE_LABEL_ABBREVIATIONS: dict[str, str] = {
    'small': 'S',
    'medium': 'M',
    'large': 'L',
}


def _representative_distribution_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        distribution = str(row.get('Distribution') or '')
        if distribution:
            grouped.setdefault(distribution, []).append(row)

    representatives: list[Mapping[str, object]] = []
    for _distribution, group in grouped.items():
        representatives.append(
            sorted(
                group,
                key=lambda row: str(row.get('ShortExperiment') or row.get('Experiment') or ''),
            )[0]
        )
    # Composition is a design audit: order conditions by their controlled
    # factors, not by the realized share that the bars happen to display.
    return sorted(representatives, key=_composition_row_sort_key)


def _composition_row_sort_key(row: Mapping[str, object]) -> tuple[int, int, int, int, str]:
    family_rank = {family_id: index for index, family_id in enumerate(EXPERIMENT_FAMILIES)}
    family = _family_id_for_row(row)
    distribution = str(row.get('Distribution') or '')
    scale = str(row.get('Factor_scale') or '')
    scale_rank = {'small': 0, 'medium': 1, 'large': 2}.get(scale, 3)

    if family == 'background_variant':
        if distribution.startswith('dilution_far_b'):
            mass = int(distribution.removeprefix('dilution_far_b') or 0)
            return family_rank[family], 0, mass, 0, distribution
        if distribution.startswith('background_'):
            topology = str(row.get('Factor_background_topology') or '')
            clusters = int(topology.split('x', 1)[0]) if 'x' in topology else 0
            return family_rank[family], 1, 0, -clusters, distribution
        return family_rank[family], 2, scale_rank, 0, distribution

    if family == 'dominance':
        if distribution.startswith('dominance_'):
            level = str(row.get('Factor_dominance_level') or '')
            level_rank = {'mild': 0, 'high': 1, 'extreme': 2}.get(level, 3)
            return family_rank[family], 0, level_rank, 0, distribution
        return family_rank[family], 1, scale_rank, 0, distribution

    if family == 'sparse_niche':
        if distribution.startswith('sparse_'):
            mode = str(row.get('Factor_sparse_mode') or '')
            level = str(row.get('Factor_sparse_level') or '')
            mode_rank = {'one': 0, 'two': 1}.get(mode, 2)
            level_rank = {'moderate': 0, 'severe': 1, 'extreme': 2}.get(level, 3)
            return family_rank[family], 0, mode_rank, level_rank, distribution
        return family_rank[family], 1, scale_rank, 0, distribution

    if family == 'near_miss_heavy':
        if distribution.startswith('near_miss_h'):
            suffix = distribution.removeprefix('near_miss_h')
            mass_token = suffix.split('_', 1)[0]
            mass = int(mass_token) if mass_token.isdigit() else 0
            return family_rank[family], 0, mass, 0, distribution
        return family_rank[family], 1, scale_rank, 0, distribution

    if family == 'balanced_clean':
        balanced_rank = {
            'scale_balanced_small': 0,
            'balanced_reference': 1,
            'scale_balanced_large': 2,
        }.get(distribution, 3)
        return family_rank[family], 0, balanced_rank, 0, distribution

    return family_rank[family], 0, 0, 0, distribution


def _family_id_for_row(row: Mapping[str, object]) -> ExperimentFamilyId:
    family_id = row.get('ExperimentFamily')
    if isinstance(family_id, str) and family_id in EXPERIMENT_FAMILIES:
        return cast(ExperimentFamilyId, family_id)
    return 'unknown'


def plot_lambda_delta_curve(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    curve_rows: Sequence[Mapping[str, object]] = (),
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.2), sharex=True)  # type: ignore[attr-defined]
    axes = list(axes)
    try:
        has_data = False
        for strategy in ('mmr', 'fac_loc'):
            color = OBJECTIVE_COLORS[strategy]
            typed_strategy = cast(StrategyName, strategy)
            summary = [row for row in curve_rows if row.get('strategy') == strategy]
            if summary:
                xs = [float_or_none(row.get('lambda_norm')) for row in summary]
                means = [float_or_none(row.get('MeanDeltaStrategyTopK_FCP')) for row in summary]
                lowers = [float_or_none(row.get('CellQ25DeltaStrategyTopK_FCP')) for row in summary]
                uppers = [float_or_none(row.get('CellQ75DeltaStrategyTopK_FCP')) for row in summary]
                safe = [float_or_none(row.get('CellSafeLambdaFraction')) for row in summary]
                distractor = [
                    float_or_none(row.get('MeanDeltaStrategyTopK_DistractorRate'))
                    for row in summary
                ]
            else:
                points = [
                    (
                        float_or_none(row.get('lambda_norm')),
                        float_or_none(row.get('DeltaStrategyTopK_FCP')),
                    )
                    for row in rows
                    if row.get('strategy') == strategy
                ]
                binned = _binned_lambda_delta_stats(points, n_bins=20)
                xs = [item['x'] for item in binned]
                means = [item['mean'] for item in binned]
                lowers = [item['q25'] for item in binned]
                uppers = [item['q75'] for item in binned]
                safe = []
                distractor = []
            points = [
                (x, mean, lower, upper)
                for x, mean, lower, upper in zip(xs, means, lowers, uppers, strict=True)
                if x is not None and mean is not None and lower is not None and upper is not None
            ]
            if not points:
                continue
            has_data = True
            valid_xs = [point[0] for point in points]
            valid_means = [point[1] for point in points]
            valid_lowers = [point[2] for point in points]
            valid_uppers = [point[3] for point in points]
            axes[0].plot(
                valid_xs,
                valid_means,
                color=color,
                linewidth=2.0,
                label=strategy_label(typed_strategy),
            )
            axes[0].fill_between(valid_xs, valid_lowers, valid_uppers, color=color, alpha=0.16)

            if summary:
                valid_safe = [
                    (x, value)
                    for x, value in zip(xs, safe, strict=True)
                    if x is not None and value is not None
                ]
                if valid_safe:
                    axes[1].plot(
                        [item[0] for item in valid_safe],
                        [item[1] for item in valid_safe],
                        color=color,
                        linewidth=2.0,
                        label=strategy_label(typed_strategy),
                    )
                valid_distractor = [
                    (x, value)
                    for x, value in zip(xs, distractor, strict=True)
                    if x is not None and value is not None
                ]
                if valid_distractor:
                    axes[2].plot(
                        [item[0] for item in valid_distractor],
                        [item[1] for item in valid_distractor],
                        color=color,
                        linewidth=2.0,
                        label=strategy_label(typed_strategy),
                    )

        if not has_data:
            plt.close(fig)  # type: ignore[attr-defined]
            return []
        axes[0].axhline(0.0, color='#202020', linewidth=0.9)
        set_axis_title(axis=axes[0], title='Validation FCP delta vs top-k across lambda')
        axes[0].set_ylabel('Mean FCP delta')
        axes[0].grid(alpha=0.25)

        axes[1].set_ylim(0.0, 1.0)
        set_axis_title(axis=axes[1], title='Cell-level lambda safety')
        axes[1].set_ylabel('Fraction at or above top-k')
        axes[1].grid(alpha=0.25)

        axes[2].axhline(0.0, color='#202020', linewidth=0.9)
        set_axis_title(axis=axes[2], title='Validation distractor-rate delta vs top-k')
        axes[2].set_xlabel('Normalized lambda within each strategy grid')
        axes[2].set_ylabel('Mean distractor-rate delta')
        axes[2].grid(alpha=0.25)
        axes[2].set_xlim(0, 1)
        handles, _labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(frameon=False)
        fig.tight_layout()
        return _save_plot_with_vector_copy(
            figure=fig,
            output_dir=output_dir,
            stem='lambda_fcp_delta_vs_topk_by_lambda',
            plot_format=plot_format,
        )
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def _binned_lambda_delta_stats(
    points: Sequence[tuple[float | None, float | None]],
    *,
    n_bins: int,
) -> list[dict[str, float]]:
    bins: list[list[float]] = [[] for _ in range(n_bins)]
    for lambda_norm, delta in points:
        if lambda_norm is None or delta is None:
            continue
        clipped = min(1.0, max(0.0, lambda_norm))
        index = min(n_bins - 1, int(clipped * n_bins))
        bins[index].append(delta)

    out: list[dict[str, float]] = []
    for index, values in enumerate(bins):
        if not values:
            continue
        sorted_values = sorted(values)
        out.append(
            {
                'x': (index + 0.5) / n_bins,
                'mean': statistics.fmean(values),
                'q25': quantile(sorted_values, 0.25),
                'q75': quantile(sorted_values, 0.75),
            }
        )
    return out


def _save_plot_with_vector_copy(
    *,
    figure: object,
    output_dir: Path,
    stem: str,
    plot_format: PlotFormat,
) -> list[Path]:
    """Save the requested report image and a vector twin when rendering PNG."""
    path = output_dir / f'{stem}.{plot_format}'
    figure.savefig(path, dpi=180, bbox_inches='tight')  # type: ignore[attr-defined]
    paths = [path]
    if plot_format == 'png':
        pdf_path = output_dir / f'{stem}.pdf'
        figure.savefig(pdf_path, bbox_inches='tight')  # type: ignore[attr-defined]
        paths.append(pdf_path)
    return paths


def plot_dataset_composition(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    plot_rows = [
        row
        for row in rows
        if all(
            float_or_none(row.get(column)) is not None
            for column in (
                'GoldPercentage',
                'NearMissDistractorPercentage',
                'BackgroundOutlierPercentage',
            )
        )
    ]
    plot_rows = _representative_distribution_rows(plot_rows)
    if not plot_rows:
        return []
    # ``ShortDistribution`` is intentionally compact for tables, but collapses
    # native-v5 IDs such as ``dominance_high`` and ``dominance_extreme`` to the
    # same token.  Composition is an audit plot, so every row needs its full,
    # human-readable distribution identifier.
    labels = [_distribution_composition_label(row) for row in plot_rows]
    gold = [float_or_none(row.get('GoldPercentage')) or 0.0 for row in plot_rows]
    near = [float_or_none(row.get('NearMissDistractorPercentage')) or 0.0 for row in plot_rows]
    background = [float_or_none(row.get('BackgroundOutlierPercentage')) or 0.0 for row in plot_rows]
    positions = _family_spaced_positions(plot_rows)
    family_gap_count = max(0, len(set(_family_id_for_row(row) for row in plot_rows)) - 1)
    fig_height = (
        COMPOSITION_ROW_HEIGHT_IN * len(labels)
        + COMPOSITION_FAMILY_GAP_HEIGHT_IN * family_gap_count
        + COMPOSITION_FIXED_HEIGHT_IN
    )
    fig, ax = plt.subplots(  # type: ignore[attr-defined]
        figsize=(COMPOSITION_FIGURE_WIDTH_IN, fig_height)
    )
    try:
        granular_gold = _gold_role_share_series(plot_rows)
        if granular_gold is None:
            displayed_gold = gold
            ax.barh(
                positions,
                gold,
                height=COMPOSITION_BAR_HEIGHT,
                label='Gold',
                color='#287C8E',
                alpha=COMPOSITION_BAR_ALPHA,
            )
        else:
            left = [0.0 for _row in plot_rows]
            for label, _column, color, values in granular_gold:
                ax.barh(
                    positions,
                    values,
                    height=COMPOSITION_BAR_HEIGHT,
                    left=left,
                    label=label,
                    color=color,
                    alpha=COMPOSITION_BAR_ALPHA,
                )
                left = [old + value for old, value in zip(left, values, strict=True)]
            displayed_gold = left
        ax.barh(
            positions,
            near,
            height=COMPOSITION_BAR_HEIGHT,
            left=displayed_gold,
            label='Near-miss distractors',
            color='#CFA79A',
            alpha=COMPOSITION_BAR_ALPHA,
        )
        bottoms = [g + n for g, n in zip(displayed_gold, near, strict=True)]
        ax.barh(
            positions,
            background,
            height=COMPOSITION_BAR_HEIGHT,
            left=bottoms,
            label='Background outliers (clusters x chunks)',
            color='#6F7890',
            alpha=COMPOSITION_BAR_ALPHA,
        )
        _annotate_background_topologies(
            ax=ax,
            positions=positions,
            lefts=bottoms,
            widths=background,
            rows=plot_rows,
        )
        ax.set_xlabel('Share of qrel pool')
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_ylim(
            positions[-1] + COMPOSITION_BAR_HEIGHT / 2.0,
            positions[0] - COMPOSITION_BAR_HEIGHT / 2.0,
        )
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.25)
        legend_handles, legend_labels = ax.get_legend_handles_labels()
        legend_ncol = min(COMPOSITION_LEGEND_COLUMNS, len(legend_handles))
        legend_rows = (len(legend_handles) + legend_ncol - 1) // legend_ncol
        fig.legend(
            legend_handles,
            legend_labels,
            loc='lower center',
            ncol=legend_ncol,
            frameon=False,
            fontsize=COMPOSITION_LEGEND_FONT_SIZE,
            bbox_to_anchor=(0.5, 0.0),
            borderaxespad=0.2,
        )
        # Reserve only the measured number of wrapped legend rows so tall audit
        # plots do not acquire conspicuous empty space below the axes.
        legend_height = (0.03 + COMPOSITION_LEGEND_ROW_HEIGHT_IN * legend_rows) / fig_height
        fig.tight_layout(rect=(0, legend_height, 1, 1), pad=0.25)
        path = output_dir / f'dataset_composition_stacked.{plot_format}'
        fig.savefig(path, dpi=180, bbox_inches='tight', pad_inches=0.02)
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def _distribution_composition_label(row: Mapping[str, object]) -> str:
    """Return an unambiguous, compact display label for a pool distribution."""
    distribution = str(row.get('Distribution') or row.get('ShortDistribution') or '')
    family = _family_id_for_row(row)
    label = _compact_distribution_composition_label(
        distribution=distribution,
        family=family,
    )
    pool_size = float_or_none(row.get('PoolSizeMean'))
    if pool_size is None:
        return label
    return f'{label} (N={pool_size:.0f})'


def _compact_distribution_composition_label(
    *,
    distribution: str,
    family: ExperimentFamilyId,
) -> str:
    """Render composition rows with the shared distribution-family abbreviations."""
    if family == 'interaction':
        return interaction_distribution_label(distribution)

    is_scale = distribution.startswith('scale_')
    token = distribution.removeprefix('scale_')
    abbreviation = DISTRIBUTION_FAMILY_ABBREVIATIONS[family]
    if family == 'balanced_clean':
        token = token.removeprefix('balanced_').replace('reference', 'ref')
    elif family == 'dominance':
        token = token.removeprefix('dominance_')
    elif family == 'sparse_niche':
        token = token.removeprefix('sparse_')
    elif family == 'near_miss_heavy':
        token = token.removeprefix('near_miss_')
    elif family == 'background_variant':
        token = token.removeprefix('background_').removeprefix('dilution_')
        token = token.removeprefix('compact_background_')

    if is_scale:
        for scale_label, abbreviation_label in SCALE_LABEL_ABBREVIATIONS.items():
            if token == scale_label:
                token = abbreviation_label
                break
            if token.endswith(f'_{scale_label}'):
                token = f'{token.removesuffix(scale_label)}{abbreviation_label}'
                break
    token = re.sub(r'(?<=\d)x(?=\d)', '\N{MULTIPLICATION SIGN}', token)
    token = re.sub(r'^h(?=\d)', 'H', token)
    token = re.sub(r'^b(?=\d)', 'B', token)
    token = token.replace('_b', '_B')
    token = token.replace('_', '-')
    return f'{abbreviation}-scale-{token}' if is_scale else f'{abbreviation}-{token}'


def _family_spaced_positions(rows: Sequence[Mapping[str, object]]) -> list[float]:
    positions: list[float] = []
    current_position = 0.0
    previous_family_id: ExperimentFamilyId | None = None
    for row in rows:
        family_id = _family_id_for_row(row)
        if previous_family_id is not None and family_id != previous_family_id:
            current_position += BETWEEN_FAMILIES_OFFSET
        positions.append(current_position)
        current_position += WITHIN_FAMILY_OFFSET
        previous_family_id = family_id
    return positions


def _gold_role_share_series(
    rows: Sequence[Mapping[str, object]],
) -> list[tuple[str, str, str, list[float]]] | None:
    series: list[tuple[str, str, str, list[float]]] = []
    for label, column, color in PRIMARY_GOLD_ROLE_STACKS:
        values: list[float] = []
        for row in rows:
            pool_size = float_or_none(row.get('PoolSizeMean'))
            count = float_or_none(row.get(column))
            if pool_size is None or pool_size <= 0.0 or count is None:
                return None
            values.append(count / pool_size)
        series.append((label, column, color, values))

    for role_label, column, topology_key, colors in FACET_GOLD_ROLE_STACKS:
        facet_counts = [_facet_count_for_role(row, column, topology_key) for row in rows]
        max_facet_count = max(facet_counts, default=0)
        if max_facet_count == 0:
            continue
        for facet_index in range(max_facet_count):
            values = []
            for row, facet_count in zip(rows, facet_counts, strict=True):
                pool_size = float_or_none(row.get('PoolSizeMean'))
                count = float_or_none(row.get(column))
                if pool_size is None or pool_size <= 0.0 or count is None:
                    return None
                values.append(count / pool_size / facet_count if facet_index < facet_count else 0.0)
            label = (
                f'Facet {facet_index + 3}'
                if role_label == 'Facet'
                else f'{role_label} {facet_index + 1}'
            )
            series.append((label, column, colors[facet_index % len(colors)], values))
    return series


def _facet_count_for_role(
    row: Mapping[str, object],
    count_column: str,
    topology_key: str,
) -> int:
    """Return the number of evidence facets from the realized cluster audit."""
    total_count = float_or_none(row.get(count_column))
    if total_count is None or total_count <= 0.0:
        return 0
    topology = _realized_cluster_topology(row)
    role_topology = topology.get(topology_key)
    if not isinstance(role_topology, Mapping):
        return 1
    clusters = float_or_none(role_topology.get('clusters_per_query'))
    if clusters is None or clusters < 1.0:
        return 1
    return max(1, round(clusters))


def _annotate_background_topologies(
    *,
    ax: Any,
    positions: Sequence[float],
    lefts: Sequence[float],
    widths: Sequence[float],
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Label background shares by realized ``clusters x chunks-per-cluster``."""
    for position, left, width, row in zip(positions, lefts, widths, rows, strict=True):
        label = _background_topology_label(row)
        if label is None or width <= 0.0:
            continue
        if width >= 0.06:
            ax.text(
                left + width / 2.0,
                position,
                label,
                ha='center',
                va='center',
                fontsize=6.5,
                color='white',
                fontweight='bold',
            )
        else:
            ax.text(
                left + width + 0.006,
                position,
                label,
                ha='left',
                va='center',
                fontsize=6.5,
                color='#455064',
            )


def _background_topology_label(row: Mapping[str, object]) -> str | None:
    topology = _realized_cluster_topology(row)
    background = topology.get('background_outlier')
    if not isinstance(background, Mapping):
        return None
    clusters = float_or_none(background.get('clusters_per_query'))
    chunks = float_or_none(background.get('chunks_per_cluster'))
    if clusters is None or chunks is None or clusters <= 0.0 or chunks <= 0.0:
        return None
    return f'{clusters:.0f}x{chunks:.0f}'


def _realized_cluster_topology(row: Mapping[str, object]) -> Mapping[str, object]:
    raw_topology = row.get('RealizedClusterTopologyJson')
    if not isinstance(raw_topology, str):
        return {}
    try:
        topology = json.loads(raw_topology)
    except json.JSONDecodeError:
        return {}
    return topology if isinstance(topology, dict) else {}
