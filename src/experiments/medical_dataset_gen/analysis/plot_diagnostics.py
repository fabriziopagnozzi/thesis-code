from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from experiments.medical_dataset_gen.analysis.analysis_constants import (
    DIVERSIFYING_STRATEGIES,
    EXPERIMENT_FAMILIES,
    EXPERIMENT_FAMILY_COLORS,
    EXPERIMENT_FAMILY_LABELS,
    ExperimentFamilyId,
)
from experiments.medical_dataset_gen.analysis.helpers import (
    float_or_none,
    quantile,
    short_experiment_label,
    short_model_label,
    strategy_label,
)
from experiments.medical_dataset_gen.analysis.models import PlotFormat


def annotate_horizontal_values(*, ax: Any, values: Sequence[float]) -> None:
    if not values:
        return
    min_value = min([0.0, *values])
    max_value = max([0.0, *values])
    span = max(max_value - min_value, 0.05)
    padding = span * 0.22
    ax.set_xlim(min_value - padding, max_value + padding)
    offset = max(span * 0.015, 0.003)
    for position, value in enumerate(values):
        if value >= 0:
            x_position = value + offset
            alignment = 'left'
        else:
            x_position = value - offset
            alignment = 'right'
        ax.text(
            x_position,
            position,
            f'{value:+.3f}',
            va='center',
            ha=alignment,
            fontsize=7,
            color='#202020',
        )


def _family_grouped_rows(
    rows: Sequence[Mapping[str, object]],
    value_column: str,
) -> list[Mapping[str, object]]:
    grouped: dict[ExperimentFamilyId, list[Mapping[str, object]]] = {}
    for row in rows:
        family_id = _family_id_for_row(row)
        grouped.setdefault(family_id, []).append(row)

    family_order: list[tuple[float, str, ExperimentFamilyId]] = []
    for family_id, group in grouped.items():
        values = [
            value
            for value in (float_or_none(row.get(value_column)) for row in group)
            if value is not None
        ]
        family_order.append((
            statistics.fmean(values) if values else float('-inf'),
            EXPERIMENT_FAMILY_LABELS[family_id],
            family_id,
        ))

    ordered_rows: list[Mapping[str, object]] = []
    for _mean_value, _label, family_id in sorted(
        family_order,
        key=lambda item: (-item[0], item[1]),
    ):
        ordered_rows.extend(
            sorted(
                grouped[family_id],
                key=lambda row: (
                    -(float_or_none(row.get(value_column)) or float('-inf')),
                    str(row.get('ShortExperiment') or row.get('Experiment') or ''),
                ),
            )
        )
    return ordered_rows


def _query_scope_paired_rows(
    rows: Sequence[Mapping[str, object]],
    value_column: str,
) -> list[Mapping[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        distribution = str(row.get('Distribution') or '')
        embedding_model = str(row.get('EmbeddingModel') or '')
        if distribution and embedding_model:
            grouped.setdefault((distribution, embedding_model), []).append(row)

    ordered_groups: list[tuple[float, tuple[str, str], list[Mapping[str, object]]]] = []
    for key, group in grouped.items():
        values = [
            value
            for value in (float_or_none(row.get(value_column)) for row in group)
            if value is not None
        ]
        if not values:
            continue
        ordered_groups.append((statistics.fmean(values), key, group))

    ordered_rows: list[Mapping[str, object]] = []
    # Horizontal plots use an inverted y-axis, so ascending row order places
    # higher-scoring pair groups lower in the rendered figure.
    for _mean_value, _key, group in sorted(ordered_groups, key=lambda item: item[0]):
        ordered_rows.extend(
            sorted(
                group,
                key=lambda row: (
                    1 if row.get('OnlyPassGeometry') is False else 0,
                    str(row.get('ShortExperiment') or row.get('Experiment') or ''),
                ),
            )
        )
    return ordered_rows


def _sorted_pair_rows_by_source_mean(
    rows: Sequence[Mapping[str, object]],
    *,
    pass_column: str,
    all_query_column: str,
) -> list[Mapping[str, object]]:
    return sorted(
        rows,
        key=lambda row: _mean_available(
            float_or_none(row.get(pass_column)),
            float_or_none(row.get(all_query_column)),
        ),
    )


def _mean_available(*values: float | None) -> float:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else float('-inf')


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
                key=lambda row: (
                    0 if row.get('OnlyPassGeometry') is True else 1,
                    str(row.get('ShortExperiment') or row.get('Experiment') or ''),
                ),
            )[0]
        )
    return _family_grouped_rows(representatives, 'GoldPercentage')


def family_color_for_row(row: Mapping[str, object]) -> str:
    return EXPERIMENT_FAMILY_COLORS[_family_id_for_row(row)]


def _family_id_for_row(row: Mapping[str, object]) -> ExperimentFamilyId:
    family_id = row.get('ExperimentFamily')
    if isinstance(family_id, str) and family_id in EXPERIMENT_FAMILIES:
        return cast(ExperimentFamilyId, family_id)
    return 'unknown'


def _add_family_legend(*, fig: Any, rows: Sequence[Mapping[str, object]]) -> None:
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
        loc='lower center',
        ncol=min(4, len(handles)),
        frameon=False,
        fontsize=8,
    )


def _color_tick_labels_by_family(*, ax: Any, rows: Sequence[Mapping[str, object]]) -> None:
    for tick_label, row in zip(ax.get_yticklabels(), rows, strict=False):
        tick_label.set_color(family_color_for_row(row))


def plot_geometry_pass_rate(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    plot_rows = _geometry_pass_rate_rows(rows)
    if not plot_rows:
        return []
    labels = [
        f'{row.get("ShortExperiment") or short_experiment_label(str(row.get("Experiment")))}/'
        f'{short_model_label(str(row.get("EmbeddingModel")))}'
        for row in plot_rows
    ]
    values = [float_or_none(row.get('GeometryPassRate')) or 0.0 for row in plot_rows]
    colors = [family_color_for_row(row) for row in plot_rows]
    fig_height = max(5.0, 0.28 * len(labels) + 1.6)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))  # type: ignore[attr-defined]
    try:
        ax.barh(range(len(labels)), values, color=colors)
        ax.set_title('Geometry filter pass rate by experiment and embedding')
        ax.set_xlabel('Pass rate')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.25)
        annotate_horizontal_values(ax=ax, values=values)
        ax.set_xlim(0, 1.08)
        _add_family_legend(fig=fig, rows=plot_rows)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        path = output_dir / f'geometry_pass_rate_by_embedding.{plot_format}'
        fig.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def _geometry_pass_rate_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    return _family_grouped_rows(
        [
            row
            for row in rows
            if float_or_none(row.get('GeometryPassRate')) is not None
        ],
        'GeometryPassRate',
    )


def plot_lambda_stability(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    plot_rows = [row for row in rows if row.get('strategy') in DIVERSIFYING_STRATEGIES]
    if not plot_rows:
        return []
    labels = [str(row.get('strategy')) for row in plot_rows]
    means = [float_or_none(row.get('selected_lambda_norm_mean')) or 0.0 for row in plot_rows]
    stds = [float_or_none(row.get('selected_lambda_norm_std')) or 0.0 for row in plot_rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))  # type: ignore[attr-defined]
    try:
        ax.bar(labels, means, yerr=stds, color=['#C47A3A', '#287C8E'], capsize=5)
        ax.set_title('Selected lambda stability')
        ax.set_ylabel('Normalized lambda mean ± std')
        ax.set_ylim(0, min(1.0, max(means + stds + [0.1]) + 0.15))
        ax.grid(axis='y', alpha=0.25)
        fig.tight_layout()
        path = output_dir / f'lambda_stability_boxplot.{plot_format}'
        fig.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def plot_lambda_safety_worst_delta(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    data: list[list[float]] = []
    labels: list[str] = []
    for strategy in DIVERSIFYING_STRATEGIES:
        values = [
            value
            for value in (
                float_or_none(row.get('WorstDeltaStrategyTopK_FCP'))
                for row in rows
                if row.get('strategy') == strategy
            )
            if value is not None
        ]
        if values:
            data.append(values)
            labels.append(strategy_label(strategy))
    if not data:
        return []

    fig, ax = plt.subplots(figsize=(6.8, 4.8))  # type: ignore[attr-defined]
    try:
        box = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        colors = ['#1F77B4', '#D62728']
        for patch, color in zip(box['boxes'], colors, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.axhline(0.0, color='#202020', linewidth=0.9)
        ax.set_title('Validation lambda safety: worst FCP delta vs top-k')
        ax.set_ylabel('Worst FacetCoveragePurity@k delta over lambda')
        ax.grid(axis='y', alpha=0.25)
        fig.tight_layout()
        path = output_dir / f'lambda_safety_worst_delta_vs_topk.{plot_format}'
        fig.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def plot_lambda_delta_curve(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))  # type: ignore[attr-defined]
    try:
        has_data = False
        for (strategy), color in (('mmr', '#1F77B4'), ('fac_loc', '#D62728')):
            points = [
                (
                    float_or_none(row.get('lambda_norm')),
                    float_or_none(row.get('DeltaStrategyTopK_FCP')),
                )
                for row in rows
                if row.get('strategy') == strategy
            ]
            binned = _binned_lambda_delta_stats(points, n_bins=20)
            if not binned:
                continue
            has_data = True
            xs = [item['x'] for item in binned]
            means = [item['mean'] for item in binned]
            lowers = [item['q25'] for item in binned]
            uppers = [item['q75'] for item in binned]
            ax.plot(xs, means, color=color, linewidth=2.0, label=strategy_label(strategy))  # type: ignore
            ax.fill_between(xs, lowers, uppers, color=color, alpha=0.16)

        if not has_data:
            plt.close(fig)  # type: ignore[attr-defined]
            return []
        ax.axhline(0.0, color='#202020', linewidth=0.9)
        ax.set_title('Validation FCP delta vs top-k across lambda')
        ax.set_xlabel('Normalized lambda within each strategy grid')
        ax.set_ylabel('Mean FacetCoveragePurity@k delta')
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = output_dir / f'lambda_fcp_delta_vs_topk_by_lambda.{plot_format}'
        fig.savefig(path, dpi=180)
        return [path]
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
        out.append({
            'x': (index + 0.5) / n_bins,
            'mean': statistics.fmean(values),
            'q25': quantile(sorted_values, 0.25),
            'q75': quantile(sorted_values, 0.75),
        })
    return out


def plot_near_optimal_width(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    data: list[list[float]] = []
    labels: list[str] = []
    for strategy in DIVERSIFYING_STRATEGIES:
        values = [
            value
            for value in (
                float_or_none(row.get('NearOptimalLambdaSpanNorm'))
                for row in rows
                if row.get('strategy') == strategy
            )
            if value is not None
        ]
        if values:
            data.append(values)
            labels.append(strategy)
    if not data:
        return []
    fig, ax = plt.subplots(figsize=(6.5, 4.5))  # type: ignore[attr-defined]
    try:
        ax.boxplot(data, tick_labels=labels, patch_artist=True)
        ax.set_title('Near-optimal lambda width')
        ax.set_ylabel('Normalized lambda span within epsilon of best FCP')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.25)
        fig.tight_layout()
        path = output_dir / f'near_optimal_lambda_width.{plot_format}'
        fig.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


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
    labels = [str(row.get('ShortDistribution') or row.get('Distribution')) for row in plot_rows]
    gold = [float_or_none(row.get('GoldPercentage')) or 0.0 for row in plot_rows]
    near = [float_or_none(row.get('NearMissDistractorPercentage')) or 0.0 for row in plot_rows]
    background = [float_or_none(row.get('BackgroundOutlierPercentage')) or 0.0 for row in plot_rows]
    positions = list(range(len(labels)))
    fig_height = max(5.0, 0.28 * len(labels) + 1.6)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))  # type: ignore[attr-defined]
    try:
        ax.barh(positions, gold, label='Gold', color='#287C8E')
        ax.barh(positions, near, left=gold, label='Near-miss distractors', color='#C47A3A')
        bottoms = [g + n for g, n in zip(gold, near, strict=True)]
        ax.barh(
            positions,
            background,
            left=bottoms,
            label='Background outliers',
            color='#6F7890',
        )
        ax.set_title('Candidate-pool composition')
        ax.set_xlabel('Share of qrel pool')
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.25)
        _color_tick_labels_by_family(ax=ax, rows=plot_rows)
        ax.legend()
        _add_family_legend(fig=fig, rows=plot_rows)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        path = output_dir / f'dataset_composition_stacked.{plot_format}'
        fig.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]
