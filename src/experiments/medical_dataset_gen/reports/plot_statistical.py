"""Figures for paired, profile-clustered statistical summaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.medical_dataset_gen.reports.helpers import float_or_none, short_experiment_id
from experiments.medical_dataset_gen.reports.models import PlotFormat
from experiments.medical_dataset_gen.reports.statistical import CORE_EMBEDDING_MODELS


def plot_paired_fcp_forest(
    *,
    plt: object,
    cell_rows: Sequence[Mapping[str, object]],
    suite_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    """Plot low-budget core-suite FCP paired effects with profile bootstrap CIs."""
    rows = [
        row
        for row in cell_rows
        if row.get('BudgetCategory') == 'low_budget'
        and row.get('EmbeddingModel') in CORE_EMBEDDING_MODELS
        and float_or_none(row.get('MeanDeltaFacLocMMR')) is not None
        and float_or_none(row.get('CI95Low')) is not None
        and float_or_none(row.get('CI95High')) is not None
    ]
    if not rows:
        return []
    rows.sort(
        key=lambda row: (
            str(row.get('ExperimentFamilyLabel') or ''),
            float_or_none(row.get('MeanDeltaFacLocMMR')) or 0.0,
            str(row.get('Experiment') or ''),
        )
    )
    labels = [
        f'{short_experiment_id(str(row.get("Experiment") or ""))} | '
        f'{str(row.get("EmbeddingModel") or "").rsplit("/", 1)[-1]}'
        for row in rows
    ]
    means = [float_or_none(row.get('MeanDeltaFacLocMMR')) or 0.0 for row in rows]
    lows = [float_or_none(row.get('CI95Low')) or 0.0 for row in rows]
    highs = [float_or_none(row.get('CI95High')) or 0.0 for row in rows]
    positions = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(11.5, max(6.0, 0.22 * len(rows) + 2.2)))  # type: ignore[attr-defined]
    try:
        ax.errorbar(
            means,
            positions,
            xerr=[
                [mean - low for mean, low in zip(means, lows, strict=True)],
                [high - mean for mean, high in zip(means, highs, strict=True)],
            ],
            fmt='o',
            color='#287C8E',
            ecolor='#6B7280',
            capsize=2.5,
            markersize=4,
        )
        ax.axvline(0.0, color='#111827', linewidth=0.9)
        threshold = float_or_none(rows[0].get('PracticalThreshold'))
        if threshold is not None:
            ax.axvspan(-threshold, threshold, color='#E5E7EB', alpha=0.65, zorder=0)
        core = next(
            (
                row
                for row in suite_rows
                if row.get('Scope') == 'Core suite' and row.get('BudgetCategory') == 'low_budget'
            ),
            None,
        )
        if core is not None:
            mean = float_or_none(core.get('MeanDeltaFacLocMMR'))
            low = float_or_none(core.get('CI95Low'))
            high = float_or_none(core.get('CI95High'))
            if mean is not None and low is not None and high is not None:
                summary_position = len(rows) + 1
                ax.errorbar(
                    [mean],
                    [summary_position],
                    xerr=[[mean - low], [high - mean]],
                    fmt='D',
                    color='#C44E52',
                    ecolor='#C44E52',
                    capsize=3,
                    markersize=6,
                )
                positions.append(summary_position)
                labels.append('Core suite (equal-family weighted)')
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel('Held-out FacLoc - MMR FCP effect (95% profile-bootstrap CI)')
        ax.set_title('Low-budget paired FCP effects in the fully crossed core suite')
        ax.grid(axis='x', alpha=0.22)
        fig.tight_layout()
        path = output_dir / f'paired_fcp_low_budget_forest.{plot_format}'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def plot_paired_fcp_config_forest(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    """Plot low-budget paired FCP effects by wording configuration."""
    plot_rows = [
        row
        for row in rows
        if row.get('BudgetCategory') == 'low_budget'
        and row.get('Scope') == 'Configuration'
        and float_or_none(row.get('MeanDeltaFacLocMMR')) is not None
        and float_or_none(row.get('CI95Low')) is not None
        and float_or_none(row.get('CI95High')) is not None
    ]
    if len({row.get('WordingConfig') for row in plot_rows if row.get('WordingConfig')}) <= 1:
        return []
    plot_rows.sort(key=_config_forest_sort_key)
    return _plot_forest(
        plt=plt,
        rows=plot_rows,
        labels=[
            str(row.get('WordingConfigLabel') or row.get('WordingConfig') or '')
            for row in plot_rows
        ],
        title='Low-budget paired FCP effects by wording configuration',
        xlabel='Held-out FacLoc - MMR FCP effect (95% profile-bootstrap CI)',
        output_path=output_dir / f'paired_fcp_low_budget_forest_by_config.{plot_format}',
        plot_format=plot_format,
    )


def plot_paired_fcp_config_embedding_forest(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    """Plot low-budget paired FCP effects by wording configuration and core model."""
    plot_rows = [
        row
        for row in rows
        if row.get('BudgetCategory') == 'low_budget'
        and row.get('Scope') == 'Configuration x embedding'
        and row.get('EmbeddingModel') in CORE_EMBEDDING_MODELS
        and float_or_none(row.get('MeanDeltaFacLocMMR')) is not None
        and float_or_none(row.get('CI95Low')) is not None
        and float_or_none(row.get('CI95High')) is not None
    ]
    if len({row.get('WordingConfig') for row in plot_rows if row.get('WordingConfig')}) <= 1:
        return []
    plot_rows.sort(key=_config_embedding_forest_sort_key)
    labels = [
        f'{row.get("WordingConfigLabel") or row.get("WordingConfig")} | '
        f'{str(row.get("EmbeddingModel") or "").rsplit("/", 1)[-1]}'
        for row in plot_rows
    ]
    return _plot_forest(
        plt=plt,
        rows=plot_rows,
        labels=labels,
        title='Low-budget paired FCP effects by wording configuration and embedding model',
        xlabel='Held-out FacLoc - MMR FCP effect (95% profile-bootstrap CI)',
        output_path=output_dir / f'paired_fcp_low_budget_forest_by_config_emb_model.{plot_format}',
        plot_format=plot_format,
    )


def _plot_forest(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    labels: Sequence[str],
    title: str,
    xlabel: str,
    output_path: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    means = [float_or_none(row.get('MeanDeltaFacLocMMR')) or 0.0 for row in rows]
    lows = [float_or_none(row.get('CI95Low')) or 0.0 for row in rows]
    highs = [float_or_none(row.get('CI95High')) or 0.0 for row in rows]
    positions = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(11.5, max(5.4, 0.36 * len(rows) + 2.0)))  # type: ignore[attr-defined]
    try:
        colors = [_forest_color(row) for row in rows]
        _draw_errorbar_points(ax, means, lows, highs, positions, colors)
        ax.axvline(0.0, color='#111827', linewidth=0.9)
        threshold = float_or_none(rows[0].get('PracticalThreshold'))
        if threshold is not None:
            ax.axvspan(-threshold, threshold, color='#E5E7EB', alpha=0.65, zorder=0)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.22)
        _add_model_legend(ax, rows)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')
        return [output_path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def _draw_errorbar_points(
    ax: Any,
    means: Sequence[float],
    lows: Sequence[float],
    highs: Sequence[float],
    positions: Sequence[int],
    colors: Sequence[str],
) -> None:
    for mean, low, high, position, color in zip(
        means,
        lows,
        highs,
        positions,
        colors,
        strict=True,
    ):
        ax.errorbar(
            [mean],
            [position],
            xerr=[[mean - low], [high - mean]],
            fmt='o',
            color=color,
            ecolor='#6B7280',
            capsize=2.5,
            markersize=4.5,
        )


def _add_model_legend(ax: Any, rows: Sequence[Mapping[str, object]]) -> None:
    from matplotlib.patches import Patch

    models = [model for model in CORE_EMBEDDING_MODELS if _model_present(rows, model)]
    if len(models) <= 1:
        return
    handles = [
        Patch(facecolor=_model_color(model), label=model.rsplit('/', 1)[-1]) for model in models
    ]
    ax.legend(
        handles=handles,
        loc='upper left',
        frameon=False,
        fontsize=9,
    )


def _model_present(rows: Sequence[Mapping[str, object]], model: str) -> bool:
    return any(row.get('EmbeddingModel') == model for row in rows)


def _forest_color(row: Mapping[str, object]) -> str:
    model = row.get('EmbeddingModel')
    if model in CORE_EMBEDDING_MODELS:
        return _model_color(str(model))
    return '#287C8E'


def _model_color(model: str) -> str:
    if model == 'BAAI/bge-m3':
        return '#287C8E'
    if model == 'Qwen/Qwen3-Embedding-0.6B':
        return '#C44E52'
    return '#6B7280'


def _config_forest_sort_key(row: Mapping[str, object]) -> tuple[tuple[int, int, int, str], float]:
    return (
        _config_sort_key(str(row.get('WordingConfig') or '')),
        -(float_or_none(row.get('MeanDeltaFacLocMMR')) or -math.inf),
    )


def _config_embedding_forest_sort_key(
    row: Mapping[str, object],
) -> tuple[tuple[int, int, int, str], int, float]:
    model_order = {model: index for index, model in enumerate(CORE_EMBEDDING_MODELS)}
    return (
        _config_sort_key(str(row.get('WordingConfig') or '')),
        model_order.get(str(row.get('EmbeddingModel') or ''), 99),
        -(float_or_none(row.get('MeanDeltaFacLocMMR')) or -math.inf),
    )


def _config_sort_key(config: str) -> tuple[int, int, int, str]:
    parts = config.split('_')
    query_mode = parts[0] if len(parts) > 0 else ''
    focus_mode = parts[2] if len(parts) > 2 else ''
    chunk_mode = parts[4] if len(parts) > 4 else ''
    query_order = {'biased': 0, 'unbiased': 1}
    focus_order = {'list': 0, 'natural': 1}
    chunk_order = {'simple': 0, 'hardened': 1}
    return (
        query_order.get(query_mode, 99),
        focus_order.get(focus_mode, 99),
        chunk_order.get(chunk_mode, 99),
        config,
    )
