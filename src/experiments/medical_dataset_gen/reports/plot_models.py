"""Focused thesis figures for representation-model comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.medical_dataset_gen.reports.helpers import embedding_model_sort_key, float_or_none
from experiments.medical_dataset_gen.reports.report_config import (
    embedding_model_color,
    embedding_model_display_label,
)

MODEL_FIGURE_TITLE_SIZE = 16
MODEL_AXIS_TITLE_SIZE = 14
MODEL_AXIS_LABEL_SIZE = 12
MODEL_TICK_LABEL_SIZE = 11
MODEL_LEGEND_SIZE = 11
MODEL_VALUE_LABEL_SIZE = 10
MODEL_HEATMAP_CELL_SIZE = 12

PASTEL_OBJECTIVE_COLORS: dict[str, str] = {
    'top_k': '#7F95B2',  # soft steel blue
    'mmr': '#4FA3C7',  # fresh cyan-blue
    'fac_loc': '#D88478',  # coral
}

PASTEL_EMBEDDING_MODEL_COLORS: dict[str, str] = {
    'Qwen/Qwen3-Embedding-0.6B': '#5B9FD3',  # blue
    'Qwen/Qwen3-Embedding-4B': '#9A6BC3',  # violet
    'abhinand/MedEmbed-large-v0.1': '#DB836B',  # terracotta/coral
    'multi-qa-mpnet-base-cos-v1': '#63AE8B',  # jade green
}

PASTEL_GEOMETRY_COLORS: dict[str, str] = {
    # Muted lilac keeps coverage stress distinct from the salmon near-miss and
    # blue background-margin series while retaining the pastel visual language.
    'coverage_stress': '#B6A6C5',
    'near_miss': '#DF8D73',
    'background': '#72B7D2',
}


def write_embedding_model_figures(
    *,
    plt: Any,
    output_dir: Path,
    metric_rows: Sequence[Mapping[str, object]],
    geometry_rows: Sequence[Mapping[str, object]],
    geometry_family_rows: Sequence[Mapping[str, object]],
    lambda_rows: Sequence[Mapping[str, object]],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.extend(_embedding_model_fcp_overview(plt=plt, rows=metric_rows, output_dir=output_dir))
    paths.extend(_embedding_geometry_overview(plt=plt, rows=geometry_rows, output_dir=output_dir))
    paths.extend(
        _embedding_geometry_family_heatmap(
            plt=plt,
            rows=geometry_family_rows,
            output_dir=output_dir,
        )
    )
    paths.extend(
        _lambda_fcp_delta_by_embedding_model(
            plt=plt,
            rows=lambda_rows,
            output_dir=output_dir,
        )
    )
    return paths


def _embedding_model_fcp_overview(
    *,
    plt: Any,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> list[Path]:
    selected = [
        row for row in rows if row.get('MetricLabel') == 'FCP' and row.get('Scope') == 'overall'
    ]
    if not selected:
        return []
    selected.sort(key=lambda row: embedding_model_sort_key(str(row.get('EmbeddingModel') or '')))
    models = [str(row.get('EmbeddingModel') or '') for row in selected]
    labels = [embedding_model_display_label(model) for model in models]
    positions = np.arange(len(selected), dtype=np.float64)

    figure, (absolute_axis, delta_axis) = plt.subplots(1, 2, figsize=(8.0, 4.6))
    try:
        width = 0.24
        objectives = (
            ('Top-k', 'MeanTopK', PASTEL_OBJECTIVE_COLORS['top_k']),
            ('MMR', 'MeanMMR', PASTEL_OBJECTIVE_COLORS['mmr']),
            ('Facility-Location', 'MeanFacLoc', PASTEL_OBJECTIVE_COLORS['fac_loc']),
        )
        for index, (label, column, color) in enumerate(objectives):
            offset = (index - 1) * width
            values = [float_or_none(row.get(column)) or 0.0 for row in selected]
            bars = absolute_axis.bar(
                positions + offset,
                values,
                width,
                label=label,
                color=color,
            )
            absolute_axis.bar_label(
                bars,
                fmt='%.3f',
                padding=2,
                fontsize=MODEL_VALUE_LABEL_SIZE,
                rotation=90,
            )
        absolute_axis.set_title(
            'Absolute retrieval performance',
            fontsize=MODEL_AXIS_TITLE_SIZE,
        )
        absolute_axis.set_ylabel('Mean FCP@k', fontsize=MODEL_AXIS_LABEL_SIZE)
        absolute_axis.set_xticks(positions, labels, rotation=28, ha='right')
        absolute_axis.tick_params(axis='both', labelsize=MODEL_TICK_LABEL_SIZE)
        absolute_axis.set_ylim(0, 1.02)
        absolute_axis.grid(axis='y', alpha=0.22)
        handles, legend_labels = absolute_axis.get_legend_handles_labels()
        figure.legend(
            handles,
            legend_labels,
            frameon=False,
            ncol=3,
            fontsize=MODEL_LEGEND_SIZE,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.995),
            columnspacing=1.0,
        )

        deltas = [float_or_none(row.get('MeanDeltaFacLocMMR')) or 0.0 for row in selected]
        bars = delta_axis.bar(
            positions,
            deltas,
            color=[PASTEL_EMBEDDING_MODEL_COLORS.get(model, '#A9A9A9') for model in models],
            width=0.72,
        )
        delta_axis.bar_label(bars, fmt='%+.3f', padding=3, fontsize=MODEL_VALUE_LABEL_SIZE)
        delta_axis.axhline(0.0, color='#555555', linewidth=0.9)
        delta_axis.axhline(0.05, color='#888888', linestyle=':', linewidth=1.0)
        delta_axis.set_title('Relative objective effect', fontsize=MODEL_AXIS_TITLE_SIZE)
        delta_axis.set_ylabel('FCP@k delta', fontsize=MODEL_AXIS_LABEL_SIZE)
        delta_axis.set_xticks(positions, labels, rotation=28, ha='right')
        delta_axis.tick_params(axis='both', labelsize=MODEL_TICK_LABEL_SIZE)
        delta_axis.grid(axis='y', alpha=0.22)
        top = max(deltas) + 0.025
        delta_axis.set_ylim(min(-0.01, min(deltas) - 0.015), top)

        figure.tight_layout(rect=(0, 0, 1, 0.90), pad=0.7)
        return _save_both(figure=figure, output_dir=output_dir, stem='embedding_model_fcp_overview')
    finally:
        plt.close(figure)


def _embedding_geometry_overview(
    *,
    plt: Any,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> list[Path]:
    if not rows:
        return []
    selected = sorted(
        rows,
        key=lambda row: embedding_model_sort_key(str(row.get('EmbeddingModel') or '')),
    )
    models = [str(row.get('EmbeddingModel') or '') for row in selected]
    labels = [embedding_model_display_label(model) for model in models]
    positions = np.arange(len(selected), dtype=np.float64)
    figure, (stress_axis, margin_axis) = plt.subplots(1, 2, figsize=(8.0, 4.6))
    try:
        stress = [float_or_none(row.get('CoverageStressRate')) or 0.0 for row in selected]
        bars = stress_axis.bar(
            positions,
            stress,
            width=0.40,
            color=PASTEL_GEOMETRY_COLORS['coverage_stress'],
        )
        stress_axis.bar_label(bars, fmt='%.3f', padding=3, fontsize=MODEL_VALUE_LABEL_SIZE)
        stress_axis.set_title('Early-ranking coverage stress', fontsize=MODEL_AXIS_TITLE_SIZE)
        stress_axis.set_ylabel('Coverage-stress rate', fontsize=MODEL_AXIS_LABEL_SIZE)
        stress_axis.set_xticks(positions, labels, rotation=28, ha='right')
        stress_axis.tick_params(axis='both', labelsize=MODEL_TICK_LABEL_SIZE)
        stress_axis.set_ylim(0, 1.05)
        stress_axis.grid(axis='y', alpha=0.22)

        width = 0.34
        series = (
            ('Gold - near miss', 'GoldNearMissMargin', PASTEL_GEOMETRY_COLORS['near_miss']),
            ('Gold - background', 'GoldBackgroundMargin', PASTEL_GEOMETRY_COLORS['background']),
        )
        for index, (label, column, color) in enumerate(series):
            values = [float_or_none(row.get(column)) or 0.0 for row in selected]
            margin_axis.bar(
                positions + (index - 0.5) * width,
                values,
                width,
                label=label,
                color=color,
            )
        margin_axis.set_title('Gold separation from distractors', fontsize=MODEL_AXIS_TITLE_SIZE)
        margin_axis.set_ylabel('Cosine margin', fontsize=MODEL_AXIS_LABEL_SIZE)
        margin_axis.set_xticks(positions, labels, rotation=28, ha='right')
        margin_axis.tick_params(axis='both', labelsize=MODEL_TICK_LABEL_SIZE)
        margin_axis.grid(axis='y', alpha=0.22)
        handles, legend_labels = margin_axis.get_legend_handles_labels()
        figure.legend(
            handles,
            legend_labels,
            frameon=False,
            ncol=2,
            fontsize=MODEL_LEGEND_SIZE,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.995),
            columnspacing=1.0,
        )

        figure.tight_layout(rect=(0, 0, 1, 0.90), pad=0.7)
        return _save_both(figure=figure, output_dir=output_dir, stem='embedding_geometry_overview')
    finally:
        plt.close(figure)


def _embedding_geometry_family_heatmap(
    *,
    plt: Any,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> list[Path]:
    if not rows:
        return []
    models = sorted(
        {str(row.get('EmbeddingModel') or '') for row in rows},
        key=embedding_model_sort_key,
    )
    families = sorted({str(row.get('ExperimentFamilyLabel') or '') for row in rows})
    values = np.full((len(models), len(families)), np.nan, dtype=np.float64)
    indexed = {
        (str(row.get('EmbeddingModel') or ''), str(row.get('ExperimentFamilyLabel') or '')): row
        for row in rows
    }
    for model_index, model in enumerate(models):
        for family_index, family in enumerate(families):
            row = indexed.get((model, family), {})
            value = float_or_none(row.get('CoverageStressRate'))
            if value is not None:
                values[model_index, family_index] = value
    figure, axis = plt.subplots(figsize=(7.5, 2.5))
    try:
        image = axis.imshow(
            values,
            vmin=0.5,
            vmax=1.0,
            cmap='viridis',
            aspect='auto',
        )
        axis.set_xticks(
            range(len(families)),
            [_heatmap_family_label(family) for family in families],
            rotation=25,
            ha='right',
        )
        axis.set_yticks(
            range(len(models)),
            [embedding_model_display_label(model) for model in models],
        )
        axis.tick_params(axis='both', labelsize=MODEL_TICK_LABEL_SIZE)
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                if np.isnan(value):
                    continue
                axis.text(
                    column_index,
                    row_index,
                    f'{value:.3f}',
                    ha='center',
                    va='center',
                    fontsize=MODEL_HEATMAP_CELL_SIZE,
                    color='white' if value < 0.74 else 'black',
                )
        colorbar = figure.colorbar(image, ax=axis)
        colorbar.set_label('Coverage stress', fontsize=MODEL_AXIS_LABEL_SIZE)
        colorbar.ax.tick_params(labelsize=MODEL_TICK_LABEL_SIZE)
        figure.tight_layout(pad=0.7)
        return _save_both(
            figure=figure,
            output_dir=output_dir,
            stem='embedding_geometry_family_heatmap',
        )
    finally:
        plt.close(figure)


def _lambda_fcp_delta_by_embedding_model(
    *,
    plt: Any,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> list[Path]:
    if not rows:
        return []
    models = sorted(
        {str(row.get('EmbeddingModel') or '') for row in rows},
        key=embedding_model_sort_key,
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 4.6), sharey=True)
    try:
        for axis, strategy, title in zip(
            axes,
            ('mmr', 'fac_loc'),
            ('MMR', 'Facility-Location'),
            strict=True,
        ):
            for model in models:
                model_rows = sorted(
                    [
                        row
                        for row in rows
                        if row.get('EmbeddingModel') == model and row.get('strategy') == strategy
                    ],
                    key=lambda row: float_or_none(row.get('lambda_norm')) or 0.0,
                )
                if not model_rows:
                    continue
                axis.plot(
                    [float_or_none(row.get('lambda_norm')) or 0.0 for row in model_rows],
                    [
                        float_or_none(row.get('MeanDeltaStrategyTopK_FCP')) or 0.0
                        for row in model_rows
                    ],
                    label=embedding_model_display_label(model),
                    color=embedding_model_color(model),
                    linewidth=2.8,
                )
            axis.axhline(0.0, color='#555555', linewidth=0.9)
            axis.set_title(title, fontsize=MODEL_AXIS_TITLE_SIZE)
            axis.set_xlabel('Normalized lambda-grid position', fontsize=MODEL_AXIS_LABEL_SIZE)
            axis.tick_params(axis='both', labelsize=MODEL_TICK_LABEL_SIZE)
            axis.grid(alpha=0.22)
        axes[0].set_ylabel(
            'Family-balanced validation FCP@k delta vs Top-k',
            fontsize=MODEL_AXIS_LABEL_SIZE,
        )
        axes[1].legend(frameon=False, fontsize=MODEL_LEGEND_SIZE)
        figure.suptitle(
            'Lambda sensitivity across representation spaces', fontsize=MODEL_FIGURE_TITLE_SIZE
        )
        figure.tight_layout(rect=(0, 0, 1, 0.94), pad=0.7)
        return _save_both(
            figure=figure,
            output_dir=output_dir,
            stem='lambda_fcp_delta_by_embedding_model',
        )
    finally:
        plt.close(figure)


def _save_both(*, figure: Any, output_dir: Path, stem: str) -> list[Path]:
    paths: list[Path] = []
    for suffix in ('png', 'pdf'):
        path = output_dir / f'{stem}.{suffix}'
        figure.savefig(path, dpi=180 if suffix == 'png' else None, bbox_inches='tight')
        paths.append(path)
    return paths


def _heatmap_family_label(label: str) -> str:
    """Use compact labels so every family remains legible at thesis width."""
    return {
        # These compact labels prevent collisions in the five-column matrix;
        # the title/caption retains the full distribution-family terminology.
        'Background variants': 'Background',
        'Balanced clean distributions': 'Balanced',
        'Dominance distributions': 'Dominance',
        'Near-miss-heavy distributions': 'Near-miss',
        'Sparse-niche distributions': 'Sparse-niche',
    }.get(label, label)
