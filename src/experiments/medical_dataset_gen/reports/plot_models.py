"""Focused thesis figures for representation-model comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.medical_dataset_gen.reports.helpers import embedding_model_sort_key, float_or_none
from experiments.medical_dataset_gen.reports.report_config import (
    OBJECTIVE_COLORS,
    embedding_model_color,
    embedding_model_display_label,
)


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
    figure, (absolute_axis, delta_axis) = plt.subplots(1, 2, figsize=(13.8, 5.5))
    try:
        width = 0.24
        objectives = (
            ('Top-k', 'MeanTopK', OBJECTIVE_COLORS['top_k']),
            ('MMR', 'MeanMMR', OBJECTIVE_COLORS['mmr']),
            ('Facility-Location', 'MeanFacLoc', OBJECTIVE_COLORS['fac_loc']),
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
            absolute_axis.bar_label(bars, fmt='%.3f', padding=2, fontsize=7.5, rotation=90)
        absolute_axis.set_title('Absolute retrieval performance')
        absolute_axis.set_ylabel('Family- and budget-balanced mean FCP@k')
        absolute_axis.set_xticks(positions, labels, rotation=20, ha='right')
        absolute_axis.set_ylim(0, 1.02)
        absolute_axis.grid(axis='y', alpha=0.22)
        absolute_axis.legend(frameon=False, ncol=3, fontsize=8.5)

        deltas = [float_or_none(row.get('MeanDeltaFacLocMMR')) or 0.0 for row in selected]
        bars = delta_axis.bar(
            positions,
            deltas,
            color=[embedding_model_color(model) for model in models],
            width=0.72,
        )
        delta_axis.bar_label(bars, fmt='%+.3f', padding=3, fontsize=9)
        delta_axis.axhline(0.0, color='#555555', linewidth=0.9)
        delta_axis.axhline(0.05, color='#888888', linestyle=':', linewidth=1.0)
        delta_axis.set_title('Relative objective effect')
        delta_axis.set_ylabel('Mean FCP@k difference (FacLoc - MMR)')
        delta_axis.set_xticks(positions, labels, rotation=20, ha='right')
        delta_axis.grid(axis='y', alpha=0.22)
        top = max(deltas) + 0.025
        delta_axis.set_ylim(min(-0.01, min(deltas) - 0.015), top)

        figure.suptitle('Retrieval results across representation spaces', fontsize=16)
        figure.tight_layout()
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
    figure, (stress_axis, margin_axis) = plt.subplots(1, 2, figsize=(13.8, 5.64))
    try:
        stress = [float_or_none(row.get('CoverageStressRate')) or 0.0 for row in selected]
        bars = stress_axis.bar(positions, stress, width=0.58, color='#287C8E')
        stress_axis.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
        stress_axis.set_title('Early-ranking coverage stress')
        stress_axis.set_ylabel('Family-balanced coverage-stress rate')
        stress_axis.set_xticks(positions, labels, rotation=20, ha='right')
        stress_axis.set_ylim(0, 1.05)
        stress_axis.grid(axis='y', alpha=0.22)

        width = 0.34
        series = (
            ('Gold - near miss', 'GoldNearMissMargin', '#D08A2E'),
            ('Gold - background', 'GoldBackgroundMargin', '#3D71B7'),
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
        margin_axis.set_title('Gold separation from distractors')
        margin_axis.set_ylabel('Within-model cosine-similarity margin')
        margin_axis.set_xticks(positions, labels, rotation=20, ha='right')
        margin_axis.grid(axis='y', alpha=0.22)
        margin_axis.legend(frameon=False)

        figure.suptitle('Representation-space audit by embedding model', fontsize=16)
        figure.tight_layout()
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
    figure, axis = plt.subplots(figsize=(10.5, 5.3))
    try:
        image = axis.imshow(values, vmin=0.5, vmax=1.0, cmap='viridis', aspect='auto')
        axis.set_title('Coverage-stress rate by embedding model and distribution family')
        axis.set_xticks(range(len(families)), families, rotation=27, ha='right')
        axis.set_yticks(
            range(len(models)),
            [embedding_model_display_label(model) for model in models],
        )
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
                    color='white' if value < 0.74 else 'black',
                )
        colorbar = figure.colorbar(image, ax=axis)
        colorbar.set_label('Mean coverage-stress rate')
        figure.tight_layout()
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
    figure, axes = plt.subplots(1, 2, figsize=(14.4, 5.76), sharey=True)
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
                    linewidth=2.2,
                )
            axis.axhline(0.0, color='#555555', linewidth=0.9)
            axis.set_title(title)
            axis.set_xlabel('Normalized lambda-grid position')
            axis.grid(alpha=0.22)
        axes[0].set_ylabel('Family-balanced validation FCP@k delta vs Top-k')
        axes[1].legend(frameon=False, fontsize=8.5)
        figure.suptitle('Lambda sensitivity across representation spaces', fontsize=16)
        figure.tight_layout()
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
