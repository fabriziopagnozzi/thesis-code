"""Compact figures for the embedding-model robustness analysis."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.medical_dataset_gen.reports.helpers import float_or_none, ordered_embedding_models
from experiments.medical_dataset_gen.reports.models import PlotFormat
from experiments.medical_dataset_gen.reports.report_config import embedding_model_color

type ReportRow = Mapping[str, object]


def write_embedding_model_figures(
    *,
    plt: Any,
    output_dir: Path,
    plot_format: PlotFormat,
    metric_rows: Sequence[ReportRow],
    geometry_model_rows: Sequence[ReportRow],
    geometry_family_rows: Sequence[ReportRow],
    lambda_model_rows: Sequence[ReportRow],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.extend(
        _plot_model_fcp_overview(
            plt=plt,
            rows=metric_rows,
            output_dir=output_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        _plot_geometry_by_model(
            plt=plt,
            rows=geometry_model_rows,
            output_dir=output_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        _plot_geometry_family_heatmap(
            plt=plt,
            rows=geometry_family_rows,
            output_dir=output_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        _plot_lambda_by_model(
            plt=plt,
            rows=lambda_model_rows,
            output_dir=output_dir,
            plot_format=plot_format,
        )
    )
    return paths


def _plot_model_fcp_overview(
    *,
    plt: Any,
    rows: Sequence[ReportRow],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    selected = [
        row for row in rows if row.get('MetricLabel') == 'FCP' and row.get('Scope') == 'overall'
    ]
    if not selected:
        return []
    labels = [str(row.get('EmbeddingModelLabel') or '') for row in selected]
    positions = list(range(len(selected)))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    try:
        width = 0.23
        for offset, field, label, color in (
            (-width, 'MeanTopK', 'Top-k', '#777777'),
            (0.0, 'MeanMMR', 'MMR', '#CC8A35'),
            (width, 'MeanFacLoc', 'Facility-Location', '#287C8E'),
        ):
            axes[0].bar(
                [position + offset for position in positions],
                [_number(row.get(field)) for row in selected],
                width=width,
                label=label,
                color=color,
            )
        axes[0].set_ylabel('Family- and budget-balanced mean FCP')
        axes[0].set_ylim(0.0, 1.0)
        axes[0].legend(frameon=False, ncol=3, fontsize=8)
        axes[0].grid(axis='y', alpha=0.2)

        colors = [embedding_model_color(str(row.get('EmbeddingModel') or '')) for row in selected]
        deltas = [_number(row.get('MeanDeltaFacLocMMR')) for row in selected]
        axes[1].bar(positions, deltas, color=colors)
        axes[1].axhline(0.0, color='#555555', linewidth=0.8)
        axes[1].axhline(0.05, color='#888888', linewidth=0.7, linestyle=':')
        axes[1].set_ylabel('Mean FCP difference (Facility-Location - MMR)')
        axes[1].grid(axis='y', alpha=0.2)
        for position, value in zip(positions, deltas, strict=True):
            axes[1].text(position, value, f'{value:+.3f}', ha='center', va='bottom', fontsize=8)

        for axis in axes:
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, rotation=20, ha='right')
        axes[0].set_title('Absolute retrieval performance')
        axes[1].set_title('Relative objective effect')
        figure.suptitle('Retrieval results across representation spaces')
        figure.tight_layout()
        path = output_dir / f'embedding_model_fcp_overview.{plot_format}'
        figure.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(figure)


def _plot_geometry_by_model(
    *,
    plt: Any,
    rows: Sequence[ReportRow],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    if not rows:
        return []
    labels = [str(row.get('EmbeddingModelLabel') or '') for row in rows]
    positions = list(range(len(rows)))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    try:
        width = 0.2
        for index, (field, label, color) in enumerate(
            (
                ('GeometryPassRate', 'Composite', '#287C8E'),
                ('FacetCompletenessPassRate', 'Facet complete', '#5F8F3F'),
                ('PrimaryAxisStressPassRate', 'Primary-axis stress', '#C47A3A'),
                ('EarlyFacetCoverageStressPassRate', 'Early-coverage stress', '#6F63A6'),
            )
        ):
            offset = (index - 1.5) * width
            axes[0].bar(
                [position + offset for position in positions],
                [_number(row.get(field)) for row in rows],
                width=width,
                label=label,
                color=color,
            )
        axes[0].set_ylim(0.0, 1.05)
        axes[0].set_ylabel('Family-balanced pass rate')
        axes[0].legend(frameon=False, fontsize=7.5, ncol=2)
        axes[0].grid(axis='y', alpha=0.2)

        margin_fields = (
            ('GoldMinusNearMissSimilarityMarginMean', 'Gold - near miss', '#CC8A35'),
            ('GoldMinusBackgroundOutlierSimilarityMarginMean', 'Gold - background', '#3B6FB6'),
        )
        for index, (field, label, color) in enumerate(margin_fields):
            offset = (index - 0.5) * 0.34
            axes[1].bar(
                [position + offset for position in positions],
                [_number(row.get(field)) for row in rows],
                width=0.34,
                label=label,
                color=color,
            )
        axes[1].axhline(0.0, color='#555555', linewidth=0.8)
        axes[1].set_ylabel('Within-model cosine-similarity margin')
        axes[1].legend(frameon=False, fontsize=8)
        axes[1].grid(axis='y', alpha=0.2)
        for axis in axes:
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, rotation=20, ha='right')
        axes[0].set_title('Categorical audit criteria')
        axes[1].set_title('Gold separation from distractors')
        figure.suptitle('Representation-space audit by embedding model')
        figure.tight_layout()
        path = output_dir / f'embedding_geometry_overview.{plot_format}'
        figure.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(figure)


def _plot_geometry_family_heatmap(
    *,
    plt: Any,
    rows: Sequence[ReportRow],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    if not rows:
        return []
    models = _ordered_unique(str(row.get('EmbeddingModelLabel') or '') for row in rows)
    families = _ordered_unique(str(row.get('ExperimentFamilyLabel') or '') for row in rows)
    lookup = {
        (
            str(row.get('EmbeddingModelLabel') or ''),
            str(row.get('ExperimentFamilyLabel') or ''),
        ): float_or_none(row.get('GeometryPassRate'))
        for row in rows
    }
    matrix = [[lookup.get((model, family)) for family in families] for model in models]
    figure, axis = plt.subplots(figsize=(9.8, 4.5))
    try:
        masked = [[float('nan') if value is None else value for value in row] for row in matrix]
        image = axis.imshow(masked, vmin=0.5, vmax=1.0, cmap='viridis', aspect='auto')
        axis.set_xticks(range(len(families)))
        axis.set_xticklabels(families, rotation=25, ha='right')
        axis.set_yticks(range(len(models)))
        axis.set_yticklabels(models)
        for y, values in enumerate(matrix):
            for x, value in enumerate(values):
                if value is not None:
                    axis.text(
                        x,
                        y,
                        f'{value:.3f}',
                        ha='center',
                        va='center',
                        fontsize=8,
                        color='white' if value < 0.72 else 'black',
                    )
        axis.set_title('Composite geometry-audit pass rate by model and family')
        colorbar = figure.colorbar(image, ax=axis, shrink=0.82)
        colorbar.set_label('Mean pass rate')
        figure.tight_layout()
        path = output_dir / f'embedding_geometry_family_heatmap.{plot_format}'
        figure.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(figure)


def _plot_lambda_by_model(
    *,
    plt: Any,
    rows: Sequence[ReportRow],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    if not rows:
        return []
    models = ordered_embedding_models(str(row.get('EmbeddingModel') or '') for row in rows)
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True)
    try:
        for axis, strategy, title in zip(
            axes,
            ('mmr', 'fac_loc'),
            ('MMR', 'Facility-Location'),
            strict=True,
        ):
            for model in models:
                selected = sorted(
                    [
                        row
                        for row in rows
                        if row.get('EmbeddingModel') == model and row.get('strategy') == strategy
                    ],
                    key=lambda row: _number(row.get('lambda_norm')),
                )
                if not selected:
                    continue
                axis.plot(
                    [_number(row.get('lambda_norm')) for row in selected],
                    [_number(row.get('MeanDeltaStrategyTopK_FCP')) for row in selected],
                    label=str(selected[0].get('EmbeddingModelLabel') or model),
                    color=embedding_model_color(model),
                    linewidth=2.0,
                )
            axis.axhline(0.0, color='#555555', linewidth=0.8)
            axis.set_title(title)
            axis.set_xlabel('Normalized lambda-grid position')
            axis.grid(alpha=0.2)
        axes[0].set_ylabel('Family-balanced validation FCP delta vs top-k')
        axes[1].legend(frameon=False, fontsize=8)
        figure.suptitle('Lambda sensitivity across representation spaces')
        figure.tight_layout()
        path = output_dir / f'lambda_fcp_delta_by_embedding_model.{plot_format}'
        figure.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(figure)


def _number(value: object) -> float:
    return float_or_none(value) or 0.0


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
