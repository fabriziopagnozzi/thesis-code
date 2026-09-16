from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from experiments.medical_dataset_gen.reports.models import PlotFormat
from experiments.medical_dataset_gen.reports.plot_diagnostics import (
    plot_dataset_composition,
    plot_lambda_delta_curve,
)
from experiments.medical_dataset_gen.reports.plot_models import write_embedding_model_figures


def write_figures(
    *,
    output_dir: Path,
    plot_format: PlotFormat,
    dataset_rows: Sequence[Mapping[str, object]],
    warnings: list[str],
    lambda_grid_delta_rows: Sequence[Mapping[str, object]] = (),
    lambda_curve_rows: Sequence[Mapping[str, object]] = (),
    embedding_metric_rows: Sequence[Mapping[str, object]] = (),
    embedding_geometry_rows: Sequence[Mapping[str, object]] = (),
    embedding_geometry_family_rows: Sequence[Mapping[str, object]] = (),
    lambda_embedding_rows: Sequence[Mapping[str, object]] = (),
) -> list[Path]:
    """Render only figures consumed by the thesis sources."""
    try:
        import matplotlib

        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
    except Exception as exc:
        warnings.append(f'plotting skipped because matplotlib could not be imported ({exc})')
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if any(
        (
            embedding_metric_rows,
            embedding_geometry_rows,
            embedding_geometry_family_rows,
            lambda_embedding_rows,
        )
    ):
        paths.extend(
            write_embedding_model_figures(
                plt=plt,
                output_dir=output_dir / 'aggregates' / 'embedding_models',
                metric_rows=embedding_metric_rows,
                geometry_rows=embedding_geometry_rows,
                geometry_family_rows=embedding_geometry_family_rows,
                lambda_rows=lambda_embedding_rows,
            )
        )
    if lambda_grid_delta_rows or lambda_curve_rows:
        paths.extend(
            plot_lambda_delta_curve(
                plt=plt,
                rows=lambda_grid_delta_rows,
                curve_rows=lambda_curve_rows,
                output_dir=output_dir,
                plot_format=plot_format,
            )
        )
    paths.extend(
        plot_dataset_composition(
            plt=plt,
            rows=dataset_rows,
            output_dir=output_dir,
            plot_format=plot_format,
        )
    )
    return paths
