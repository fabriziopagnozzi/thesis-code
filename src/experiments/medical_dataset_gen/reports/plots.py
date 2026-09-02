from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl

from experiments.medical_dataset_gen.evaluation.eval_plots import plot_metrics_k_curves_for_lambda
from experiments.medical_dataset_gen.reports.models import ExperimentRecord, PlotFormat
from experiments.medical_dataset_gen.reports.plot_aggregates import (
    plot_budget_delta_columns,
    plot_fcp_family_budget_heatmaps,
    plot_fcp_family_budget_heatmaps_by_embedding_model,
    plot_metric_budget_outcomes,
    plot_metric_family_delta_heatmap_by_embedding_model,
    plot_metric_family_delta_heatmap_low_budget,
    plot_metric_family_delta_heatmap_low_budget_best_embedding_model,
)
from experiments.medical_dataset_gen.reports.plot_config_differences import (
    plot_config_fcp_budget_delta_heatmaps,
    plot_config_fcp_budget_delta_heatmaps_by_distribution,
    plot_config_fcp_family_budget_delta_heatmaps,
    plot_config_fcp_family_budget_delta_heatmaps_by_embedding_model,
    plot_config_metric_delta_heatmap_low_budget,
    plot_config_metric_delta_heatmap_low_budget_by_distribution_embedding_model,
    plot_config_metric_delta_heatmap_low_budget_by_embedding_model,
    plot_config_metric_family_delta_heatmap_low_budget,
    plot_config_metric_family_delta_heatmap_low_budget_by_embedding_model,
)
from experiments.medical_dataset_gen.reports.plot_diagnostics import (
    plot_dataset_composition,
    plot_geometry_coverage_stress_rate,
    plot_lambda_delta_curve,
    plot_lambda_safety_worst_delta,
    plot_lambda_stability,
    plot_near_optimal_width,
)
from experiments.medical_dataset_gen.reports.plot_models import write_embedding_model_figures
from experiments.medical_dataset_gen.reports.plot_statistical import (
    plot_paired_fcp_config_embedding_forest,
    plot_paired_fcp_config_forest,
    plot_paired_fcp_forest,
)
from experiments.medical_dataset_gen.reports.report_config import (
    BUDGET_CATEGORIES,
    LEGACY_LOW_BUDGET_TOKEN,
    REPORT_METRIC_SPECS,
)


def write_figures(
    *,
    output_dir: Path,
    plot_format: PlotFormat,
    max_rows: int,
    budget_rows: Sequence[Mapping[str, object]],
    geometry_rows: Sequence[Mapping[str, object]],
    lambda_rows: Sequence[Mapping[str, object]],
    lambda_grid_delta_rows: Sequence[Mapping[str, object]],
    lambda_safety_rows: Sequence[Mapping[str, object]],
    near_optimal_rows: Sequence[Mapping[str, object]],
    dataset_rows: Sequence[Mapping[str, object]],
    metric_summary_rows: Sequence[Mapping[str, object]],
    metric_family_summary_rows: Sequence[Mapping[str, object]],
    metric_family_budget_summary_rows: Sequence[Mapping[str, object]],
    paired_cell_rows: Sequence[Mapping[str, object]],
    paired_suite_rows: Sequence[Mapping[str, object]],
    paired_config_suite_rows: Sequence[Mapping[str, object]],
    cross_query_chunk_modes: bool,
    warnings: list[str],
    lambda_curve_rows: Sequence[Mapping[str, object]] = (),
    lambda_reference_record: ExperimentRecord | None = None,
    embedding_metric_rows: Sequence[Mapping[str, object]] = (),
    embedding_geometry_rows: Sequence[Mapping[str, object]] = (),
    embedding_geometry_family_rows: Sequence[Mapping[str, object]] = (),
    lambda_embedding_rows: Sequence[Mapping[str, object]] = (),
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
    except Exception as exc:
        warnings.append(f'plotting skipped because matplotlib could not be imported ({exc})')
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / 'metrics'
    aggregate_dir = output_dir / 'aggregates'
    cross_config_dir = aggregate_dir / 'cross_config'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(cross_config_dir, ignore_errors=True)
    if cross_query_chunk_modes:
        cross_config_dir.mkdir(parents=True, exist_ok=True)
    for obsolete_stem in (
        'fcp_delta_by_experiment',
        'facloc_vs_topk_delta_by_experiment',
        f'{LEGACY_LOW_BUDGET_TOKEN}_fcp_deltas_by_experiment',
        'all_facet_clean_rate_by_experiment',
        f'query_scope_{LEGACY_LOW_BUDGET_TOKEN}_delta_shift_by_experiment',
    ):
        (output_dir / f'{obsolete_stem}.{plot_format}').unlink(missing_ok=True)
    for obsolete_path in output_dir.glob(f'*_deltas_by_experiment.{plot_format}'):
        obsolete_path.unlink(missing_ok=True)
    for obsolete_path in metrics_dir.glob(
        f'*_{LEGACY_LOW_BUDGET_TOKEN}_deltas_by_experiment.{plot_format}'
    ):
        obsolete_path.unlink(missing_ok=True)
    for obsolete_stem in ('aggregate_metric_family_delta_heatmap',):
        (aggregate_dir / f'{obsolete_stem}.{plot_format}').unlink(missing_ok=True)
    paths: list[Path] = []
    paths.extend(
        write_embedding_model_figures(
            plt=plt,
            output_dir=aggregate_dir / 'embedding_models',
            metric_rows=embedding_metric_rows,
            geometry_rows=embedding_geometry_rows,
            geometry_family_rows=embedding_geometry_family_rows,
            lambda_rows=lambda_embedding_rows,
        )
    )

    paths.extend(
        plot_metric_budget_outcomes(
            plt=plt,
            rows=metric_summary_rows,
            output_dir=aggregate_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        plot_metric_family_delta_heatmap_low_budget(
            plt=plt,
            rows=metric_family_budget_summary_rows,
            output_dir=aggregate_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        plot_metric_family_delta_heatmap_by_embedding_model(
            plt=plt,
            rows=budget_rows,
            output_dir=aggregate_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        plot_metric_family_delta_heatmap_low_budget_best_embedding_model(
            plt=plt,
            rows=budget_rows,
            output_dir=aggregate_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        plot_fcp_family_budget_heatmaps(
            plt=plt,
            rows=metric_family_budget_summary_rows,
            output_dir=aggregate_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        plot_fcp_family_budget_heatmaps_by_embedding_model(
            plt=plt,
            rows=budget_rows,
            output_dir=aggregate_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        plot_paired_fcp_forest(
            plt=plt,
            cell_rows=paired_cell_rows,
            suite_rows=paired_suite_rows,
            output_dir=aggregate_dir,
            plot_format=plot_format,
        )
    )
    if cross_query_chunk_modes:
        paths.extend(
            plot_config_fcp_budget_delta_heatmaps(
                plt=plt,
                rows=budget_rows,
                output_dir=cross_config_dir,
                plot_format=plot_format,
            )
        )
        paths.extend(
            plot_config_metric_delta_heatmap_low_budget(
                plt=plt,
                rows=budget_rows,
                output_dir=cross_config_dir,
                plot_format=plot_format,
            )
        )
        paths.extend(
            plot_config_metric_delta_heatmap_low_budget_by_embedding_model(
                plt=plt,
                rows=budget_rows,
                output_dir=cross_config_dir,
                plot_format=plot_format,
            )
        )
        paths.extend(
            plot_config_metric_delta_heatmap_low_budget_by_distribution_embedding_model(
                plt=plt,
                rows=budget_rows,
                output_dir=cross_config_dir,
                plot_format=plot_format,
            )
        )
        paths.extend(
            plot_config_fcp_budget_delta_heatmaps_by_distribution(
                plt=plt,
                rows=budget_rows,
                output_dir=cross_config_dir,
                plot_format=plot_format,
            )
        )
        paths.extend(
            plot_config_fcp_family_budget_delta_heatmaps(
                plt=plt,
                rows=budget_rows,
                output_dir=cross_config_dir,
                plot_format=plot_format,
            )
        )
        paths.extend(
            plot_config_fcp_family_budget_delta_heatmaps_by_embedding_model(
                plt=plt,
                rows=budget_rows,
                output_dir=cross_config_dir,
                plot_format=plot_format,
            )
        )
        paths.extend(
            plot_config_metric_family_delta_heatmap_low_budget(
                plt=plt,
                rows=budget_rows,
                output_dir=cross_config_dir,
                plot_format=plot_format,
            )
        )
        paths.extend(
            plot_config_metric_family_delta_heatmap_low_budget_by_embedding_model(
                plt=plt,
                rows=budget_rows,
                output_dir=cross_config_dir,
                plot_format=plot_format,
            )
        )
        paths.extend(
            plot_paired_fcp_config_forest(
                plt=plt,
                rows=paired_config_suite_rows,
                output_dir=cross_config_dir,
                plot_format=plot_format,
            )
        )
        paths.extend(
            plot_paired_fcp_config_embedding_forest(
                plt=plt,
                rows=paired_config_suite_rows,
                output_dir=cross_config_dir,
                plot_format=plot_format,
            )
        )

    for category in BUDGET_CATEGORIES:
        category_rows = [row for row in budget_rows if row.get('BudgetCategory') == category]
        for spec in REPORT_METRIC_SPECS:
            paths.extend(
                plot_budget_delta_columns(
                    plt=plt,
                    rows=category_rows,
                    category=category,
                    spec=spec,
                    output_dir=metrics_dir,
                    plot_format=plot_format,
                )
            )
    paths.extend(
        plot_geometry_coverage_stress_rate(
            plt=plt,
            rows=geometry_rows,
            output_dir=output_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        plot_lambda_stability(
            plt=plt,
            rows=lambda_rows,
            output_dir=output_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        plot_lambda_safety_worst_delta(
            plt=plt,
            rows=lambda_safety_rows,
            output_dir=output_dir,
            plot_format=plot_format,
        )
    )
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
        plot_near_optimal_width(
            plt=plt,
            rows=near_optimal_rows,
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
    if lambda_reference_record is not None:
        paths.extend(
            _plot_reference_lambda_metrics(
                record=lambda_reference_record,
                output_dir=output_dir,
                warnings=warnings,
            )
        )
    return paths


def _plot_reference_lambda_metrics(
    *,
    record: ExperimentRecord,
    output_dir: Path,
    warnings: list[str],
) -> list[Path]:
    grid_path = record.paths.table_path('evaluation_selection_stats')
    if not grid_path.is_file():
        warnings.append(f'{record.name}: reference lambda plot skipped; validation grid missing')
        return []
    try:
        stats = pl.read_parquet(grid_path)
        reference_dir = output_dir / 'lambda_reference'
        reference_dir.mkdir(parents=True, exist_ok=True)
        plot_metrics_k_curves_for_lambda(
            stats_df=stats,
            out_dir=reference_dir,
            plot_data_split='validation',
        )
    except Exception as exc:
        warnings.append(f'{record.name}: reference lambda plot failed ({exc})')
        return []
    path = reference_dir / 'metrics_k_curves_for_lambda.png'
    return [path] if path.is_file() else []
