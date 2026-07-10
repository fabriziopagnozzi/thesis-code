from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from experiments.medical_dataset_gen.analysis.models import PlotFormat
from experiments.medical_dataset_gen.analysis.plot_aggregates import (
    _plot_budget_delta_columns,
    _plot_fcp_family_budget_heatmaps,
    _plot_metric_budget_outcomes,
    _plot_metric_family_delta_heatmap,
)
from experiments.medical_dataset_gen.analysis.plot_diagnostics import (
    _plot_dataset_composition,
    _plot_geometry_pass_rate,
    _plot_lambda_delta_curve,
    _plot_lambda_safety_worst_delta,
    _plot_lambda_stability,
    _plot_near_optimal_width,
)
from experiments.medical_dataset_gen.analysis.report_config import (
    BUDGET_CATEGORIES,
    DELTA_METRIC_PLOT_SPECS,
    LEGACY_LOW_BUDGET_TOKEN,
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
    warnings: list[str],
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
    metrics_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir.mkdir(parents=True, exist_ok=True)
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
    paths: list[Path] = []

    paths.extend(
        _plot_metric_budget_outcomes(
            plt=plt,
            rows=metric_summary_rows,
            output_dir=aggregate_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        _plot_metric_family_delta_heatmap(
            plt=plt,
            rows=metric_family_summary_rows,
            output_dir=aggregate_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        _plot_fcp_family_budget_heatmaps(
            plt=plt,
            rows=metric_family_budget_summary_rows,
            output_dir=aggregate_dir,
            plot_format=plot_format,
        )
    )

    for category in BUDGET_CATEGORIES:
        category_rows = [row for row in budget_rows if row.get('BudgetCategory') == category]
        for spec in DELTA_METRIC_PLOT_SPECS:
            paths.extend(
                _plot_budget_delta_columns(
                    plt=plt,
                    rows=category_rows,
                    category=category,
                    spec=spec,
                    output_dir=metrics_dir,
                    plot_format=plot_format,
                )
            )
    paths.extend(
        _plot_geometry_pass_rate(
            plt=plt,
            rows=geometry_rows,
            output_dir=output_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        _plot_lambda_stability(
            plt=plt,
            rows=lambda_rows,
            output_dir=output_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        _plot_lambda_safety_worst_delta(
            plt=plt,
            rows=lambda_safety_rows,
            output_dir=output_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        _plot_lambda_delta_curve(
            plt=plt,
            rows=lambda_grid_delta_rows,
            output_dir=output_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        _plot_near_optimal_width(
            plt=plt,
            rows=near_optimal_rows,
            output_dir=output_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        _plot_dataset_composition(
            plt=plt,
            rows=dataset_rows,
            output_dir=output_dir,
            plot_format=plot_format,
        )
    )
    return paths
