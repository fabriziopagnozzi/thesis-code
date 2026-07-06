import argparse
import inspect
import sys
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import cast

import polars as pl

from experiments.medical_dataset_gen.evaluation.eval_plots_configs import (
    ANSWER_ROUGE_EVAL_PLOT_FILE_NAMES,
    DEFAULT_ENABLED_EVAL_PLOT_NAMES,
    EVAL_PLOT_FILE_NAMES,
    EvalPlotCallContext,
    EvalPlotFileName,
)
from experiments.medical_dataset_gen.schemas.global_config_schemas import (
    ExperimentCfg,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    load_config_from_cli,
    paths_for,
    setup_logging,
)
from experiments.medical_dataset_gen.utils.io_utils import read_parquet

_VALIDATION_GRID_PLOT_NAMES: set[EvalPlotFileName] = {
    'metrics_k_curves_for_lambda',
    'metrics_heatmap_k_lambda_grid',
    'metrics_heatmap_k_lambda_grid_html',
    'metrics_delta_vs_topk_k_curves_for_lambda',
    'answer_metrics_k_curves_for_lambda',
    'diagnostics_k_curves_for_lambda',
}


def run_eval_plots(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    selected_plots: set[EvalPlotFileName] | None = None,
) -> None:

    stats_path = paths.table_path('evaluation_stats')
    results_path = paths.table_path('evaluation_results')
    if not stats_path.exists() or not results_path.exists():
        print('Skipping eval figures: evaluation_stats or evaluation_results not found')
        return

    stats_df = read_parquet(paths, 'evaluation_stats')
    results_df = read_parquet(paths, 'evaluation_results')
    if stats_df.is_empty() or results_df.is_empty():
        print('Skipping eval figures: evaluation tables are empty')
        return
    validation_grid_stats_df = stats_df
    validation_results_df = results_df
    if cfg.evaluation.mode == 'testing':
        validation_results_df = results_df.filter(pl.col('split') == 'validation')
        results_df = results_df.filter(pl.col('split') == 'test')
        selection_stats_path = paths.table_path('evaluation_selection_stats')
        if selection_stats_path.exists():
            validation_grid_stats_df = read_parquet(paths, 'evaluation_selection_stats')
        if results_df.is_empty():
            print('Skipping eval figures: no test rows in evaluation_results')
            return

    out_dir = paths.figures_dir / 'evaluation'
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_jobs = build_plot_jobs(
        cfg=cfg,
        stats_df=stats_df,
        validation_grid_stats_df=validation_grid_stats_df,
        results_df=results_df,
        validation_results_df=validation_results_df,
        out_dir=out_dir,
    )
    selected_job_names = (
        [name for name, _ in plot_jobs if name in selected_plots]
        if selected_plots is not None
        else [name for name, _ in plot_jobs]
    )

    for plot_name, plot_callable in plot_jobs:
        if selected_plots is not None and plot_name not in selected_plots:
            continue
        plot_callable()

    selection_note = f' ({", ".join(selected_job_names)})' if selected_plots is not None else ''
    print(f'[plots] saved evaluation figures to {out_dir}{selection_note}')


def build_plot_callable(
    plot_name: EvalPlotFileName,
    plot_context: EvalPlotCallContext,
) -> Callable[[], None]:
    plot_fn_name = f'plot_{plot_name}'
    plot_module = import_module('experiments.medical_dataset_gen.evaluation.eval_plots')
    plot_fn = getattr(plot_module, plot_fn_name, None)
    if plot_fn is None or not callable(plot_fn):
        raise ValueError(f'Missing plot function: {plot_fn_name}')

    signature = inspect.signature(plot_fn)
    missing_params = [
        name
        for name, param in signature.parameters.items()
        if name not in plot_context and param.default is inspect.Parameter.empty
    ]
    if missing_params:
        missing = ', '.join(missing_params)
        raise ValueError(
            f'Unsupported plot function signature for {plot_fn_name}: missing {missing}'
        )

    ordered_kwargs = {
        name: plot_context[name] for name in signature.parameters if name in plot_context
    }
    typed_plot_fn = cast(Callable[..., None], plot_fn)
    return lambda: typed_plot_fn(**ordered_kwargs)


def build_plot_jobs(
    cfg: ExperimentCfg,
    stats_df: pl.DataFrame,
    validation_grid_stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
    validation_results_df: pl.DataFrame,
    out_dir: Path,
) -> list[tuple[EvalPlotFileName, Callable[[], None]]]:
    available_plot_names: list[EvalPlotFileName] = [
        plot_name
        for plot_name in DEFAULT_ENABLED_EVAL_PLOT_NAMES
        if cfg.retrieval.compute_answer_rouge or plot_name not in ANSWER_ROUGE_EVAL_PLOT_FILE_NAMES
    ]
    return [
        (
            plot_name,
            build_plot_callable(
                plot_name,
                _plot_context_for_name(
                    plot_name=plot_name,
                    cfg=cfg,
                    stats_df=stats_df,
                    validation_grid_stats_df=validation_grid_stats_df,
                    results_df=results_df,
                    validation_results_df=validation_results_df,
                    out_dir=out_dir,
                ),
            ),
        )
        for plot_name in available_plot_names
    ]


def _plot_context_for_name(
    *,
    plot_name: EvalPlotFileName,
    cfg: ExperimentCfg,
    stats_df: pl.DataFrame,
    validation_grid_stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
    validation_results_df: pl.DataFrame,
    out_dir: Path,
) -> EvalPlotCallContext:
    uses_validation_grid = (
        cfg.evaluation.mode == 'testing' and plot_name in _VALIDATION_GRID_PLOT_NAMES
    )
    effective_stats_df = (
        validation_grid_stats_df
        if uses_validation_grid
        else stats_df
    )
    effective_results_df = validation_results_df if uses_validation_grid else results_df
    return {
        'stats_df': effective_stats_df,
        'results_df': effective_results_df,
        'out_dir': out_dir,
        'lambda_selection': cfg.evaluation.lambda_selection,
        'plot_theme': cfg.evaluation.plot_theme,
        'plot_data_split': 'validation' if uses_validation_grid else 'test',
    }


def parse_plot_names(raw_value: str | None) -> set[EvalPlotFileName] | None:
    if raw_value is None:
        return None

    plot_names: set[str] = {part.strip() for part in raw_value.split(',') if part.strip()}
    if not plot_names:
        raise ValueError('--plots was provided but no plot names were specified')

    if plot_names:
        unknown_plots = sorted(plot_names - EVAL_PLOT_FILE_NAMES)
        if unknown_plots:
            available = ', '.join(sorted(EVAL_PLOT_FILE_NAMES))
            unknown = ', '.join(unknown_plots)
            raise ValueError(f'Unknown plot name(s): {unknown}. Available plots: {available}')

    return cast(set[EvalPlotFileName] | None, plot_names)


def parse_plots_cli_args(argv: list[str]) -> tuple[ExperimentCfg, set[EvalPlotFileName] | None]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--plots',
        type=str,
        help='Comma-separated plot names to generate selectively.',
    )
    args, remaining_argv = parser.parse_known_args(argv)

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *remaining_argv]
        cfg = load_config_from_cli()
    finally:
        sys.argv = original_argv
    return cfg, parse_plot_names(args.plots)


if __name__ == '__main__':
    cli_argv = sys.argv[1:]
    cfg, selected_plots = parse_plots_cli_args(cli_argv)
    paths = paths_for(cfg)
    setup_logging(paths)
    run_eval_plots(cfg, paths, selected_plots=selected_plots)
