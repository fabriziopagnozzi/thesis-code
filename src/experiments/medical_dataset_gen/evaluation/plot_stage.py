import argparse
import inspect
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Literal, cast

import polars as pl

from experiments.medical_dataset_gen.evaluation.eval_plot_data import EvaluationResultLookup
from experiments.medical_dataset_gen.evaluation.eval_plots_configs import (
    ANSWER_ROUGE_EVAL_PLOT_FILE_NAMES,
    DEFAULT_ENABLED_EVAL_PLOT_NAMES,
    EVAL_PLOT_FILE_NAMES,
    EvalPlotCallContext,
    EvalPlotFileName,
)
from experiments.medical_dataset_gen.evaluation.statistics import stats_for_evaluation_mode
from experiments.medical_dataset_gen.utils.cli_parsing import parse_comma_separated_names
from experiments.medical_dataset_gen.utils.global_schemas import (
    ExperimentCfg,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
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
type EvaluationPlotPopulation = Literal['all', 'passing', 'non_passing']
_EVALUATION_PLOT_POPULATIONS: tuple[EvaluationPlotPopulation, ...] = (
    'all',
    'passing',
    'non_passing',
)
_EVALUATION_PLOT_POPULATION_DIRS: dict[EvaluationPlotPopulation, str] = {
    'all': 'all',
    'passing': 'passing',
    'non_passing': 'non-passing',
}


def run_eval_plots(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    selected_plots: set[EvalPlotFileName] | None = None,
) -> None:

    results_path = paths.table_path('evaluation_results')
    if not results_path.exists():
        print('Skipping eval figures: evaluation_results not found')
        return

    results_df = read_parquet(paths, 'evaluation_results')
    if results_df.is_empty():
        print('Skipping eval figures: evaluation_results is empty')
        return

    results_df = _ensure_geometry_pass_column(results_df, paths)
    rendered_dirs: list[Path] = []
    for population in _EVALUATION_PLOT_POPULATIONS:
        population_results = _filter_population_results(results_df, population)
        if population_results.is_empty():
            print(f'[plots] skipping {population}: no evaluation rows')
            continue

        stats_df, validation_grid_stats_df, _report_grid_stats_df = stats_for_evaluation_mode(
            population_results,
            mode=cfg.evaluation.mode,
            cfg=cfg,
        )
        if stats_df.is_empty():
            print(f'[plots] skipping {population}: no aggregate evaluation stats')
            continue

        validation_results_df = population_results
        plot_results_df = population_results
        if cfg.evaluation.mode == 'testing':
            validation_results_df = population_results.filter(pl.col('split') == 'validation')
            plot_results_df = population_results.filter(pl.col('split') == 'test')
            if plot_results_df.is_empty():
                print(f'[plots] skipping {population}: no test rows in evaluation_results')
                continue
            if validation_grid_stats_df.is_empty():
                print(f'[plots] skipping {population}: no validation rows for lambda curves')
                continue

        out_dir = paths.figures_dir / 'evaluation' / _EVALUATION_PLOT_POPULATION_DIRS[population]
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_jobs = build_plot_jobs(
            cfg=cfg,
            stats_df=stats_df,
            validation_grid_stats_df=validation_grid_stats_df,
            results_df=plot_results_df,
            validation_results_df=validation_results_df,
            result_lookup=EvaluationResultLookup(plot_results_df),
            validation_result_lookup=EvaluationResultLookup(validation_results_df),
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
        rendered_dirs.append(out_dir)

        selection_note = f' ({", ".join(selected_job_names)})' if selected_plots is not None else ''
        print(f'[plots] saved {population} evaluation figures to {out_dir}{selection_note}')

    if not rendered_dirs:
        print('Skipping eval figures: no population produced figures')


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
    result_lookup: EvaluationResultLookup,
    validation_result_lookup: EvaluationResultLookup,
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
                    result_lookup=result_lookup,
                    validation_result_lookup=validation_result_lookup,
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
    result_lookup: EvaluationResultLookup,
    validation_result_lookup: EvaluationResultLookup,
    out_dir: Path,
) -> EvalPlotCallContext:
    uses_validation_grid = (
        cfg.evaluation.mode == 'testing' and plot_name in _VALIDATION_GRID_PLOT_NAMES
    )
    effective_stats_df = validation_grid_stats_df if uses_validation_grid else stats_df
    effective_results_df = validation_results_df if uses_validation_grid else results_df
    effective_result_lookup = validation_result_lookup if uses_validation_grid else result_lookup
    return {
        'stats_df': effective_stats_df,
        'results_df': effective_results_df,
        'result_lookup': effective_result_lookup,
        'out_dir': out_dir,
        'lambda_selection': cfg.evaluation.lambda_selection,
        'plot_theme': cfg.evaluation.plot_theme,
        'plot_data_split': 'validation' if uses_validation_grid else 'test',
    }


def _ensure_geometry_pass_column(
    results_df: pl.DataFrame,
    paths: MedicalDatasetGenPaths,
) -> pl.DataFrame:
    if 'passes_geometry_filter' in results_df.columns:
        return results_df
    if 'passes_filter' in results_df.columns:
        return results_df.rename({'passes_filter': 'passes_geometry_filter'})

    geometry_path = paths.table_path('geometry_stats')
    if not geometry_path.is_file() or 'query_id' not in results_df.columns:
        return results_df.with_columns(pl.lit(True).alias('passes_geometry_filter'))

    geometry = read_parquet(paths, 'geometry_stats').select(
        'query_id',
        pl.col('passes_filter').fill_null(False).alias('passes_geometry_filter'),
    )
    return results_df.join(geometry, on='query_id', how='left').with_columns(
        pl.col('passes_geometry_filter').fill_null(False)
    )


def _filter_population_results(
    results_df: pl.DataFrame,
    population: EvaluationPlotPopulation,
) -> pl.DataFrame:
    if population == 'all':
        return results_df
    pass_value = population == 'passing'
    return results_df.filter(pl.col('passes_geometry_filter').fill_null(False) == pass_value)


def parse_plot_names(raw_value: str | None) -> set[EvalPlotFileName] | None:
    parsed = parse_comma_separated_names(
        raw_value=raw_value,
        valid_names=EVAL_PLOT_FILE_NAMES,
        option_name='--plots',
    )
    if parsed is None:
        return None
    return cast(set[EvalPlotFileName], set(parsed))


def parse_plots_cli_args(argv: list[str]) -> set[EvalPlotFileName] | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--plots',
        type=str,
        help='Comma-separated plot names to generate selectively.',
    )
    args, remaining_argv = parser.parse_known_args(argv)
    if remaining_argv:
        parser.error(f'unknown evaluation-plot argument(s): {" ".join(remaining_argv)}')
    return parse_plot_names(args.plots)
