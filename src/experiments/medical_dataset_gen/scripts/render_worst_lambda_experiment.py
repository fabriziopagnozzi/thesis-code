"""Render lambda-sensitivity metrics for the worst validation-grid experiment.

The report CSV is used to identify the minimum strategy-minus-top-k FCP delta.
The plot itself is rendered from the corresponding validation statistics
parquet, so it uses the same source data as the report's lambda analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import polars as pl

matplotlib.use('Agg')

from experiments.medical_dataset_gen.evaluation.eval_plots import (
    plot_metrics_k_curves_for_lambda,
)

_DELTA_COLUMN = 'DeltaStrategyTopK_FCP'
_DEFAULT_STRATEGIES = ('fac_loc', 'mmr')
_REQUIRED_COLUMNS = (
    'Experiment',
    'Distribution',
    'RunLabel',
    'strategy',
    'k',
    'lam',
    'TopK_FCP',
    'Strategy_FCP',
    _DELTA_COLUMN,
    'GridStatsPath',
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find each strategy's lowest validation-grid FCP delta in an existing "
            'report and render metrics_k_curves_for_lambda for the implicated experiments.'
        )
    )
    parser.add_argument(
        '--report-dir',
        type=Path,
        default=Path('/home/fab/Projects/thesis-writing/reports/current'),
        help='Existing report directory containing data/lambda_grid_fcp_delta.csv.',
    )
    parser.add_argument(
        '--strategies',
        nargs='+',
        choices=_DEFAULT_STRATEGIES,
        default=list(_DEFAULT_STRATEGIES),
        help='Diversifying strategies whose minima determine the experiments to render.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help=(
            'Destination directory. Defaults to '
            '<report-dir>/figures/diagnostics/worst_lambda_experiment.'
        ),
    )
    return parser.parse_args()


def _worst_rows(csv_path: Path, strategies: list[str]) -> pl.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f'missing report CSV: {csv_path}')

    scan = pl.scan_csv(csv_path)
    available = set(scan.collect_schema().names())
    missing = sorted(set(_REQUIRED_COLUMNS) - available)
    if missing:
        raise ValueError(f'{csv_path} is missing required columns: {", ".join(missing)}')

    rows = scan.select(_REQUIRED_COLUMNS).filter(pl.col('strategy').is_in(strategies)).collect()
    absent = sorted(set(strategies) - set(rows['strategy'].unique().to_list()))
    if absent:
        raise ValueError(f'no lambda-grid rows for strategies: {", ".join(absent)}')

    return (
        rows.sort([_DELTA_COLUMN, 'Experiment', 'k', 'lam'])
        .group_by('strategy', maintain_order=True)
        .first()
        .sort('strategy')
    )


def _safe_path_component(value: object) -> str:
    text = str(value).strip()
    return ''.join(
        character if character.isalnum() or character in '-_.' else '_' for character in text
    )


def _float_value(value: object) -> float:
    if not isinstance(value, int | float | str):
        raise TypeError(f'expected a numeric value, got {value!r}')
    return float(value)


def _validate_delta(row: dict[str, object]) -> None:
    reported = _float_value(row[_DELTA_COLUMN])
    recomputed = _float_value(row['Strategy_FCP']) - _float_value(row['TopK_FCP'])
    if abs(reported - recomputed) > 1e-12:
        raise ValueError(
            f'{row["strategy"]} delta mismatch: report={reported}, recomputed={recomputed}'
        )


def main() -> None:
    args = _parse_args()
    report_dir = args.report_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else report_dir / 'figures' / 'diagnostics' / 'worst_lambda_experiment'
    )
    worst_rows = _worst_rows(
        report_dir / 'data' / 'lambda_grid_fcp_delta.csv',
        args.strategies,
    )

    rendered: dict[tuple[str, str], Path] = {}
    for row in worst_rows.iter_rows(named=True):
        _validate_delta(row)
        experiment = str(row['Experiment'])
        grid_stats_path = Path(str(row['GridStatsPath']))
        render_key = (experiment, str(grid_stats_path))
        if render_key not in rendered:
            if not grid_stats_path.is_file():
                raise FileNotFoundError(f'missing validation-grid parquet: {grid_stats_path}')
            experiment_dir = output_dir / (
                f'{_safe_path_component(row["Distribution"])}__'
                f'{_safe_path_component(row["RunLabel"])}'
            )
            experiment_dir.mkdir(parents=True, exist_ok=True)
            plot_metrics_k_curves_for_lambda(
                stats_df=pl.read_parquet(grid_stats_path),
                out_dir=experiment_dir,
                plot_data_split='validation',
            )
            plot_path = experiment_dir / 'metrics_k_curves_for_lambda.png'
            if not plot_path.is_file():
                raise RuntimeError(f'plot was not generated: {plot_path}')
            rendered[render_key] = plot_path

        print(
            f'{row["strategy"]}: experiment={experiment}, k={row["k"]}, '
            f'lambda={_float_value(row["lam"]):.6f}, '
            f'top_k_fcp={_float_value(row["TopK_FCP"]):.6f}, '
            f'strategy_fcp={_float_value(row["Strategy_FCP"]):.6f}, '
            f'delta={_float_value(row[_DELTA_COLUMN]):.6f}'
        )

    for path in rendered.values():
        print(path)


if __name__ == '__main__':
    main()
