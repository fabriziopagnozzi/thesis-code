from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import polars as pl

from experiments.medical_dataset_gen.reports.latex_tables import (
    THESIS_RESULT_MACROS_PATH,
    render_thesis_result_macros,
)

type ReportRow = dict[str, object]


def generate_exp_results_macros(
    *,
    report_dir: Path,
    output_path: Path = THESIS_RESULT_MACROS_PATH,
) -> Path:
    """Regenerate thesis result macros from an existing report's CSV artifacts."""
    data_dir = report_dir.expanduser().resolve() / 'data'
    if not data_dir.is_dir():
        raise FileNotFoundError(f'Report data directory not found: {data_dir}')

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_thesis_result_macros(
            geometry_rows=_read_rows(data_dir / 'geometry_filter_summary.csv'),
            comparison_rows=_read_rows(data_dir / 'comparison_by_k.csv'),
            budget_rows=_read_rows(data_dir / 'budget_strategy_summary.csv'),
            lambda_safety_rows=_read_rows(data_dir / 'lambda_safety_summary.csv'),
            metric_summary_rows=_read_rows(data_dir / 'metric_aggregate_summary.csv'),
            metric_family_summary_rows=_read_rows(data_dir / 'metric_family_summary.csv'),
            paired_suite_rows=_read_rows(
                data_dir / 'paired_suite_effect_summary.csv', required=False
            ),
            embedding_summary_rows=_read_rows(data_dir / 'embedding_model_summary.csv'),
            require_complete_wording_grid=True,
        )
    )
    return output_path


def _read_rows(path: Path, *, required: bool = True) -> list[ReportRow]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f'Required report artifact not found: {path}')
        return []
    if path.stat().st_size == 0:
        if required:
            raise ValueError(f'Required report artifact is empty: {path}')
        return []
    return cast(list[ReportRow], pl.read_csv(path).to_dicts())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Regenerate exp_results_macros.tex from an existing experiment report.'
    )
    parser.add_argument(
        '--report-dir',
        type=Path,
        required=True,
        help='Existing experiment_comparison report containing data/*.csv.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=THESIS_RESULT_MACROS_PATH,
        help=f'Output TeX file. Defaults to {THESIS_RESULT_MACROS_PATH}.',
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    output_path = generate_exp_results_macros(
        report_dir=cast(Path, args.report_dir),
        output_path=cast(Path, args.output),
    )
    print(f'wrote result macros to {output_path}')


if __name__ == '__main__':
    main()
