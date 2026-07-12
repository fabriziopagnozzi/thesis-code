from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from experiments.medical_dataset_gen.analysis.analysis_constants import TABLEFMT_OPTS
from experiments.medical_dataset_gen.analysis.models import CliArgs, PlotFormat
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths


def parse_args(argv: Sequence[str] | None = None) -> CliArgs:
    default_results_dir = MedicalDatasetGenPaths.results_dir
    parser = argparse.ArgumentParser(
        description='Discover completed medical dataset experiments and compare retrieval results.'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=default_results_dir,
        help='Root directory containing experiment result folders.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory where report files are written. Defaults to <results-dir>/_reports/experiment_comparison.',
    )
    parser.add_argument(
        '--include-scrapped',
        action='store_true',
        help='Include experiments under 00_scrapped.',
    )
    parser.add_argument(
        '--experiments',
        nargs='*',
        default=(),
        help='Optional experiment names or prefixes. Parent names include completed child runs.',
    )
    parser.add_argument(
        '--max-table-rows',
        type=int,
        default=100,
        help='Maximum rows rendered per markdown table.',
    )
    parser.add_argument(
        '--tablefmt',
        type=str,
        choices=TABLEFMT_OPTS,
        default='grid',
        help='tabulate table format used in markdown reports.',
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip matplotlib figure generation.',
    )
    parser.add_argument(
        '--plot-format',
        choices=('png', 'pdf', 'svg'),
        default='png',
        help='Matplotlib figure file format.',
    )
    parser.add_argument(
        '--near-optimal-epsilon',
        type=float,
        default=0.01,
        help='A lambda is near-optimal when FCP is within this absolute margin of the best FCP.',
    )
    parser.add_argument(
        '--bootstrap-replicates',
        type=int,
        default=1000,
        help='Number of deterministic profile-cluster bootstrap replicates for paired inference.',
    )
    parser.add_argument(
        '--bootstrap-seed',
        type=int,
        default=20260712,
        help='Random seed for deterministic paired-inference bootstrap resampling.',
    )
    parsed = parser.parse_args(argv)
    results_dir = parsed.results_dir.expanduser().resolve()
    output_dir = (
        parsed.output_dir.expanduser().resolve()
        if parsed.output_dir is not None
        else results_dir / '_reports' / 'experiment_comparison'
    )

    return CliArgs(
        results_dir=results_dir,
        output_dir=output_dir,
        include_scrapped=bool(parsed.include_scrapped),
        experiments=tuple(str(exp) for exp in parsed.experiments),
        max_table_rows=max(1, int(parsed.max_table_rows)),
        tablefmt=str(parsed.tablefmt),
        plots=not bool(parsed.no_plots),
        plot_format=cast(PlotFormat, parsed.plot_format),
        near_optimal_epsilon=max(0.0, float(parsed.near_optimal_epsilon)),
        bootstrap_replicates=max(100, int(parsed.bootstrap_replicates)),
        bootstrap_seed=int(parsed.bootstrap_seed),
    )
