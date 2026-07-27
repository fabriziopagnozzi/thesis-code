from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from experiments.medical_dataset_gen.reports.analysis_constants import TABLEFMT_OPTS
from experiments.medical_dataset_gen.reports.models import (
    CliArgs,
    MainQueryScope,
    PlotFormat,
    RefreshMode,
)
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
    refresh_group = parser.add_mutually_exclusive_group()
    refresh_group.add_argument(
        '--refresh-plots',
        type=Path,
        default=None,
        help='Regenerate only figures in an existing report directory from its data/*.csv artifacts.',
    )
    refresh_group.add_argument(
        '--refresh-latex-macros',
        type=Path,
        default=None,
        help=(
            'Regenerate only latex/exp_results_macros.tex in an existing report directory '
            'from its data/*.csv artifacts.'
        ),
    )
    refresh_group.add_argument(
        '--refresh-output-artifacts',
        dest='refresh_output_artifacts',
        type=Path,
        default=None,
        help='Deprecated alias for --refresh-plots.',
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
        '--experiment-regex',
        default=None,
        help='Optional Python regex applied to resolved completed experiment names.',
    )
    parser.add_argument(
        '--exclude-experiment-regex',
        default=None,
        help='Optional Python regex for resolved completed experiment names to exclude.',
    )
    parser.add_argument(
        '--artifact-version',
        default=None,
        help='Optional exact local artifact version to report, for example v4.',
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
        '--cross-query-chunk-modes',
        action='store_true',
        help=(
            'Enable cross-wording-configuration analyses across query_mode, focus_mode, '
            'and chunk_text_mode triples. This is intended for full all-mode reports.'
        ),
    )
    parser.add_argument(
        '--main-query-scope',
        choices=('all', 'geometry_eligible'),
        default='all',
        help=(
            'Population used for primary strategy/comparison summaries. '
            'geometry_eligible recomputes them from queries that pass the geometry filter.'
        ),
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
    refresh_report_dir = (
        parsed.refresh_plots
        or parsed.refresh_latex_macros
        or parsed.refresh_output_artifacts
    )
    refresh_mode: RefreshMode | None = None
    if parsed.refresh_latex_macros is not None:
        refresh_mode = 'latex_macros'
    elif parsed.refresh_plots is not None or parsed.refresh_output_artifacts is not None:
        refresh_mode = 'plots'
    if refresh_report_dir is not None and parsed.output_dir is not None:
        parser.error('--output-dir cannot be combined with refresh-only arguments')
    if refresh_mode == 'plots' and parsed.no_plots:
        parser.error('--no-plots cannot be combined with --refresh-plots')
    if parsed.experiment_regex is not None:
        try:
            re.compile(str(parsed.experiment_regex))
        except re.error as exc:
            parser.error(f'invalid --experiment-regex: {exc}')
    if parsed.exclude_experiment_regex is not None:
        try:
            re.compile(str(parsed.exclude_experiment_regex))
        except re.error as exc:
            parser.error(f'invalid --exclude-experiment-regex: {exc}')
    results_dir = parsed.results_dir.expanduser().resolve()
    refresh_report_dir = (
        refresh_report_dir.expanduser().resolve()
        if refresh_report_dir is not None
        else None
    )
    output_dir = (
        refresh_report_dir
        if refresh_report_dir is not None
        else (
            parsed.output_dir.expanduser().resolve()
            if parsed.output_dir is not None
            else results_dir / '_reports' / 'experiment_comparison'
        )
    )

    return CliArgs(
        results_dir=results_dir,
        output_dir=output_dir,
        include_scrapped=bool(parsed.include_scrapped),
        experiments=tuple(str(exp) for exp in parsed.experiments),
        experiment_regex=str(parsed.experiment_regex)
        if parsed.experiment_regex is not None
        else None,
        exclude_experiment_regex=(
            str(parsed.exclude_experiment_regex)
            if parsed.exclude_experiment_regex is not None
            else None
        ),
        artifact_version=(
            str(parsed.artifact_version).strip() if parsed.artifact_version is not None else None
        ),
        max_table_rows=max(1, int(parsed.max_table_rows)),
        tablefmt=str(parsed.tablefmt),
        plots=not bool(parsed.no_plots),
        plot_format=cast(PlotFormat, parsed.plot_format),
        near_optimal_epsilon=max(0.0, float(parsed.near_optimal_epsilon)),
        cross_query_chunk_modes=bool(parsed.cross_query_chunk_modes),
        refresh_report_dir=refresh_report_dir,
        refresh_mode=refresh_mode,
        bootstrap_replicates=max(100, int(parsed.bootstrap_replicates)),
        bootstrap_seed=int(parsed.bootstrap_seed),
        main_query_scope=cast(MainQueryScope, parsed.main_query_scope),
    )
