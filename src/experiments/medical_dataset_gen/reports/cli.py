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
from helpers.dir_paths import THESIS_REPORTS_DIR


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
    suite_group = parser.add_mutually_exclusive_group()
    suite_group.add_argument(
        '--suite',
        default=None,
        help='Materialized v5 suite ID. Uses its manifest instead of legacy directory discovery.',
    )
    suite_group.add_argument(
        '--suite-base',
        default=None,
        help=(
            'Native suite ID whose experiment_specs-derived suites should be discovered and '
            'combined into one report.'
        ),
    )
    parser.add_argument(
        '--where',
        default=None,
        help='Suite-only comma-separated equality filters, for example family_id=dominance,scale=medium.',
    )
    parser.add_argument(
        '--suite-regex',
        default=None,
        help='Regex applied to derived suite IDs discovered by --suite-base.',
    )
    parser.add_argument(
        '--strict-suite',
        action='store_true',
        help=(
            'Reject incomplete declared suite contrasts. Without this flag, partial/smoke reports '
            'omit incomplete strict contrasts while retaining the completed-cell summaries.'
        ),
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help=(
            'Directory where report files are written. Defaults to '
            '<thesis-documents>/reports/experiment_comparison.'
        ),
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
        '--embedding-models',
        nargs='+',
        default=(),
        help=(
            'Optional exact embedding model names to include, for example '
            'BAAI/bge-m3 Qwen/Qwen3-Embedding-0.6B. Defaults to all discovered models.'
        ),
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
            'and chunk_text_mode triples, including the singleton label-only query mode. '
            'This is intended for full all-mode reports.'
        ),
        default=True,
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
    parser.add_argument(
        '--lambda-analysis',
        action='store_true',
        help='Generate lambda-grid, stability, safety, and near-optimal-width diagnostics.',
        default=True,
    )
    parser.add_argument(
        '--global-lambda-analysis',
        action='store_true',
        help='Generate global-lambda transfer validity outputs.',
    )
    parser.add_argument(
        '--lodo-analysis',
        action='store_true',
        help='Generate leave-one-distribution-out lambda-transfer outputs.',
    )
    parser.add_argument(
        '--paired-statistics',
        action='store_true',
        help='Generate paired query/profile datasets, bootstrap summaries, and statistical tables.',
    )
    parser.add_argument(
        '--validity-analysis',
        action='store_true',
        help='Generate geometry-population and synthetic-artifact validity diagnostics.',
        default=True,
    )
    parser.add_argument(
        '--full-report',
        action='store_true',
        help='Enable every optional expensive analysis.',
    )
    parsed = parser.parse_args(argv)
    suite_selected = parsed.suite is not None or parsed.suite_base is not None
    if suite_selected:
        incompatible = [
            option
            for option, value in (
                ('--experiments', parsed.experiments),
                ('--experiment-regex', parsed.experiment_regex),
                ('--exclude-experiment-regex', parsed.exclude_experiment_regex),
                ('--artifact-version', parsed.artifact_version),
                ('--include-scrapped', parsed.include_scrapped),
            )
            if value
        ]
        if incompatible:
            parser.error(
                '--suite/--suite-base use manifest selection and cannot combine with '
                + ', '.join(incompatible)
            )
    elif parsed.where is not None or parsed.strict_suite or parsed.suite_regex is not None:
        parser.error('--where, --strict-suite, and --suite-regex require --suite or --suite-base')
    if parsed.suite_regex is not None and parsed.suite_base is None:
        parser.error('--suite-regex requires --suite-base')
    if parsed.suite_regex is not None:
        try:
            re.compile(str(parsed.suite_regex))
        except re.error as exc:
            parser.error(f'invalid --suite-regex: {exc}')
    refresh_report_dir = (
        parsed.refresh_plots or parsed.refresh_latex_macros or parsed.refresh_output_artifacts
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
    configured_results_dir = parsed.results_dir.expanduser()
    default_output_dir = THESIS_REPORTS_DIR / 'experiment_comparison'
    results_dir = configured_results_dir.resolve()
    refresh_report_dir = (
        refresh_report_dir.expanduser().resolve() if refresh_report_dir is not None else None
    )
    output_dir = (
        refresh_report_dir
        if refresh_report_dir is not None
        else (
            parsed.output_dir.expanduser().resolve()
            if parsed.output_dir is not None
            else default_output_dir.resolve()
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
        embedding_models=tuple(_normalized_embedding_models(parsed.embedding_models)),
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
        lambda_analysis=bool(parsed.lambda_analysis),
        global_lambda_analysis=bool(parsed.global_lambda_analysis),
        lodo_analysis=bool(parsed.lodo_analysis),
        paired_statistics=bool(parsed.paired_statistics),
        validity_analysis=bool(parsed.validity_analysis),
        full_report=bool(parsed.full_report),
        suite_id=str(parsed.suite).strip() if parsed.suite is not None else None,
        suite_base_id=str(parsed.suite_base).strip() if parsed.suite_base is not None else None,
        suite_regex=str(parsed.suite_regex).strip() if parsed.suite_regex is not None else None,
        suite_where=str(parsed.where).strip() if parsed.where is not None else None,
        strict_suite=bool(parsed.strict_suite),
    )


def _normalized_embedding_models(raw_values: Sequence[object]) -> list[str]:
    models: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        for part in str(raw_value).split(','):
            model = part.strip()
            if not model or model in seen:
                continue
            seen.add(model)
            models.append(model)
    return models
