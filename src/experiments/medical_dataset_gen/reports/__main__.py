from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from experiments.medical_dataset_gen.evaluation.lambda_selection import (
    LAMBDA_SELECTION_MAXIMIZING_METRIC,
)
from experiments.medical_dataset_gen.reports.analysis_constants import REPORT_FILES
from experiments.medical_dataset_gen.reports.artifacts import (
    render_experiment_config_recap,
    write_csv,
)
from experiments.medical_dataset_gen.reports.cli import parse_args
from experiments.medical_dataset_gen.reports.discovery import (
    discover_experiments,
    load_experiment_record,
)
from experiments.medical_dataset_gen.reports.latex_tables import (
    render_thesis_aggregate_tables,
    render_thesis_result_macros,
    thesis_aggregate_tables_path,
    thesis_latex_dir,
    thesis_result_macros_path,
    thesis_statistical_tables_path,
)
from experiments.medical_dataset_gen.reports.models import CliArgs, ExperimentRecord, ReportOutputs
from experiments.medical_dataset_gen.reports.plots import write_figures
from experiments.medical_dataset_gen.reports.rendering import (
    render_interesting_findings,
    render_report,
)
from experiments.medical_dataset_gen.reports.report_config import LEGACY_LOW_BUDGET_TOKEN
from experiments.medical_dataset_gen.reports.rows import (
    dataset_distribution_row,
    experiment_manifest_row,
    geometry_filter_row,
    lambda_grid_fcp_delta_rows,
    lambda_safety_summary_rows,
    near_optimal_lambda_rows,
    selected_strategy_rows,
)
from experiments.medical_dataset_gen.reports.statistical import (
    cell_effect_summary_rows,
    configuration_suite_effect_summary_rows,
    leave_one_out_sensitivity_rows,
    render_statistical_latex_table,
    suite_effect_summary_rows,
    write_paired_effect_datasets,
)
from experiments.medical_dataset_gen.reports.summaries import (
    budget_category_rows_from_comparisons,
    comparison_by_k_rows,
    embedding_model_summary_rows,
    experiment_family_budget_summary_rows,
    experiment_family_summary_rows,
    lambda_stability_rows,
    metric_aggregate_summary_rows,
    metric_family_budget_summary_rows,
    metric_family_summary_rows,
)
from experiments.medical_dataset_gen.reports.validity import (
    geometry_population_strategy_rows,
    global_lambda_strategy_rows,
    lodo_lambda_strategy_rows,
    synthetic_artifact_diagnostic_rows,
)
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths


def run_report(args: CliArgs) -> ReportOutputs:
    if args.refresh_report_dir is not None:
        if args.refresh_mode == 'plots':
            return refresh_report_plots(args)
        if args.refresh_mode == 'latex_macros':
            return refresh_latex_macros(args)
        raise ValueError('refresh_report_dir requires a refresh mode')

    old_results_dir = MedicalDatasetGenPaths.results_dir
    MedicalDatasetGenPaths.results_dir = args.results_dir
    try:
        _progress(f'preparing output directory: {args.output_dir}')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = args.output_dir / 'data'
        shutil.rmtree(data_dir, ignore_errors=True)
        shutil.rmtree(args.output_dir / '_figures', ignore_errors=True)
        if not args.cross_query_chunk_modes:
            shutil.rmtree(
                args.output_dir / 'figures' / 'aggregates' / 'cross_config', ignore_errors=True
            )
        _remove_obsolete_flat_data_files(args.output_dir)
        warnings: list[str] = []
        _progress(f'discovering completed experiments under: {args.results_dir}')
        records = discover_experiments(
            args.results_dir,
            include_scrapped=args.include_scrapped,
            requested_experiments=args.experiments,
            experiment_regex=args.experiment_regex,
            exclude_experiment_regex=args.exclude_experiment_regex,
            artifact_version=args.artifact_version,
            warnings=warnings,
        )
        _progress(f'discovered {len(records)} completed experiments')
        plot_and_recap_records = records

        _progress('loading manifest, dataset, and geometry rows')
        manifest_rows = [experiment_manifest_row(record) for record in records]
        dataset_rows = [dataset_distribution_row(record, warnings=warnings) for record in records]
        geometry_rows = [geometry_filter_row(record, warnings=warnings) for record in records]

        _progress('computing geometry-population validity summaries')
        geometry_population_strategy_rows_data = geometry_population_strategy_rows(
            plot_and_recap_records,
            warnings=warnings,
        )
        geometry_population_comparison_rows = comparison_by_k_rows(
            geometry_population_strategy_rows_data
        )

        _progress('loading selected strategy rows and near-optimal lambda summaries')
        strategy_rows: list[dict[str, object]] = []
        near_optimal_rows: list[dict[str, object]] = []
        for record in records:
            strategy_rows.extend(selected_strategy_rows(record, warnings=warnings))
            near_optimal_rows.extend(
                near_optimal_lambda_rows(
                    record,
                    epsilon=args.near_optimal_epsilon,
                    warnings=warnings,
                )
            )
        if args.main_query_scope == 'geometry_eligible':
            strategy_rows = [
                dict(row)
                for row in geometry_population_strategy_rows_data
                if row.get('GeometryPopulation') == 'geometry_eligible'
            ]

        _progress('computing lambda-grid and main comparison summaries')
        lambda_grid_delta_rows = lambda_grid_fcp_delta_rows(records, warnings=warnings)
        lambda_safety_rows = lambda_safety_summary_rows(lambda_grid_delta_rows)
        comparison_rows = comparison_by_k_rows(strategy_rows)
        _progress('computing global-lambda validity summaries')
        global_lambda_strategy_rows_data = global_lambda_strategy_rows(records, warnings=warnings)
        global_lambda_comparison_rows = comparison_by_k_rows(global_lambda_strategy_rows_data)
        global_lambda_budget_rows = budget_category_rows_from_comparisons(
            global_lambda_comparison_rows
        )
        global_lambda_metric_summary_rows = metric_aggregate_summary_rows(
            comparison_rows=global_lambda_comparison_rows,
            budget_rows=global_lambda_budget_rows,
        )
        _progress('computing leave-one-distribution-out lambda validity summaries')
        lodo_lambda_strategy_rows_data = lodo_lambda_strategy_rows(records, warnings=warnings)
        lodo_lambda_comparison_rows = comparison_by_k_rows(lodo_lambda_strategy_rows_data)
        lodo_lambda_budget_rows = budget_category_rows_from_comparisons(lodo_lambda_comparison_rows)
        lodo_lambda_metric_summary_rows = metric_aggregate_summary_rows(
            comparison_rows=lodo_lambda_comparison_rows,
            budget_rows=lodo_lambda_budget_rows,
        )
        _progress('computing synthetic-artifact diagnostics')
        synthetic_artifact_diagnostic_rows_data = synthetic_artifact_diagnostic_rows(
            records,
            warnings=warnings,
        )
        _progress('computing aggregate family, budget, metric, and embedding summaries')
        family_summary_rows = experiment_family_summary_rows(comparison_rows)
        budget_rows = budget_category_rows_from_comparisons(comparison_rows)
        family_budget_summary_rows = experiment_family_budget_summary_rows(budget_rows)
        metric_family_summary_rows_data = metric_family_summary_rows(comparison_rows)
        metric_family_budget_summary_rows_data = metric_family_budget_summary_rows(budget_rows)
        metric_summary_rows = metric_aggregate_summary_rows(
            comparison_rows=comparison_rows,
            budget_rows=budget_rows,
        )
        low_budget_rows = [row for row in budget_rows if row.get('BudgetCategory') == 'low_budget']
        lambda_rows = lambda_stability_rows(strategy_rows, near_optimal_rows)
        embedding_summary_rows = embedding_model_summary_rows(
            manifest_rows=manifest_rows,
            geometry_rows=geometry_rows,
            low_budget_rows=low_budget_rows,
        )
        _progress('writing paired query/profile effect datasets')
        paired_profile_effects = write_paired_effect_datasets(
            records=records,
            strategy_rows=strategy_rows,
            output_dir=data_dir,
            warnings=warnings,
        )
        _progress('computing paired cell bootstrap summaries')
        paired_cell_rows = cell_effect_summary_rows(
            profile_effects=paired_profile_effects,
            budget_rows=budget_rows,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        _progress('computing paired suite bootstrap summaries')
        paired_suite_rows = suite_effect_summary_rows(
            profile_effects=paired_profile_effects,
            budget_rows=budget_rows,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        wording_configurations = _wording_configurations_for_rows(budget_rows)
        cross_triplet_analysis_enabled = (
            args.cross_query_chunk_modes and len(wording_configurations) > 1
        )
        if args.cross_query_chunk_modes:
            _progress('computing paired cross-triplet bootstrap summaries')
            paired_config_suite_rows = configuration_suite_effect_summary_rows(
                profile_effects=paired_profile_effects,
                budget_rows=budget_rows,
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.bootstrap_seed,
            )
        else:
            _progress('skipping cross-triplet summaries; pass --cross-query-chunk-modes to enable')
            paired_config_suite_rows = []
        paired_sensitivity_rows = leave_one_out_sensitivity_rows(
            profile_effects=paired_profile_effects,
            budget_rows=budget_rows,
        )
        # The partitioned Parquet artifacts are now complete and all summary
        # rows have been derived. Free the suite-sized profile frame before
        # matplotlib builds the report figures.
        del paired_profile_effects

        _progress('writing CSV report artifacts')
        (args.output_dir / f'{LEGACY_LOW_BUDGET_TOKEN}_strategy_summary.csv').unlink(
            missing_ok=True
        )
        write_csv(data_dir / 'experiment_manifest.csv', manifest_rows)
        write_csv(data_dir / 'dataset_distribution.csv', dataset_rows)
        write_csv(data_dir / 'geometry_filter_summary.csv', geometry_rows)
        write_csv(data_dir / 'strategy_by_k.csv', strategy_rows)
        write_csv(data_dir / 'comparison_by_k.csv', comparison_rows)
        write_csv(
            data_dir / 'geometry_population_strategy_by_k.csv',
            geometry_population_strategy_rows_data,
        )
        write_csv(
            data_dir / 'geometry_population_comparison_by_k.csv',
            geometry_population_comparison_rows,
        )
        write_csv(data_dir / 'global_lambda_strategy_by_k.csv', global_lambda_strategy_rows_data)
        write_csv(data_dir / 'global_lambda_comparison_by_k.csv', global_lambda_comparison_rows)
        write_csv(
            data_dir / 'global_lambda_metric_aggregate_summary.csv',
            global_lambda_metric_summary_rows,
        )
        write_csv(data_dir / 'lodo_lambda_strategy_by_k.csv', lodo_lambda_strategy_rows_data)
        write_csv(data_dir / 'lodo_lambda_comparison_by_k.csv', lodo_lambda_comparison_rows)
        write_csv(
            data_dir / 'lodo_lambda_metric_aggregate_summary.csv',
            lodo_lambda_metric_summary_rows,
        )
        write_csv(
            data_dir / 'synthetic_artifact_diagnostics.csv',
            synthetic_artifact_diagnostic_rows_data,
        )
        write_csv(data_dir / 'experiment_family_summary.csv', family_summary_rows)
        write_csv(
            data_dir / 'experiment_family_budget_summary.csv',
            family_budget_summary_rows,
        )
        write_csv(data_dir / 'metric_family_summary.csv', metric_family_summary_rows_data)
        write_csv(
            data_dir / 'metric_family_budget_summary.csv',
            metric_family_budget_summary_rows_data,
        )
        write_csv(data_dir / 'metric_aggregate_summary.csv', metric_summary_rows)
        write_csv(data_dir / 'budget_strategy_summary.csv', budget_rows)
        write_csv(data_dir / 'low_budget_strategy_summary.csv', low_budget_rows)
        write_csv(data_dir / 'lambda_stability.csv', lambda_rows)
        write_csv(data_dir / 'lambda_grid_fcp_delta.csv', lambda_grid_delta_rows)
        write_csv(data_dir / 'lambda_safety_summary.csv', lambda_safety_rows)
        write_csv(data_dir / 'near_optimal_lambda_width.csv', near_optimal_rows)
        write_csv(data_dir / 'embedding_model_summary.csv', embedding_summary_rows)
        write_csv(data_dir / 'paired_cell_effect_summary.csv', paired_cell_rows)
        write_csv(data_dir / 'paired_suite_effect_summary.csv', paired_suite_rows)
        write_csv(data_dir / 'paired_config_suite_effect_summary.csv', paired_config_suite_rows)
        write_csv(data_dir / 'paired_leave_one_out_sensitivity.csv', paired_sensitivity_rows)
        _progress('writing report LaTeX tables and macros')
        _write_thesis_outputs_from_rows(
            geometry_rows=geometry_rows,
            comparison_rows=comparison_rows,
            budget_rows=budget_rows,
            lambda_safety_rows=lambda_safety_rows,
            metric_summary_rows=metric_summary_rows,
            metric_family_summary_rows=metric_family_summary_rows_data,
            metric_family_budget_summary_rows=metric_family_budget_summary_rows_data,
            paired_suite_rows=paired_suite_rows,
            embedding_summary_rows=embedding_summary_rows,
            require_complete_wording_grid=False,
            output_dir=args.output_dir,
        )

        figures: list[Path] = []
        if args.plots:
            _progress('rendering report figures')
            figures = write_figures(
                output_dir=args.output_dir / 'figures',
                plot_format=args.plot_format,
                max_rows=args.max_table_rows,
                budget_rows=budget_rows,
                geometry_rows=geometry_rows,
                lambda_rows=lambda_rows,
                lambda_grid_delta_rows=lambda_grid_delta_rows,
                lambda_safety_rows=lambda_safety_rows,
                near_optimal_rows=near_optimal_rows,
                dataset_rows=dataset_rows,
                metric_summary_rows=metric_summary_rows,
                metric_family_summary_rows=metric_family_summary_rows_data,
                metric_family_budget_summary_rows=metric_family_budget_summary_rows_data,
                paired_cell_rows=paired_cell_rows,
                paired_suite_rows=paired_suite_rows,
                paired_config_suite_rows=paired_config_suite_rows,
                cross_query_chunk_modes=args.cross_query_chunk_modes,
                warnings=warnings,
            )
            _progress(f'rendered {len(figures)} figures')
        else:
            _progress('skipping figure rendering because --no-plots is set')

        _progress('rendering markdown reports and manifest')
        warnings = _dedupe_warnings(warnings)
        report_text = render_report(
            args=args,
            experiment_count=len(records),
            dataset_rows=dataset_rows,
            geometry_rows=geometry_rows,
            comparison_rows=comparison_rows,
            family_summary_rows=family_summary_rows,
            family_budget_summary_rows=family_budget_summary_rows,
            metric_family_summary_rows=metric_family_summary_rows_data,
            metric_family_budget_summary_rows=metric_family_budget_summary_rows_data,
            metric_summary_rows=metric_summary_rows,
            low_budget_rows=low_budget_rows,
            lambda_rows=lambda_rows,
            lambda_safety_rows=lambda_safety_rows,
            embedding_summary_rows=embedding_summary_rows,
            paired_config_suite_rows=paired_config_suite_rows,
            figures=figures,
        )
        (args.output_dir / 'txt_report.md').write_text(report_text)
        (args.output_dir / 'txt_report_highlights.md').write_text(
            render_interesting_findings(
                comparison_rows=comparison_rows,
                low_budget_rows=low_budget_rows,
                family_summary_rows=family_summary_rows,
                family_budget_summary_rows=family_budget_summary_rows,
                metric_family_summary_rows=metric_family_summary_rows_data,
                metric_family_budget_summary_rows=metric_family_budget_summary_rows_data,
                metric_summary_rows=metric_summary_rows,
                geometry_rows=geometry_rows,
                lambda_rows=lambda_rows,
                lambda_safety_rows=lambda_safety_rows,
                embedding_summary_rows=embedding_summary_rows,
                tablefmt=args.tablefmt,
                max_table_rows=args.max_table_rows,
            )
        )
        (args.output_dir / 'txt_experiments_config_recap.md').write_text(
            render_experiment_config_recap(plot_and_recap_records)
        )
        (args.output_dir / 'warnings.txt').write_text(
            '\n'.join(warnings) + ('\n' if warnings else '')
        )
        (args.output_dir / 'manifest.json').write_text(
            json.dumps(
                {
                    'generated_at_utc': datetime.now(UTC).isoformat(),
                    'results_dir': str(args.results_dir),
                    'output_dir': str(args.output_dir),
                    'include_scrapped': args.include_scrapped,
                    'requested_experiments': list(args.experiments),
                    'experiment_regex': args.experiment_regex,
                    'exclude_experiment_regex': args.exclude_experiment_regex,
                    'artifact_version': args.artifact_version,
                    'main_query_scope': args.main_query_scope,
                    'experiments_discovered': len(records),
                    'cross_query_chunk_modes': args.cross_query_chunk_modes,
                    'wording_configurations': wording_configurations,
                    'cross_triplet_analysis_enabled': cross_triplet_analysis_enabled,
                    'cross_triplet_figures_enabled': args.plots and cross_triplet_analysis_enabled,
                    'warnings_count': len(warnings),
                    'figures': [str(path.relative_to(args.output_dir)) for path in figures],
                    'files': list(REPORT_FILES),
                    'lambda_selection_metric': LAMBDA_SELECTION_MAXIMIZING_METRIC,
                    'near_optimal_epsilon': args.near_optimal_epsilon,
                    'bootstrap_replicates': args.bootstrap_replicates,
                    'bootstrap_seed': args.bootstrap_seed,
                    'validity_outputs': {
                        'geometry_population_runs': len(geometry_population_strategy_rows_data),
                        'global_lambda_runs': len(global_lambda_strategy_rows_data),
                        'lodo_lambda_runs': len(lodo_lambda_strategy_rows_data),
                        'synthetic_artifact_diagnostic_rows': len(
                            synthetic_artifact_diagnostic_rows_data
                        ),
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + '\n'
        )
        _progress('report generation complete')
        return ReportOutputs(
            output_dir=args.output_dir,
            experiments_discovered=len(records),
            experiments_loaded=sum(1 for record in records if record.cfg is not None),
            warnings_count=len(warnings),
            figures_count=len(figures),
        )
    finally:
        MedicalDatasetGenPaths.results_dir = old_results_dir


def refresh_report_plots(args: CliArgs) -> ReportOutputs:
    report_dir = args.refresh_report_dir or args.output_dir
    data_dir = report_dir / 'data'
    if not data_dir.is_dir():
        raise FileNotFoundError(f'report data directory not found: {data_dir}')

    warnings: list[str] = []
    _progress(f'loading existing report CSV artifacts from: {data_dir}')
    manifest_rows = _read_report_csv_rows(data_dir, 'experiment_manifest.csv')
    dataset_rows = _read_report_csv_rows(data_dir, 'dataset_distribution.csv')
    geometry_rows = _read_report_csv_rows(data_dir, 'geometry_filter_summary.csv')
    metric_family_summary_rows_data = _read_report_csv_rows(data_dir, 'metric_family_summary.csv')
    metric_family_budget_summary_rows_data = _read_report_csv_rows(
        data_dir, 'metric_family_budget_summary.csv'
    )
    metric_summary_rows = _read_report_csv_rows(data_dir, 'metric_aggregate_summary.csv')
    budget_rows = _read_report_csv_rows(data_dir, 'budget_strategy_summary.csv', required=True)
    lambda_rows = _read_report_csv_rows(data_dir, 'lambda_stability.csv')
    lambda_grid_delta_rows = _read_report_csv_rows(data_dir, 'lambda_grid_fcp_delta.csv')
    lambda_safety_rows = _read_report_csv_rows(data_dir, 'lambda_safety_summary.csv')
    near_optimal_rows = _read_report_csv_rows(data_dir, 'near_optimal_lambda_width.csv')
    paired_cell_rows = _read_report_csv_rows(data_dir, 'paired_cell_effect_summary.csv')
    paired_suite_rows = _read_report_csv_rows(data_dir, 'paired_suite_effect_summary.csv')
    paired_config_suite_rows = _read_report_csv_rows(
        data_dir, 'paired_config_suite_effect_summary.csv'
    )
    _progress('rendering report figures from existing CSV artifacts')
    figures = write_figures(
        output_dir=report_dir / 'figures',
        plot_format=args.plot_format,
        max_rows=args.max_table_rows,
        budget_rows=budget_rows,
        geometry_rows=geometry_rows,
        lambda_rows=lambda_rows,
        lambda_grid_delta_rows=lambda_grid_delta_rows,
        lambda_safety_rows=lambda_safety_rows,
        near_optimal_rows=near_optimal_rows,
        dataset_rows=dataset_rows,
        metric_summary_rows=metric_summary_rows,
        metric_family_summary_rows=metric_family_summary_rows_data,
        metric_family_budget_summary_rows=metric_family_budget_summary_rows_data,
        paired_cell_rows=paired_cell_rows,
        paired_suite_rows=paired_suite_rows,
        paired_config_suite_rows=paired_config_suite_rows,
        cross_query_chunk_modes=args.cross_query_chunk_modes,
        warnings=warnings,
    )
    _progress(f'rendered {len(figures)} figures')
    if warnings:
        _progress(f'plot-only refresh finished with {len(warnings)} warnings')
    else:
        _progress('plot-only refresh complete')
    return ReportOutputs(
        output_dir=report_dir,
        experiments_discovered=len(manifest_rows),
        experiments_loaded=len(manifest_rows),
        warnings_count=len(warnings),
        figures_count=len(figures),
    )


def refresh_latex_macros(args: CliArgs) -> ReportOutputs:
    report_dir = args.refresh_report_dir or args.output_dir
    data_dir = report_dir / 'data'
    if not data_dir.is_dir():
        raise FileNotFoundError(f'report data directory not found: {data_dir}')

    _progress(f'loading existing report CSV artifacts from: {data_dir}')
    manifest_rows = _read_report_csv_rows(data_dir, 'experiment_manifest.csv')
    geometry_rows = _read_report_csv_rows(data_dir, 'geometry_filter_summary.csv')
    comparison_rows = _read_report_csv_rows(data_dir, 'comparison_by_k.csv')
    budget_rows = _read_report_csv_rows(data_dir, 'budget_strategy_summary.csv', required=True)
    lambda_safety_rows = _read_report_csv_rows(data_dir, 'lambda_safety_summary.csv')
    metric_summary_rows = _read_report_csv_rows(data_dir, 'metric_aggregate_summary.csv')
    metric_family_summary_rows_data = _read_report_csv_rows(data_dir, 'metric_family_summary.csv')
    paired_suite_rows = _read_report_csv_rows(data_dir, 'paired_suite_effect_summary.csv')
    embedding_summary_rows = _read_report_csv_rows(data_dir, 'embedding_model_summary.csv')

    output_path = thesis_result_macros_path(report_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _progress(f'writing LaTeX result macros: {output_path}')
    output_path.write_text(
        render_thesis_result_macros(
            geometry_rows=geometry_rows,
            comparison_rows=comparison_rows,
            budget_rows=budget_rows,
            lambda_safety_rows=lambda_safety_rows,
            metric_summary_rows=metric_summary_rows,
            metric_family_summary_rows=metric_family_summary_rows_data,
            paired_suite_rows=paired_suite_rows,
            embedding_summary_rows=embedding_summary_rows,
            require_complete_wording_grid=args.cross_query_chunk_modes,
        )
    )
    _progress('LaTeX macro refresh complete')
    return ReportOutputs(
        output_dir=report_dir,
        experiments_discovered=len(manifest_rows),
        experiments_loaded=len(manifest_rows),
        warnings_count=0,
        figures_count=0,
    )


def _dedupe_warnings(warnings: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for warning in warnings:
        if warning in seen:
            continue
        seen.add(warning)
        deduped.append(warning)
    return deduped


def _progress(message: str) -> None:
    print(f'[reports] {message}', flush=True)


def _read_report_csv_rows(
    data_dir: Path,
    filename: str,
    *,
    required: bool = False,
) -> list[dict[str, object]]:
    path = data_dir / filename
    if not path.is_file():
        if required:
            raise FileNotFoundError(f'required report CSV not found: {path}')
        return []
    if path.stat().st_size == 0:
        return []
    return [dict(row) for row in pl.read_csv(path, infer_schema_length=None).to_dicts()]


def _load_refresh_recap_records(
    *,
    report_dir: Path,
    fallback_results_dir: Path,
    manifest_rows: Sequence[Mapping[str, object]],
    warnings: list[str],
) -> list[ExperimentRecord]:
    results_dir = _saved_report_results_dir(report_dir) or fallback_results_dir
    if not results_dir.is_dir():
        return []
    experiment_names = [
        str(row.get('Experiment') or '') for row in manifest_rows if row.get('Experiment')
    ]
    old_results_dir = MedicalDatasetGenPaths.results_dir
    MedicalDatasetGenPaths.results_dir = results_dir
    try:
        return [
            load_experiment_record(results_dir, name, warnings=warnings)
            for name in experiment_names
        ]
    finally:
        MedicalDatasetGenPaths.results_dir = old_results_dir


def _saved_report_results_dir(report_dir: Path) -> Path | None:
    manifest_path = report_dir / 'manifest.json'
    if not manifest_path.is_file():
        return None
    try:
        manifest: object = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(manifest, dict):
        return None
    raw_results_dir = manifest.get('results_dir')
    return (
        Path(raw_results_dir).expanduser().resolve() if isinstance(raw_results_dir, str) else None
    )


def _remove_obsolete_flat_data_files(output_dir: Path) -> None:
    for report_file in REPORT_FILES:
        path = Path(report_file)
        if len(path.parts) != 2 or path.parts[0] != 'data':
            continue
        obsolete_path = output_dir / path.name
        if path.suffix == '.csv':
            obsolete_path.unlink(missing_ok=True)
        elif report_file.endswith('/'):
            shutil.rmtree(obsolete_path, ignore_errors=True)


def _wording_configurations_for_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    labels_by_config: dict[str, str] = {}
    for row in rows:
        config = str(row.get('WordingConfig') or '')
        if not config:
            continue
        labels_by_config.setdefault(config, str(row.get('WordingConfigLabel') or config))
    return [
        {'config': config, 'label': label}
        for config, label in sorted(labels_by_config.items(), key=lambda item: item[0])
    ]


def _write_thesis_outputs_from_rows(
    *,
    geometry_rows: Sequence[Mapping[str, object]],
    comparison_rows: Sequence[Mapping[str, object]],
    budget_rows: Sequence[Mapping[str, object]],
    lambda_safety_rows: Sequence[Mapping[str, object]],
    metric_summary_rows: Sequence[Mapping[str, object]],
    metric_family_summary_rows: Sequence[Mapping[str, object]],
    metric_family_budget_summary_rows: Sequence[Mapping[str, object]],
    paired_suite_rows: Sequence[Mapping[str, object]],
    embedding_summary_rows: Sequence[Mapping[str, object]],
    require_complete_wording_grid: bool,
    output_dir: Path,
) -> None:
    latex_dir = thesis_latex_dir(output_dir)
    latex_dir.mkdir(parents=True, exist_ok=True)
    thesis_aggregate_tables_path(output_dir).write_text(
        render_thesis_aggregate_tables(
            metric_summary_rows=metric_summary_rows,
            metric_family_summary_rows=metric_family_summary_rows,
            metric_family_budget_summary_rows=metric_family_budget_summary_rows,
        )
    )
    thesis_result_macros_path(output_dir).write_text(
        render_thesis_result_macros(
            geometry_rows=geometry_rows,
            comparison_rows=comparison_rows,
            budget_rows=budget_rows,
            lambda_safety_rows=lambda_safety_rows,
            metric_summary_rows=metric_summary_rows,
            metric_family_summary_rows=metric_family_summary_rows,
            paired_suite_rows=paired_suite_rows,
            embedding_summary_rows=embedding_summary_rows,
            require_complete_wording_grid=require_complete_wording_grid,
        )
    )
    thesis_statistical_tables_path(output_dir).write_text(
        render_statistical_latex_table(paired_suite_rows)
    )


def main() -> None:
    outputs = run_report(parse_args())
    print(f'wrote report files to {outputs.output_dir}')
    print(
        f'experiments: {outputs.experiments_loaded}/{outputs.experiments_discovered} configs loaded; '
        f'warnings: {outputs.warnings_count}; figures: {outputs.figures_count}'
    )


if __name__ == '__main__':
    main()
