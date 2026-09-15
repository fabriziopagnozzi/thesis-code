from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from experiments.medical_dataset_gen.evaluation.lambda_selection import (
    LAMBDA_SELECTION_MAXIMIZING_METRIC,
)
from experiments.medical_dataset_gen.reports.analysis_constants import REPORT_FILES
from experiments.medical_dataset_gen.reports.analysis_scope import interaction_rows, primary_rows
from experiments.medical_dataset_gen.reports.artifacts import write_csv
from experiments.medical_dataset_gen.reports.cli import parse_args
from experiments.medical_dataset_gen.reports.discovery import (
    discover_experiments,
    discover_suite_experiments,
    load_report_logical_suite,
    suite_cells_matching_where,
)
from experiments.medical_dataset_gen.reports.geometry_coverage import (
    representation_audit_manifest_metadata,
)
from experiments.medical_dataset_gen.reports.helpers import ordered_embedding_models
from experiments.medical_dataset_gen.reports.latex_macros import (
    render_thesis_result_macros,
    thesis_latex_dir,
    thesis_result_macros_path,
)
from experiments.medical_dataset_gen.reports.model_analysis import (
    embedding_geometry_summary_rows,
    embedding_metric_range_rows,
    embedding_metric_summary_rows,
    lambda_curve_by_embedding_model_rows,
    model_grid_coverage_rows,
)
from experiments.medical_dataset_gen.reports.models import CliArgs, ExperimentRecord, ReportOutputs
from experiments.medical_dataset_gen.reports.plots import write_figures
from experiments.medical_dataset_gen.reports.rendering import (
    render_interesting_findings,
    render_report,
)
from experiments.medical_dataset_gen.reports.report_config import LEGACY_LOW_BUDGET_TOKEN
from experiments.medical_dataset_gen.reports.report_io import report_io_workers
from experiments.medical_dataset_gen.reports.rows import (
    dataset_distribution_row,
    experiment_manifest_row,
    geometry_filter_row,
    lambda_grid_fcp_delta_rows,
    lambda_safety_summary_rows,
    selected_strategy_rows,
)
from experiments.medical_dataset_gen.reports.suite_analysis import (
    matched_contrast_rows,
    write_suite_factor_figures,
)
from experiments.medical_dataset_gen.reports.summaries import (
    budget_category_rows_from_comparisons,
    comparison_by_k_rows,
    embedding_model_summary_rows,
    experiment_family_budget_summary_rows,
    experiment_family_summary_rows,
    lambda_curve_summary_rows,
    lambda_robustness_summary_rows,
    metric_aggregate_summary_rows,
    metric_family_budget_summary_rows,
    metric_family_summary_rows,
)
from experiments.medical_dataset_gen.reports.validity import synthetic_artifact_diagnostic_rows
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
        shutil.rmtree(args.output_dir / 'latex', ignore_errors=True)
        # Optional analyses can disappear between runs; rebuild the figure tree
        # so disabled-analysis plots cannot survive as stale report artifacts.
        shutil.rmtree(args.output_dir / 'figures', ignore_errors=True)
        _remove_obsolete_flat_data_files(args.output_dir)
        warnings: list[str] = []
        _progress(f'discovering completed experiments under: {args.results_dir}')
        suite_mode = args.suite_id is not None or args.suite_base_id is not None
        suite_selection = None
        if suite_mode:
            suite_selection = load_report_logical_suite(
                results_dir=args.results_dir,
                suite_id=args.suite_id or args.suite_base_id or '',
                suite_base_id=args.suite_base_id,
                suite_regex=args.suite_regex,
                warnings=warnings,
            )
            records = discover_suite_experiments(
                args.results_dir,
                suite_id=args.suite_id or args.suite_base_id or '',
                where=args.suite_where,
                warnings=warnings,
                suite_base_id=args.suite_base_id,
                suite_regex=args.suite_regex,
                logical_suite=suite_selection.logical_suite,
            )
        else:
            records = discover_experiments(
                args.results_dir,
                include_scrapped=args.include_scrapped,
                requested_experiments=args.experiments,
                experiment_regex=args.experiment_regex,
                exclude_experiment_regex=args.exclude_experiment_regex,
                warnings=warnings,
            )
        discovered_count = len(records)
        _progress(f'discovered {discovered_count} completed experiments')
        records, effective_embedding_models = _records_for_embedding_models(
            records=records,
            requested_embedding_models=args.embedding_models,
        )
        suite_manifest = None
        if suite_mode:
            assert suite_selection is not None
            suite_manifest = suite_selection.logical_suite.manifest
        _progress(f'{len(records)} experiments remain after embedding-model filtering')
        _progress(
            'using embedding models: '
            + (', '.join(effective_embedding_models) if effective_embedding_models else 'none')
        )
        _progress('loading manifest, dataset, and geometry rows')
        with ThreadPoolExecutor(max_workers=report_io_workers(len(records))) as executor:
            loaded_input_rows = executor.map(_load_report_input_rows, records)
            manifest_rows: list[dict[str, object]] = []
            dataset_rows: list[dict[str, object]] = []
            geometry_rows: list[dict[str, object]] = []
            for manifest_row, dataset_row, geometry_row, row_warnings in loaded_input_rows:
                manifest_rows.append(manifest_row)
                dataset_rows.append(dataset_row)
                geometry_rows.append(geometry_row)
                warnings.extend(row_warnings)

        _progress('loading selected strategy rows')
        strategy_rows: list[dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=report_io_workers(len(records))) as executor:
            loaded_strategy_rows = executor.map(_load_selected_rows, records)
            for selected_rows, row_warnings in loaded_strategy_rows:
                strategy_rows.extend(selected_rows)
                warnings.extend(row_warnings)

        _progress('computing lambda-grid diagnostics')
        lambda_grid_delta_rows = lambda_grid_fcp_delta_rows(records, warnings=warnings)
        lambda_safety_rows = lambda_safety_summary_rows(lambda_grid_delta_rows)
        lambda_curve_rows = lambda_curve_summary_rows(lambda_grid_delta_rows)
        lambda_robustness_rows = lambda_robustness_summary_rows(lambda_safety_rows)
        comparison_rows = comparison_by_k_rows(strategy_rows)
        suite_contrast_rows: list[dict[str, object]] = []
        if suite_mode:
            assert suite_manifest is not None
            suite_scope = (
                suite_cells_matching_where(suite_manifest, args.suite_where)
                if args.suite_where is not None
                else None
            )
            suite_contrast_rows = matched_contrast_rows(
                manifest=suite_manifest,
                comparison_rows=comparison_rows,
                enforce_strict=args.strict_suite,
                scope_cell_ids=suite_scope,
            )
        _progress('computing synthetic-artifact diagnostics')
        synthetic_artifact_diagnostic_rows_data = synthetic_artifact_diagnostic_rows(
            records,
            warnings=warnings,
        )
        _progress('computing aggregate family, budget, metric, and embedding summaries')
        primary_comparison_rows = primary_rows(comparison_rows)
        family_summary_rows = experiment_family_summary_rows(primary_comparison_rows)
        budget_rows = budget_category_rows_from_comparisons(primary_comparison_rows)
        family_budget_summary_rows = experiment_family_budget_summary_rows(budget_rows)
        metric_family_summary_rows_data = metric_family_summary_rows(primary_comparison_rows)
        metric_family_budget_summary_rows_data = metric_family_budget_summary_rows(budget_rows)
        metric_summary_rows = metric_aggregate_summary_rows(
            comparison_rows=primary_comparison_rows,
            budget_rows=budget_rows,
        )
        model_grid_rows, model_grid_missing_rows = model_grid_coverage_rows(
            comparison_rows,
            embedding_models=effective_embedding_models,
        )
        embedding_model_grid_complete = bool(model_grid_rows) and all(
            row.get('Complete') is True for row in model_grid_rows
        )
        embedding_metric_rows = embedding_metric_summary_rows(budget_rows)
        embedding_metric_ranges = embedding_metric_range_rows(
            embedding_metric_rows,
            complete_crossing=embedding_model_grid_complete,
        )
        embedding_geometry_rows, embedding_geometry_family_rows = embedding_geometry_summary_rows(
            geometry_rows
        )
        lambda_embedding_rows = lambda_curve_by_embedding_model_rows(lambda_grid_delta_rows)
        low_budget_rows = [row for row in budget_rows if row.get('BudgetCategory') == 'low_budget']
        embedding_summary_rows = embedding_model_summary_rows(
            manifest_rows=manifest_rows,
            geometry_rows=geometry_rows,
            low_budget_rows=low_budget_rows,
        )
        wording_configurations = _wording_configurations_for_rows(budget_rows)
        _progress('writing CSV report artifacts')
        (args.output_dir / f'{LEGACY_LOW_BUDGET_TOKEN}_strategy_summary.csv').unlink(
            missing_ok=True
        )
        write_csv(data_dir / 'experiment_manifest.csv', manifest_rows)
        write_csv(data_dir / 'dataset_distribution.csv', dataset_rows)
        write_csv(data_dir / 'geometry_filter_summary.csv', geometry_rows)
        write_csv(data_dir / 'strategy_by_k.csv', strategy_rows)
        write_csv(data_dir / 'comparison_by_k.csv', comparison_rows)
        if suite_mode:
            write_csv(data_dir / 'suite_matched_contrasts.csv', suite_contrast_rows)
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
        write_csv(data_dir / 'lambda_grid_fcp_delta.csv', lambda_grid_delta_rows)
        write_csv(data_dir / 'lambda_safety_summary.csv', lambda_safety_rows)
        write_csv(data_dir / 'lambda_curve_summary.csv', lambda_curve_rows)
        write_csv(data_dir / 'lambda_robustness_summary.csv', lambda_robustness_rows)
        write_csv(data_dir / 'embedding_model_summary.csv', embedding_summary_rows)
        write_csv(data_dir / 'embedding_model_grid_coverage.csv', model_grid_rows)
        write_csv(data_dir / 'embedding_model_grid_missing.csv', model_grid_missing_rows)
        write_csv(data_dir / 'embedding_model_metric_summary.csv', embedding_metric_rows)
        write_csv(data_dir / 'embedding_model_metric_ranges.csv', embedding_metric_ranges)
        write_csv(data_dir / 'embedding_geometry_summary.csv', embedding_geometry_rows)
        write_csv(
            data_dir / 'embedding_geometry_family_summary.csv',
            embedding_geometry_family_rows,
        )
        write_csv(
            data_dir / 'lambda_curve_by_embedding_model.csv',
            lambda_embedding_rows,
        )
        _progress('writing thesis LaTeX macros')
        _write_thesis_outputs_from_rows(
            geometry_rows=geometry_rows,
            comparison_rows=comparison_rows,
            budget_rows=budget_rows,
            lambda_safety_rows=lambda_safety_rows,
            synthetic_artifact_rows=synthetic_artifact_diagnostic_rows_data,
            metric_summary_rows=metric_summary_rows,
            metric_family_summary_rows=metric_family_summary_rows_data,
            metric_family_budget_summary_rows=metric_family_budget_summary_rows_data,
            embedding_summary_rows=embedding_summary_rows,
            embedding_metric_rows=embedding_metric_rows,
            embedding_metric_range_rows=embedding_metric_ranges,
            embedding_geometry_rows=embedding_geometry_rows,
            embedding_models=effective_embedding_models,
            require_complete_wording_grid=True,
            warnings=warnings,
            lambda_curve_rows=lambda_curve_rows,
            lambda_robustness_rows=lambda_robustness_rows,
            output_dir=args.output_dir,
        )

        figures: list[Path] = []
        if args.plots:
            _progress('rendering report figures')
            figures = write_figures(
                output_dir=args.output_dir / 'figures',
                plot_format=args.plot_format,
                lambda_grid_delta_rows=primary_rows(lambda_grid_delta_rows),
                dataset_rows=primary_rows(dataset_rows),
                warnings=warnings,
                lambda_curve_rows=lambda_curve_rows,
                embedding_metric_rows=embedding_metric_rows,
                embedding_geometry_rows=embedding_geometry_rows,
                embedding_geometry_family_rows=embedding_geometry_family_rows,
                lambda_embedding_rows=lambda_embedding_rows,
            )
            figures.extend(
                write_figures(
                    output_dir=args.output_dir / 'figures' / 'interactions',
                    plot_format=args.plot_format,
                    dataset_rows=interaction_rows(dataset_rows),
                    warnings=warnings,
                )
            )
            if suite_mode:
                figures.extend(
                    write_suite_factor_figures(
                        output_dir=args.output_dir,
                        contrast_rows=suite_contrast_rows,
                    )
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
            lambda_safety_rows=lambda_safety_rows,
            lambda_robustness_rows=lambda_robustness_rows,
            embedding_summary_rows=embedding_summary_rows,
            embedding_metric_rows=embedding_metric_rows,
            embedding_metric_range_rows=embedding_metric_ranges,
            embedding_geometry_rows=embedding_geometry_rows,
            model_grid_rows=model_grid_rows,
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
                lambda_safety_rows=lambda_safety_rows,
                embedding_summary_rows=embedding_summary_rows,
                tablefmt=args.tablefmt,
                max_table_rows=args.max_table_rows,
            )
        )
        (args.output_dir / 'warnings.txt').write_text(
            '\n'.join(warnings) + ('\n' if warnings else '')
        )
        generated_files = sorted(
            path.relative_to(args.output_dir).as_posix()
            for path in args.output_dir.rglob('*')
            if path.is_file() and path.name != 'manifest.json'
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
                    'requested_embedding_models': list(args.embedding_models),
                    'effective_embedding_models': list(effective_embedding_models),
                    'embedding_model_grid_complete': embedding_model_grid_complete,
                    'thesis_ready_model_crossing': {
                        'complete': embedding_model_grid_complete,
                        'models': len(model_grid_rows),
                        'missing_cells': len(model_grid_missing_rows),
                    },
                    'suite_id': args.suite_id,
                    'suite_base_id': args.suite_base_id,
                    'suite_regex': args.suite_regex,
                    'suite_ids': list(suite_selection.suite_ids) if suite_selection else [],
                    'suite_where': args.suite_where,
                    'strict_suite': args.strict_suite,
                    'experiments_discovered': discovered_count,
                    'experiments_after_embedding_filter': len(records),
                    'wording_configurations': wording_configurations,
                    'warnings_count': len(warnings),
                    'figures': [str(path.relative_to(args.output_dir)) for path in figures],
                    'files': [*generated_files, 'manifest.json'],
                    'lambda_selection_metric': LAMBDA_SELECTION_MAXIMIZING_METRIC,
                    'representation_audit': representation_audit_manifest_metadata(
                        suite_selection.suite_ids if suite_selection else ()
                    ),
                    'synthetic_artifact_diagnostic_rows': len(
                        synthetic_artifact_diagnostic_rows_data
                    ),
                    'suite_matched_contrasts': len(suite_contrast_rows),
                },
                indent=2,
                sort_keys=True,
            )
            + '\n'
        )
        _progress('report generation complete')
        return ReportOutputs(
            output_dir=args.output_dir,
            experiments_discovered=discovered_count,
            experiments_loaded=sum(1 for record in records if record.cfg is not None),
            warnings_count=len(warnings),
            figures_count=len(figures),
        )
    finally:
        MedicalDatasetGenPaths.results_dir = old_results_dir


def refresh_report_plots(args: CliArgs) -> ReportOutputs:
    if args.embedding_models:
        raise ValueError(
            '--embedding-models requires a full report regeneration; refresh-only commands use '
            'the already aggregated CSV artifacts.'
        )
    report_dir = args.refresh_report_dir or args.output_dir
    data_dir = report_dir / 'data'
    if not data_dir.is_dir():
        raise FileNotFoundError(f'report data directory not found: {data_dir}')

    warnings: list[str] = []
    _progress(f'loading existing report CSV artifacts from: {data_dir}')
    manifest_rows = _read_report_csv_rows(data_dir, 'experiment_manifest.csv')
    dataset_rows = _read_report_csv_rows(data_dir, 'dataset_distribution.csv')
    lambda_grid_delta_rows = _read_report_csv_rows(data_dir, 'lambda_grid_fcp_delta.csv')
    lambda_curve_rows = _read_report_csv_rows(data_dir, 'lambda_curve_summary.csv', required=False)
    suite_contrast_rows = _read_report_csv_rows(data_dir, 'suite_matched_contrasts.csv')
    embedding_metric_rows = _read_report_csv_rows(data_dir, 'embedding_model_metric_summary.csv')
    embedding_geometry_rows = _read_report_csv_rows(data_dir, 'embedding_geometry_summary.csv')
    embedding_geometry_family_rows = _read_report_csv_rows(
        data_dir, 'embedding_geometry_family_summary.csv'
    )
    lambda_embedding_rows = _read_report_csv_rows(data_dir, 'lambda_curve_by_embedding_model.csv')
    _progress('rendering report figures from existing CSV artifacts')
    shutil.rmtree(report_dir / 'figures', ignore_errors=True)
    figures = write_figures(
        output_dir=report_dir / 'figures',
        plot_format=args.plot_format,
        lambda_grid_delta_rows=primary_rows(lambda_grid_delta_rows),
        dataset_rows=primary_rows(dataset_rows),
        warnings=warnings,
        lambda_curve_rows=lambda_curve_rows,
        embedding_metric_rows=embedding_metric_rows,
        embedding_geometry_rows=embedding_geometry_rows,
        embedding_geometry_family_rows=embedding_geometry_family_rows,
        lambda_embedding_rows=lambda_embedding_rows,
    )
    figures.extend(
        write_figures(
            output_dir=report_dir / 'figures' / 'interactions',
            plot_format=args.plot_format,
            dataset_rows=interaction_rows(dataset_rows),
            warnings=warnings,
        )
    )
    figures.extend(
        write_suite_factor_figures(
            output_dir=report_dir,
            contrast_rows=suite_contrast_rows,
        )
    )
    _update_refreshed_figure_manifest(report_dir=report_dir, figures=figures)
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
    if args.embedding_models:
        raise ValueError(
            '--embedding-models requires a full report regeneration; refresh-only commands use '
            'the already aggregated CSV artifacts.'
        )
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
    lambda_curve_rows = _read_report_csv_rows(data_dir, 'lambda_curve_summary.csv', required=False)
    lambda_robustness_rows = _read_report_csv_rows(
        data_dir, 'lambda_robustness_summary.csv', required=False
    )
    synthetic_artifact_rows = _read_report_csv_rows(data_dir, 'synthetic_artifact_diagnostics.csv')
    metric_summary_rows = _read_report_csv_rows(data_dir, 'metric_aggregate_summary.csv')
    metric_family_summary_rows_data = _read_report_csv_rows(data_dir, 'metric_family_summary.csv')
    metric_family_budget_summary_rows_data = _read_report_csv_rows(
        data_dir, 'metric_family_budget_summary.csv'
    )
    embedding_summary_rows = _read_report_csv_rows(data_dir, 'embedding_model_summary.csv')
    embedding_metric_rows = _read_report_csv_rows(data_dir, 'embedding_model_metric_summary.csv')
    embedding_metric_range_rows = _read_report_csv_rows(
        data_dir, 'embedding_model_metric_ranges.csv'
    )
    embedding_geometry_rows = _read_report_csv_rows(data_dir, 'embedding_geometry_summary.csv')
    warnings_path = report_dir / 'warnings.txt'
    wording_warning_prefixes = (
        'Wording result macros were omitted:',
        'Wording configurations use different held-out test-query counts:',
    )
    warnings = [
        line
        for line in (warnings_path.read_text().splitlines() if warnings_path.is_file() else [])
        if line and not line.startswith(wording_warning_prefixes)
    ]

    output_path = thesis_result_macros_path(report_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _progress(f'writing LaTeX result macros: {output_path}')
    output_path.write_text(
        render_thesis_result_macros(
            geometry_rows=geometry_rows,
            comparison_rows=comparison_rows,
            budget_rows=budget_rows,
            lambda_safety_rows=lambda_safety_rows,
            lambda_curve_rows=lambda_curve_rows,
            lambda_robustness_rows=lambda_robustness_rows,
            synthetic_artifact_rows=synthetic_artifact_rows,
            metric_summary_rows=metric_summary_rows,
            metric_family_summary_rows=metric_family_summary_rows_data,
            metric_family_budget_summary_rows=metric_family_budget_summary_rows_data,
            embedding_summary_rows=embedding_summary_rows,
            embedding_metric_rows=embedding_metric_rows,
            embedding_metric_range_rows=embedding_metric_range_rows,
            embedding_geometry_rows=embedding_geometry_rows,
            embedding_models=_effective_embedding_models_for_rows(budget_rows),
            require_complete_wording_grid=True,
            warnings=warnings,
        )
    )
    warnings = _dedupe_warnings(warnings)
    warnings_path.write_text('\n'.join(warnings) + ('\n' if warnings else ''))
    _progress('LaTeX macro refresh complete')
    return ReportOutputs(
        output_dir=report_dir,
        experiments_discovered=len(manifest_rows),
        experiments_loaded=len(manifest_rows),
        warnings_count=len(warnings),
        figures_count=0,
    )


def _records_for_embedding_models(
    *,
    records: Sequence[ExperimentRecord],
    requested_embedding_models: Sequence[str],
) -> tuple[list[ExperimentRecord], tuple[str, ...]]:
    available_models = ordered_embedding_models(record.embedding_model for record in records)
    if not requested_embedding_models:
        return list(records), tuple(available_models)

    requested_models = tuple(requested_embedding_models)
    available_model_set = set(available_models)
    requested_model_set = set(requested_models)
    missing_models = [model for model in requested_models if model not in available_model_set]
    if missing_models:
        raise ValueError(
            'Requested embedding models were not found in the discovered experiments: '
            f'{missing_models}. Available models: {available_models}.'
        )
    filtered_records = [
        record for record in records if record.embedding_model in requested_model_set
    ]
    if not filtered_records:
        raise ValueError('No experiments remain after applying --embedding-models.')
    return filtered_records, requested_models


def _effective_embedding_models_for_rows(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    return tuple(
        ordered_embedding_models(
            str(row.get('EmbeddingModel') or '') for row in rows if row.get('EmbeddingModel')
        )
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


def _load_report_input_rows(
    record: ExperimentRecord,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], tuple[str, ...]]:
    """Read one experiment's independent inputs for ordered parallel collection."""
    warnings: list[str] = []
    return (
        experiment_manifest_row(record),
        dataset_distribution_row(record, warnings=warnings),
        geometry_filter_row(record, warnings=warnings),
        tuple(warnings),
    )


def _load_selected_rows(
    record: ExperimentRecord,
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    """Load one record's selected-strategy rows for ordered parallel collection."""
    warnings: list[str] = []
    selected_rows = selected_strategy_rows(record, warnings=warnings)
    return selected_rows, tuple(warnings)


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


def _update_refreshed_figure_manifest(*, report_dir: Path, figures: Sequence[Path]) -> None:
    """Keep a plot-only refresh auditable without recomputing report tables."""
    manifest_path = report_dir / 'manifest.json'
    if not manifest_path.is_file():
        return
    try:
        manifest: object = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return
    if not isinstance(manifest, dict):
        return

    figure_paths = sorted(
        str(path.relative_to(report_dir))
        for path in figures
        if path.is_file() and path.is_relative_to(report_dir)
    )
    manifest['figures'] = figure_paths
    existing_files = manifest.get('files')
    non_figure_files = (
        [
            str(path)
            for path in existing_files
            if isinstance(path, str) and not path.startswith('figures/')
        ]
        if isinstance(existing_files, list)
        else []
    )
    manifest['files'] = [*non_figure_files, *figure_paths]
    manifest['figures_refreshed_at_utc'] = datetime.now(UTC).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')


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
    lambda_curve_rows: Sequence[Mapping[str, object]],
    lambda_robustness_rows: Sequence[Mapping[str, object]],
    synthetic_artifact_rows: Sequence[Mapping[str, object]],
    metric_summary_rows: Sequence[Mapping[str, object]],
    metric_family_summary_rows: Sequence[Mapping[str, object]],
    metric_family_budget_summary_rows: Sequence[Mapping[str, object]],
    embedding_summary_rows: Sequence[Mapping[str, object]],
    embedding_metric_rows: Sequence[Mapping[str, object]],
    embedding_metric_range_rows: Sequence[Mapping[str, object]],
    embedding_geometry_rows: Sequence[Mapping[str, object]],
    embedding_models: Sequence[str],
    require_complete_wording_grid: bool,
    warnings: list[str],
    output_dir: Path,
) -> None:
    latex_dir = thesis_latex_dir(output_dir)
    latex_dir.mkdir(parents=True, exist_ok=True)
    thesis_result_macros_path(output_dir).write_text(
        render_thesis_result_macros(
            geometry_rows=geometry_rows,
            comparison_rows=comparison_rows,
            budget_rows=budget_rows,
            lambda_safety_rows=lambda_safety_rows,
            synthetic_artifact_rows=synthetic_artifact_rows,
            metric_summary_rows=metric_summary_rows,
            metric_family_summary_rows=metric_family_summary_rows,
            metric_family_budget_summary_rows=metric_family_budget_summary_rows,
            embedding_summary_rows=embedding_summary_rows,
            embedding_metric_rows=embedding_metric_rows,
            embedding_metric_range_rows=embedding_metric_range_rows,
            embedding_geometry_rows=embedding_geometry_rows,
            embedding_models=embedding_models,
            require_complete_wording_grid=require_complete_wording_grid,
            warnings=warnings,
            lambda_curve_rows=lambda_curve_rows,
            lambda_robustness_rows=lambda_robustness_rows,
        )
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
