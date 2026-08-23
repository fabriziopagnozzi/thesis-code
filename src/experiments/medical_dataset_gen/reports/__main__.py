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
from experiments.medical_dataset_gen.reports.analysis_scope import interaction_rows, primary_rows
from experiments.medical_dataset_gen.reports.artifacts import (
    render_experiment_config_recap,
    write_csv,
)
from experiments.medical_dataset_gen.reports.cli import parse_args
from experiments.medical_dataset_gen.reports.discovery import (
    discover_experiments,
    discover_suite_experiments,
    load_experiment_record,
    suite_cells_matching_where,
)
from experiments.medical_dataset_gen.reports.helpers import ordered_embedding_models
from experiments.medical_dataset_gen.reports.latex_macros import render_thesis_result_macros
from experiments.medical_dataset_gen.reports.latex_tables import (
    render_thesis_aggregate_tables,
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
from experiments.medical_dataset_gen.reports.suite_analysis import (
    analysis_series_rows,
    crossing_rows,
    factor_interaction_rows,
    matched_contrast_rows,
    report_eligible_manifest,
    suite_distribution_and_family_rows,
    write_suite_factor_figures,
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
from experiments.medical_dataset_gen.suites.core import load_logical_suite
from experiments.medical_dataset_gen.suites.geometry import apply_frozen_separability_strata
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
        # Optional analyses can disappear between runs; rebuild the figure tree
        # so disabled-analysis plots cannot survive as stale report artifacts.
        shutil.rmtree(args.output_dir / 'figures', ignore_errors=True)
        _remove_obsolete_flat_data_files(args.output_dir)
        warnings: list[str] = []
        _progress(f'discovering completed experiments under: {args.results_dir}')
        records = (
            discover_suite_experiments(
                args.results_dir,
                suite_id=args.suite_id,
                where=args.suite_where,
                warnings=warnings,
            )
            if args.suite_id is not None
            else discover_experiments(
                args.results_dir,
                include_scrapped=args.include_scrapped,
                requested_experiments=args.experiments,
                experiment_regex=args.experiment_regex,
                exclude_experiment_regex=args.exclude_experiment_regex,
                artifact_version=args.artifact_version,
                warnings=warnings,
            )
        )
        discovered_count = len(records)
        _progress(f'discovered {discovered_count} completed experiments')
        records, effective_embedding_models = _records_for_embedding_models(
            records=records,
            requested_embedding_models=args.embedding_models,
        )
        suite_manifest = None
        if args.suite_id is not None:
            materialized_manifest = load_logical_suite(args.results_dir, args.suite_id).manifest
            suite_manifest, excluded_distributions = report_eligible_manifest(materialized_manifest)
            if excluded_distributions:
                eligible_names = {cell.name for cell in suite_manifest.cells}
                excluded_records = len(records) - sum(
                    record.name in eligible_names for record in records
                )
                records = [record for record in records if record.name in eligible_names]
                warnings.append(
                    'excluded legacy background variants that do not satisfy the background-outlier '
                    f'definition: {", ".join(sorted(excluded_distributions))} '
                    f'({excluded_records} completed run-profile cells)'
                )
        _progress(f'{len(records)} experiments remain after embedding-model filtering')
        _progress(
            'using embedding models: '
            + (', '.join(effective_embedding_models) if effective_embedding_models else 'none')
        )
        plot_and_recap_records = records
        lambda_analysis_enabled = args.run_lambda_analysis
        global_lambda_analysis_enabled = args.run_global_lambda_analysis
        lodo_analysis_enabled = args.run_lodo_analysis
        paired_statistics_enabled = args.run_paired_statistics
        validity_analysis_enabled = args.run_validity_analysis
        geometry_population_enabled = (
            validity_analysis_enabled or args.main_query_scope == 'geometry_eligible'
        )

        _progress('loading manifest, dataset, and geometry rows')
        manifest_rows = [experiment_manifest_row(record) for record in records]
        dataset_rows = [dataset_distribution_row(record, warnings=warnings) for record in records]
        geometry_rows = [geometry_filter_row(record, warnings=warnings) for record in records]

        geometry_population_strategy_rows_data: list[dict[str, object]] = []
        if geometry_population_enabled:
            _progress('computing geometry-population validity summaries')
            geometry_population_strategy_rows_data = geometry_population_strategy_rows(
                plot_and_recap_records,
                warnings=warnings,
            )
        else:
            _progress('skipping geometry-population validity analysis')
        geometry_population_comparison_rows = comparison_by_k_rows(
            geometry_population_strategy_rows_data
        )

        _progress('loading selected strategy rows')
        strategy_rows: list[dict[str, object]] = []
        near_optimal_rows: list[dict[str, object]] = []
        for record in records:
            strategy_rows.extend(selected_strategy_rows(record, warnings=warnings))
            if lambda_analysis_enabled:
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

        lambda_grid_delta_rows: list[dict[str, object]] = []
        lambda_safety_rows: list[dict[str, object]] = []
        if lambda_analysis_enabled:
            _progress('computing lambda-grid diagnostics')
            lambda_grid_delta_rows = lambda_grid_fcp_delta_rows(records, warnings=warnings)
            lambda_safety_rows = lambda_safety_summary_rows(lambda_grid_delta_rows)
        else:
            _progress('skipping lambda-grid diagnostics')
        comparison_rows = comparison_by_k_rows(strategy_rows)
        suite_distribution_rows: list[dict[str, object]] = []
        suite_family_rows: list[dict[str, object]] = []
        suite_contrast_rows: list[dict[str, object]] = []
        suite_analysis_series_rows: list[dict[str, object]] = []
        suite_interaction_rows: list[dict[str, object]] = []
        suite_crossing_rows: list[dict[str, object]] = []
        suite_separability_rows: list[dict[str, object]] = []
        if args.suite_id is not None:
            assert suite_manifest is not None
            suite_scope = (
                suite_cells_matching_where(suite_manifest, args.suite_where)
                if args.suite_where is not None
                else None
            )
            suite_distribution_rows, suite_family_rows = suite_distribution_and_family_rows(
                comparison_rows
            )
            suite_contrast_rows = matched_contrast_rows(
                manifest=suite_manifest,
                comparison_rows=comparison_rows,
                enforce_strict=args.strict_suite,
                scope_cell_ids=suite_scope,
            )
            suite_analysis_series_rows = analysis_series_rows(
                manifest=suite_manifest,
                comparison_rows=comparison_rows,
                enforce_strict=args.strict_suite,
                scope_cell_ids=suite_scope,
            )
            suite_interaction_rows = factor_interaction_rows(
                manifest=suite_manifest,
                comparison_rows=comparison_rows,
            )
            suite_crossing_rows = crossing_rows(suite_contrast_rows)
            write_suite_factor_figures(
                output_dir=args.output_dir,
                contrast_rows=suite_contrast_rows,
            )
            suite_separability_rows = apply_frozen_separability_strata(
                results_dir=args.results_dir,
                suite_id=args.suite_id,
            )
            eligible_distributions = {
                distribution.distribution_id for distribution in suite_manifest.distributions
            }
            suite_separability_rows = [
                row
                for row in suite_separability_rows
                if row.get('Distribution') in eligible_distributions
            ]
        global_lambda_strategy_rows_data: list[dict[str, object]] = []
        if global_lambda_analysis_enabled:
            _progress('computing global-lambda validity summaries')
            global_lambda_strategy_rows_data = global_lambda_strategy_rows(
                records, warnings=warnings
            )
        else:
            _progress('skipping global-lambda validity analysis')
        global_lambda_comparison_rows = comparison_by_k_rows(global_lambda_strategy_rows_data)
        global_lambda_budget_rows = budget_category_rows_from_comparisons(
            global_lambda_comparison_rows
        )
        global_lambda_metric_summary_rows = metric_aggregate_summary_rows(
            comparison_rows=global_lambda_comparison_rows,
            budget_rows=global_lambda_budget_rows,
        )
        lodo_lambda_strategy_rows_data: list[dict[str, object]] = []
        if lodo_analysis_enabled:
            _progress('computing leave-one-distribution-out lambda validity summaries')
            lodo_lambda_strategy_rows_data = lodo_lambda_strategy_rows(records, warnings=warnings)
        else:
            _progress('skipping leave-one-distribution-out analysis')
        lodo_lambda_comparison_rows = comparison_by_k_rows(lodo_lambda_strategy_rows_data)
        lodo_lambda_budget_rows = budget_category_rows_from_comparisons(lodo_lambda_comparison_rows)
        lodo_lambda_metric_summary_rows = metric_aggregate_summary_rows(
            comparison_rows=lodo_lambda_comparison_rows,
            budget_rows=lodo_lambda_budget_rows,
        )
        synthetic_artifact_diagnostic_rows_data: list[dict[str, object]] = []
        if validity_analysis_enabled:
            _progress('computing synthetic-artifact diagnostics')
            synthetic_artifact_diagnostic_rows_data = synthetic_artifact_diagnostic_rows(
                records,
                warnings=warnings,
            )
        else:
            _progress('skipping synthetic-artifact diagnostics')
        _progress('computing aggregate family, budget, metric, and embedding summaries')
        primary_comparison_rows = primary_rows(comparison_rows)
        interaction_comparison_rows = interaction_rows(comparison_rows)
        family_summary_rows = experiment_family_summary_rows(primary_comparison_rows)
        budget_rows = budget_category_rows_from_comparisons(primary_comparison_rows)
        interaction_budget_rows = budget_category_rows_from_comparisons(interaction_comparison_rows)
        family_budget_summary_rows = experiment_family_budget_summary_rows(budget_rows)
        interaction_family_budget_summary_rows = metric_family_budget_summary_rows(
            interaction_budget_rows
        )
        metric_family_summary_rows_data = metric_family_summary_rows(primary_comparison_rows)
        interaction_metric_family_summary_rows = metric_family_summary_rows(
            interaction_comparison_rows
        )
        metric_family_budget_summary_rows_data = metric_family_budget_summary_rows(budget_rows)
        interaction_metric_summary_rows = metric_aggregate_summary_rows(
            comparison_rows=interaction_comparison_rows,
            budget_rows=interaction_budget_rows,
        )
        metric_summary_rows = metric_aggregate_summary_rows(
            comparison_rows=primary_comparison_rows,
            budget_rows=budget_rows,
        )
        low_budget_rows = [row for row in budget_rows if row.get('BudgetCategory') == 'low_budget']
        lambda_rows = (
            lambda_stability_rows(strategy_rows, near_optimal_rows)
            if lambda_analysis_enabled
            else []
        )
        embedding_summary_rows = embedding_model_summary_rows(
            manifest_rows=manifest_rows,
            geometry_rows=geometry_rows,
            low_budget_rows=low_budget_rows,
        )
        wording_configurations = _wording_configurations_for_rows(budget_rows)
        cross_triplet_analysis_enabled = (
            args.cross_query_chunk_modes and len(wording_configurations) > 1
        )
        paired_cell_rows: list[dict[str, object]] = []
        paired_suite_rows: list[dict[str, object]] = []
        paired_config_suite_rows: list[dict[str, object]] = []
        paired_sensitivity_rows: list[dict[str, object]] = []
        if paired_statistics_enabled:
            _progress('writing paired query/profile effect datasets')
            paired_profile_effects = write_paired_effect_datasets(
                records=records,
                strategy_rows=strategy_rows,
                output_dir=data_dir,
                warnings=warnings,
            )
            _progress('computing paired bootstrap summaries')
            paired_cell_rows = cell_effect_summary_rows(
                profile_effects=paired_profile_effects,
                budget_rows=budget_rows,
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.bootstrap_seed,
            )
            paired_suite_rows = suite_effect_summary_rows(
                profile_effects=paired_profile_effects,
                budget_rows=budget_rows,
                embedding_models=effective_embedding_models,
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.bootstrap_seed,
            )
            if args.cross_query_chunk_modes:
                paired_config_suite_rows = configuration_suite_effect_summary_rows(
                    profile_effects=paired_profile_effects,
                    budget_rows=budget_rows,
                    embedding_models=effective_embedding_models,
                    bootstrap_replicates=args.bootstrap_replicates,
                    bootstrap_seed=args.bootstrap_seed,
                )
            paired_sensitivity_rows = leave_one_out_sensitivity_rows(
                profile_effects=paired_profile_effects,
                budget_rows=budget_rows,
                embedding_models=effective_embedding_models,
            )
            del paired_profile_effects
        else:
            _progress('skipping paired statistical analysis')

        _progress('writing CSV report artifacts')
        (args.output_dir / f'{LEGACY_LOW_BUDGET_TOKEN}_strategy_summary.csv').unlink(
            missing_ok=True
        )
        write_csv(data_dir / 'experiment_manifest.csv', manifest_rows)
        write_csv(data_dir / 'dataset_distribution.csv', dataset_rows)
        write_csv(data_dir / 'geometry_filter_summary.csv', geometry_rows)
        write_csv(data_dir / 'strategy_by_k.csv', strategy_rows)
        write_csv(data_dir / 'comparison_by_k.csv', comparison_rows)
        if args.suite_id is not None:
            write_csv(data_dir / 'suite_distribution_summary.csv', suite_distribution_rows)
            write_csv(data_dir / 'suite_family_balanced_summary.csv', suite_family_rows)
            write_csv(data_dir / 'suite_matched_contrasts.csv', suite_contrast_rows)
            write_csv(data_dir / 'suite_analysis_series.csv', suite_analysis_series_rows)
            write_csv(data_dir / 'suite_factor_interactions.csv', suite_interaction_rows)
            write_csv(data_dir / 'suite_factor_crossings.csv', suite_crossing_rows)
            write_csv(data_dir / 'suite_separability_test_strata.csv', suite_separability_rows)
        if geometry_population_enabled:
            write_csv(
                data_dir / 'geometry_population_strategy_by_k.csv',
                geometry_population_strategy_rows_data,
            )
            write_csv(
                data_dir / 'geometry_population_comparison_by_k.csv',
                geometry_population_comparison_rows,
            )
        if global_lambda_analysis_enabled:
            write_csv(
                data_dir / 'global_lambda_strategy_by_k.csv',
                global_lambda_strategy_rows_data,
            )
            write_csv(data_dir / 'global_lambda_comparison_by_k.csv', global_lambda_comparison_rows)
            write_csv(
                data_dir / 'global_lambda_metric_aggregate_summary.csv',
                global_lambda_metric_summary_rows,
            )
        if lodo_analysis_enabled:
            write_csv(data_dir / 'lodo_lambda_strategy_by_k.csv', lodo_lambda_strategy_rows_data)
            write_csv(data_dir / 'lodo_lambda_comparison_by_k.csv', lodo_lambda_comparison_rows)
            write_csv(
                data_dir / 'lodo_lambda_metric_aggregate_summary.csv',
                lodo_lambda_metric_summary_rows,
            )
        if validity_analysis_enabled:
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
        write_csv(data_dir / 'interaction_budget_strategy_summary.csv', interaction_budget_rows)
        write_csv(
            data_dir / 'interaction_metric_family_summary.csv',
            interaction_metric_family_summary_rows,
        )
        write_csv(
            data_dir / 'interaction_metric_family_budget_summary.csv',
            interaction_family_budget_summary_rows,
        )
        write_csv(
            data_dir / 'interaction_metric_aggregate_summary.csv', interaction_metric_summary_rows
        )
        write_csv(data_dir / 'low_budget_strategy_summary.csv', low_budget_rows)
        if lambda_analysis_enabled:
            write_csv(data_dir / 'lambda_stability.csv', lambda_rows)
            write_csv(data_dir / 'lambda_grid_fcp_delta.csv', lambda_grid_delta_rows)
            write_csv(data_dir / 'lambda_safety_summary.csv', lambda_safety_rows)
            write_csv(data_dir / 'near_optimal_lambda_width.csv', near_optimal_rows)
        write_csv(data_dir / 'embedding_model_summary.csv', embedding_summary_rows)
        if paired_statistics_enabled:
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
            synthetic_artifact_rows=synthetic_artifact_diagnostic_rows_data,
            metric_summary_rows=metric_summary_rows,
            metric_family_summary_rows=metric_family_summary_rows_data,
            metric_family_budget_summary_rows=metric_family_budget_summary_rows_data,
            paired_suite_rows=paired_suite_rows,
            embedding_summary_rows=embedding_summary_rows,
            embedding_models=effective_embedding_models,
            require_complete_wording_grid=args.cross_query_chunk_modes,
            warnings=warnings,
            paired_statistics=paired_statistics_enabled,
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
                geometry_rows=primary_rows(geometry_rows),
                lambda_rows=primary_rows(lambda_rows),
                lambda_grid_delta_rows=primary_rows(lambda_grid_delta_rows),
                lambda_safety_rows=primary_rows(lambda_safety_rows),
                near_optimal_rows=primary_rows(near_optimal_rows),
                dataset_rows=primary_rows(dataset_rows),
                metric_summary_rows=metric_summary_rows,
                metric_family_summary_rows=metric_family_summary_rows_data,
                metric_family_budget_summary_rows=metric_family_budget_summary_rows_data,
                paired_cell_rows=primary_rows(paired_cell_rows),
                paired_suite_rows=paired_suite_rows,
                paired_config_suite_rows=paired_config_suite_rows,
                cross_query_chunk_modes=args.cross_query_chunk_modes,
                warnings=warnings,
            )
            figures.extend(
                write_figures(
                    output_dir=args.output_dir / 'figures' / 'interactions',
                    plot_format=args.plot_format,
                    max_rows=args.max_table_rows,
                    budget_rows=interaction_budget_rows,
                    geometry_rows=interaction_rows(geometry_rows),
                    lambda_rows=interaction_rows(lambda_rows),
                    lambda_grid_delta_rows=interaction_rows(lambda_grid_delta_rows),
                    lambda_safety_rows=interaction_rows(lambda_safety_rows),
                    near_optimal_rows=interaction_rows(near_optimal_rows),
                    dataset_rows=interaction_rows(dataset_rows),
                    metric_summary_rows=interaction_metric_summary_rows,
                    metric_family_summary_rows=interaction_metric_family_summary_rows,
                    metric_family_budget_summary_rows=interaction_family_budget_summary_rows,
                    paired_cell_rows=interaction_rows(paired_cell_rows),
                    paired_suite_rows=[
                        row
                        for row in paired_suite_rows
                        if row.get('Scope') == 'Interaction experiments'
                    ],
                    paired_config_suite_rows=[],
                    cross_query_chunk_modes=args.cross_query_chunk_modes,
                    warnings=warnings,
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
        (args.output_dir / 'latex' / 'txt_experiments_config_recap.md').write_text(
            render_experiment_config_recap(plot_and_recap_records)
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
                    'artifact_version': args.artifact_version,
                    'suite_id': args.suite_id,
                    'suite_where': args.suite_where,
                    'strict_suite': args.strict_suite,
                    'main_query_scope': args.main_query_scope,
                    'experiments_discovered': discovered_count,
                    'experiments_after_embedding_filter': len(records),
                    'cross_query_chunk_modes': args.cross_query_chunk_modes,
                    'wording_configurations': wording_configurations,
                    'cross_triplet_analysis_enabled': cross_triplet_analysis_enabled,
                    'cross_triplet_figures_enabled': args.plots and cross_triplet_analysis_enabled,
                    'optional_analyses': {
                        'lambda': lambda_analysis_enabled,
                        'global_lambda': global_lambda_analysis_enabled,
                        'lodo': lodo_analysis_enabled,
                        'paired_statistics': paired_statistics_enabled,
                        'validity': validity_analysis_enabled,
                    },
                    'warnings_count': len(warnings),
                    'figures': [str(path.relative_to(args.output_dir)) for path in figures],
                    'files': [*generated_files, 'manifest.json'],
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
                    'suite_outputs': {
                        'distribution_rows': len(suite_distribution_rows),
                        'family_rows': len(suite_family_rows),
                        'matched_contrasts': len(suite_contrast_rows),
                        'factor_interactions': len(suite_interaction_rows),
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
    geometry_rows = _read_report_csv_rows(data_dir, 'geometry_filter_summary.csv')
    metric_family_summary_rows_data = _read_report_csv_rows(data_dir, 'metric_family_summary.csv')
    metric_family_budget_summary_rows_data = _read_report_csv_rows(
        data_dir, 'metric_family_budget_summary.csv'
    )
    metric_summary_rows = _read_report_csv_rows(data_dir, 'metric_aggregate_summary.csv')
    budget_rows = _read_report_csv_rows(data_dir, 'budget_strategy_summary.csv', required=True)
    interaction_budget_rows = _read_report_csv_rows(
        data_dir, 'interaction_budget_strategy_summary.csv'
    )
    interaction_metric_family_summary_rows = _read_report_csv_rows(
        data_dir, 'interaction_metric_family_summary.csv'
    )
    interaction_metric_family_budget_summary_rows = _read_report_csv_rows(
        data_dir, 'interaction_metric_family_budget_summary.csv'
    )
    interaction_metric_summary_rows = _read_report_csv_rows(
        data_dir, 'interaction_metric_aggregate_summary.csv'
    )
    lambda_rows = _read_report_csv_rows(data_dir, 'lambda_stability.csv')
    lambda_grid_delta_rows = _read_report_csv_rows(data_dir, 'lambda_grid_fcp_delta.csv')
    lambda_safety_rows = _read_report_csv_rows(data_dir, 'lambda_safety_summary.csv')
    near_optimal_rows = _read_report_csv_rows(data_dir, 'near_optimal_lambda_width.csv')
    paired_cell_rows = _read_report_csv_rows(data_dir, 'paired_cell_effect_summary.csv')
    paired_suite_rows = _read_report_csv_rows(data_dir, 'paired_suite_effect_summary.csv')
    paired_config_suite_rows = _read_report_csv_rows(
        data_dir, 'paired_config_suite_effect_summary.csv'
    )
    suite_contrast_rows = _read_report_csv_rows(data_dir, 'suite_matched_contrasts.csv')
    _progress('rendering report figures from existing CSV artifacts')
    figures = write_figures(
        output_dir=report_dir / 'figures',
        plot_format=args.plot_format,
        max_rows=args.max_table_rows,
        budget_rows=budget_rows,
        geometry_rows=primary_rows(geometry_rows),
        lambda_rows=primary_rows(lambda_rows),
        lambda_grid_delta_rows=primary_rows(lambda_grid_delta_rows),
        lambda_safety_rows=primary_rows(lambda_safety_rows),
        near_optimal_rows=primary_rows(near_optimal_rows),
        dataset_rows=primary_rows(dataset_rows),
        metric_summary_rows=metric_summary_rows,
        metric_family_summary_rows=metric_family_summary_rows_data,
        metric_family_budget_summary_rows=metric_family_budget_summary_rows_data,
        paired_cell_rows=primary_rows(paired_cell_rows),
        paired_suite_rows=paired_suite_rows,
        paired_config_suite_rows=paired_config_suite_rows,
        cross_query_chunk_modes=args.cross_query_chunk_modes,
        warnings=warnings,
    )
    figures.extend(
        write_figures(
            output_dir=report_dir / 'figures' / 'interactions',
            plot_format=args.plot_format,
            max_rows=args.max_table_rows,
            budget_rows=interaction_budget_rows,
            geometry_rows=interaction_rows(geometry_rows),
            lambda_rows=interaction_rows(lambda_rows),
            lambda_grid_delta_rows=interaction_rows(lambda_grid_delta_rows),
            lambda_safety_rows=interaction_rows(lambda_safety_rows),
            near_optimal_rows=interaction_rows(near_optimal_rows),
            dataset_rows=interaction_rows(dataset_rows),
            metric_summary_rows=interaction_metric_summary_rows,
            metric_family_summary_rows=interaction_metric_family_summary_rows,
            metric_family_budget_summary_rows=interaction_metric_family_budget_summary_rows,
            paired_cell_rows=interaction_rows(paired_cell_rows),
            paired_suite_rows=[
                row for row in paired_suite_rows if row.get('Scope') == 'Interaction experiments'
            ],
            paired_config_suite_rows=[],
            cross_query_chunk_modes=args.cross_query_chunk_modes,
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
    synthetic_artifact_rows = _read_report_csv_rows(data_dir, 'synthetic_artifact_diagnostics.csv')
    metric_summary_rows = _read_report_csv_rows(data_dir, 'metric_aggregate_summary.csv')
    metric_family_summary_rows_data = _read_report_csv_rows(data_dir, 'metric_family_summary.csv')
    paired_suite_rows = _read_report_csv_rows(data_dir, 'paired_suite_effect_summary.csv')
    embedding_summary_rows = _read_report_csv_rows(data_dir, 'embedding_model_summary.csv')
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
            synthetic_artifact_rows=synthetic_artifact_rows,
            metric_summary_rows=metric_summary_rows,
            metric_family_summary_rows=metric_family_summary_rows_data,
            paired_suite_rows=paired_suite_rows,
            embedding_summary_rows=embedding_summary_rows,
            embedding_models=_effective_embedding_models_for_rows(budget_rows),
            require_complete_wording_grid=args.cross_query_chunk_modes,
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
    synthetic_artifact_rows: Sequence[Mapping[str, object]],
    metric_summary_rows: Sequence[Mapping[str, object]],
    metric_family_summary_rows: Sequence[Mapping[str, object]],
    metric_family_budget_summary_rows: Sequence[Mapping[str, object]],
    paired_suite_rows: Sequence[Mapping[str, object]],
    embedding_summary_rows: Sequence[Mapping[str, object]],
    embedding_models: Sequence[str],
    require_complete_wording_grid: bool,
    warnings: list[str],
    paired_statistics: bool,
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
            synthetic_artifact_rows=synthetic_artifact_rows,
            metric_summary_rows=metric_summary_rows,
            metric_family_summary_rows=metric_family_summary_rows,
            paired_suite_rows=paired_suite_rows,
            embedding_summary_rows=embedding_summary_rows,
            embedding_models=embedding_models,
            require_complete_wording_grid=require_complete_wording_grid,
            warnings=warnings,
        )
    )
    statistical_tables_path = thesis_statistical_tables_path(output_dir)
    if paired_statistics:
        statistical_tables_path.write_text(render_statistical_latex_table(paired_suite_rows))
    else:
        statistical_tables_path.unlink(missing_ok=True)


def main() -> None:
    outputs = run_report(parse_args())
    print(f'wrote report files to {outputs.output_dir}')
    print(
        f'experiments: {outputs.experiments_loaded}/{outputs.experiments_discovered} configs loaded; '
        f'warnings: {outputs.warnings_count}; figures: {outputs.figures_count}'
    )


if __name__ == '__main__':
    main()
