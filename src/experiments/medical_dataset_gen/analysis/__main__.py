from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from experiments.medical_dataset_gen.analysis.analysis_constants import REPORT_FILES
from experiments.medical_dataset_gen.analysis.artifacts import (
    render_experiment_config_recap,
    write_csv,
)
from experiments.medical_dataset_gen.analysis.cli import parse_args
from experiments.medical_dataset_gen.analysis.discovery import discover_experiments
from experiments.medical_dataset_gen.analysis.latex_tables import (
    THESIS_AGGREGATE_TABLES_FILENAME,
    THESIS_RESULT_MACROS_FILENAME,
    render_thesis_aggregate_tables,
    render_thesis_result_macros,
)
from experiments.medical_dataset_gen.analysis.models import CliArgs, ExperimentRecord, ReportOutputs
from experiments.medical_dataset_gen.analysis.plots import write_figures
from experiments.medical_dataset_gen.analysis.rendering import (
    render_interesting_findings,
    render_report,
)
from experiments.medical_dataset_gen.analysis.report_config import LEGACY_LOW_BUDGET_TOKEN
from experiments.medical_dataset_gen.analysis.rows import (
    dataset_distribution_row,
    experiment_manifest_row,
    geometry_filter_row,
    lambda_grid_fcp_delta_rows,
    lambda_safety_summary_rows,
    near_optimal_lambda_rows,
    selected_strategy_rows,
)
from experiments.medical_dataset_gen.analysis.summaries import (
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
from experiments.medical_dataset_gen.evaluation.lambda_selection import (
    LAMBDA_SELECTION_MAXIMIZING_METRIC,
)
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths


def run_report(args: CliArgs) -> ReportOutputs:
    old_results_dir = MedicalDatasetGenPaths.results_dir
    MedicalDatasetGenPaths.results_dir = args.results_dir
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        plot_and_recap_records = discover_experiments(
            args.results_dir,
            include_scrapped=args.include_scrapped,
            requested_experiments=args.experiments,
            warnings=warnings,
            include_all_query=True,
        )
        records = [
            record for record in plot_and_recap_records if record.only_pass_geometry is not False
        ]

        manifest_rows = [experiment_manifest_row(record) for record in records]
        dataset_rows = [dataset_distribution_row(record, warnings=warnings) for record in records]
        geometry_rows = [geometry_filter_row(record, warnings=warnings) for record in records]

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

        lambda_grid_delta_rows = lambda_grid_fcp_delta_rows(records, warnings=warnings)
        lambda_safety_rows = lambda_safety_summary_rows(lambda_grid_delta_rows)
        comparison_rows = comparison_by_k_rows(strategy_rows)
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

        (args.output_dir / f'{LEGACY_LOW_BUDGET_TOKEN}_strategy_summary.csv').unlink(
            missing_ok=True
        )
        shutil.rmtree(args.output_dir / 'data', ignore_errors=True)
        write_csv(args.output_dir / 'experiment_manifest.csv', manifest_rows)
        write_csv(args.output_dir / 'dataset_distribution.csv', dataset_rows)
        write_csv(args.output_dir / 'geometry_filter_summary.csv', geometry_rows)
        write_csv(args.output_dir / 'strategy_by_k.csv', strategy_rows)
        write_csv(args.output_dir / 'comparison_by_k.csv', comparison_rows)
        write_csv(args.output_dir / 'experiment_family_summary.csv', family_summary_rows)
        write_csv(
            args.output_dir / 'experiment_family_budget_summary.csv',
            family_budget_summary_rows,
        )
        write_csv(args.output_dir / 'metric_family_summary.csv', metric_family_summary_rows_data)
        write_csv(
            args.output_dir / 'metric_family_budget_summary.csv',
            metric_family_budget_summary_rows_data,
        )
        write_csv(args.output_dir / 'metric_aggregate_summary.csv', metric_summary_rows)
        write_csv(args.output_dir / 'budget_strategy_summary.csv', budget_rows)
        write_csv(args.output_dir / 'low_budget_strategy_summary.csv', low_budget_rows)
        write_csv(args.output_dir / 'lambda_stability.csv', lambda_rows)
        write_csv(args.output_dir / 'lambda_grid_fcp_delta.csv', lambda_grid_delta_rows)
        write_csv(args.output_dir / 'lambda_safety_summary.csv', lambda_safety_rows)
        write_csv(args.output_dir / 'near_optimal_lambda_width.csv', near_optimal_rows)
        write_csv(args.output_dir / 'embedding_model_summary.csv', embedding_summary_rows)
        (args.output_dir / THESIS_AGGREGATE_TABLES_FILENAME).write_text(
            render_thesis_aggregate_tables(
                metric_summary_rows=metric_summary_rows,
                metric_family_summary_rows=metric_family_summary_rows_data,
                metric_family_budget_summary_rows=metric_family_budget_summary_rows_data,
            )
        )
        (args.output_dir / THESIS_RESULT_MACROS_FILENAME).write_text(
            render_thesis_result_macros(
                geometry_rows=geometry_rows,
                comparison_rows=comparison_rows,
                budget_rows=budget_rows,
                lambda_safety_rows=lambda_safety_rows,
            )
        )

        figures: list[Path] = []
        if args.plots:
            plot_budget_rows = _plot_budget_rows(
                records=plot_and_recap_records,
                base_records=records,
                base_budget_rows=budget_rows,
                warnings=warnings,
            )
            figures = write_figures(
                output_dir=args.output_dir / '_figures',
                plot_format=args.plot_format,
                max_rows=args.max_table_rows,
                budget_rows=plot_budget_rows,
                geometry_rows=geometry_rows,
                lambda_rows=lambda_rows,
                lambda_grid_delta_rows=lambda_grid_delta_rows,
                lambda_safety_rows=lambda_safety_rows,
                near_optimal_rows=near_optimal_rows,
                dataset_rows=dataset_rows,
                metric_summary_rows=metric_summary_rows,
                metric_family_summary_rows=metric_family_summary_rows_data,
                metric_family_budget_summary_rows=metric_family_budget_summary_rows_data,
                warnings=warnings,
            )

        report_text = render_report(
            args=args,
            records=records,
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
                    'experiments_discovered': len(records),
                    'warnings_count': len(warnings),
                    'figures': [str(path.relative_to(args.output_dir)) for path in figures],
                    'files': list(REPORT_FILES),
                    'lambda_selection_metric': LAMBDA_SELECTION_MAXIMIZING_METRIC,
                    'near_optimal_epsilon': args.near_optimal_epsilon,
                },
                indent=2,
                sort_keys=True,
            )
            + '\n'
        )
        return ReportOutputs(
            output_dir=args.output_dir,
            experiments_discovered=len(records),
            experiments_loaded=sum(1 for record in records if record.cfg is not None),
            warnings_count=len(warnings),
            figures_count=len(figures),
        )
    finally:
        MedicalDatasetGenPaths.results_dir = old_results_dir


def _plot_budget_rows(
    *,
    records: list[ExperimentRecord],
    base_records: list[ExperimentRecord],
    base_budget_rows: list[dict[str, object]],
    warnings: list[str],
) -> list[dict[str, object]]:
    base_names = {str(record.name) for record in base_records}
    extra_records = [record for record in records if str(record.name) not in base_names]
    if not extra_records:
        return base_budget_rows

    extra_strategy_rows: list[dict[str, object]] = []
    for record in extra_records:
        extra_strategy_rows.extend(selected_strategy_rows(record, warnings=warnings))
    extra_comparison_rows = comparison_by_k_rows(extra_strategy_rows)
    return [
        *base_budget_rows,
        *budget_category_rows_from_comparisons(extra_comparison_rows),
    ]


def main() -> None:
    outputs = run_report(parse_args())
    print(f'wrote report files to {outputs.output_dir}')
    print(
        f'experiments: {outputs.experiments_loaded}/{outputs.experiments_discovered} configs loaded; '
        f'warnings: {outputs.warnings_count}; figures: {outputs.figures_count}'
    )


if __name__ == '__main__':
    main()
