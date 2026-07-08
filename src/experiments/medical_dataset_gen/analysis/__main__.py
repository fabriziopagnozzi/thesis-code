from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast, runtime_checkable

import polars as pl
import yaml
from tabulate import tabulate

from experiments.medical_dataset_gen.analysis.analysis_constants import (
    DEFAULT_TABLE_COL_WIDTH,
    DIVERSIFYING_STRATEGIES,
    EVALUATION_METRICS,
    EXPERIMENT_FAMILIES,
    EXPERIMENT_FAMILY_COLORS,
    EXPERIMENT_FAMILY_LABELS,
    FCP_TIE_EPSILON,
    HELDOUT_SELECTION_COLUMNS,
    INTEGER_TABLE_COLUMNS,
    METRIC_LABELS,
    REPORT_FILES,
    ROLE_COUNT_COLUMNS,
    STRATEGIES,
    TABLE_COL_WIDTHS,
    TABLE_HEADERS,
    TABLEFMT_OPTS,
    ExperimentFamilyId,
    StrategyName,
)
from experiments.medical_dataset_gen.evaluation.lambda_selection import (
    LAMBDA_SELECTION_MAXIMIZING_METRIC,
    select_best_lambda_rows,
)
from experiments.medical_dataset_gen.schemas.global_config_schemas import (
    ExperimentCfg,
    LambdaSelectionCfg,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    _load_raw_experiment_config,
    child_experiment_names,
    load_config,
    resolve_experiment_name,
)

type PlotFormat = Literal['png', 'pdf', 'svg']


@runtime_checkable
class ScalarItem(Protocol):
    def item(self) -> object: ...


@dataclass(frozen=True)
class CliArgs:
    results_dir: Path
    output_dir: Path
    include_scrapped: bool
    experiments: tuple[str, ...]
    max_table_rows: int
    tablefmt: str
    plots: bool
    plot_format: PlotFormat
    near_optimal_epsilon: float


@dataclass(frozen=True)
class ExperimentRecord:
    name: str
    experiment_dir: Path
    distribution_id: str
    run_label: str
    is_subexperiment: bool
    cfg: ExperimentCfg | None
    paths: MedicalDatasetGenPaths
    config_error: str | None
    family_id: ExperimentFamilyId
    family_label: str

    @property
    def embedding_model(self) -> str:
        if self.cfg is None:
            return 'unknown'
        return str(self.cfg.embeddings.model_name)

    @property
    def only_pass_geometry(self) -> bool | None:
        if self.cfg is None:
            return None
        return bool(self.cfg.retrieval.only_pass_geometry)


@dataclass(frozen=True)
class ReportOutputs:
    output_dir: Path
    experiments_discovered: int
    experiments_loaded: int
    warnings_count: int
    figures_count: int


def run_report(args: CliArgs) -> ReportOutputs:
    old_results_dir = MedicalDatasetGenPaths.results_dir
    MedicalDatasetGenPaths.results_dir = args.results_dir
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        records = discover_experiments(
            args.results_dir,
            include_scrapped=args.include_scrapped,
            requested_experiments=args.experiments,
            warnings=warnings,
        )

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

        comparison_rows = comparison_by_k_rows(strategy_rows)
        headline_rows = headline_rows_from_comparisons(comparison_rows)
        lambda_rows = lambda_stability_rows(strategy_rows, near_optimal_rows)
        embedding_summary_rows = embedding_model_summary_rows(
            manifest_rows=manifest_rows,
            geometry_rows=geometry_rows,
            headline_rows=headline_rows,
        )
        embedding_pair_rows = embedding_query_scope_pair_rows(headline_rows)

        write_csv(args.output_dir / 'experiment_manifest.csv', manifest_rows)
        write_csv(args.output_dir / 'dataset_distribution.csv', dataset_rows)
        write_csv(args.output_dir / 'geometry_filter_summary.csv', geometry_rows)
        write_csv(args.output_dir / 'strategy_by_k.csv', strategy_rows)
        write_csv(args.output_dir / 'comparison_by_k.csv', comparison_rows)
        write_csv(args.output_dir / 'headline_strategy_summary.csv', headline_rows)
        write_csv(args.output_dir / 'lambda_stability.csv', lambda_rows)
        write_csv(args.output_dir / 'near_optimal_lambda_width.csv', near_optimal_rows)
        write_csv(args.output_dir / 'embedding_model_summary.csv', embedding_summary_rows)
        write_csv(args.output_dir / 'embedding_query_scope_pairs.csv', embedding_pair_rows)

        figures: list[Path] = []
        if args.plots:
            figures = write_figures(
                output_dir=args.output_dir / '_figures',
                plot_format=args.plot_format,
                max_rows=args.max_table_rows,
                headline_rows=headline_rows,
                embedding_pair_rows=embedding_pair_rows,
                geometry_rows=geometry_rows,
                lambda_rows=lambda_rows,
                near_optimal_rows=near_optimal_rows,
                dataset_rows=dataset_rows,
                warnings=warnings,
            )

        report_text = render_report(
            args=args,
            records=records,
            dataset_rows=dataset_rows,
            geometry_rows=geometry_rows,
            comparison_rows=comparison_rows,
            headline_rows=headline_rows,
            lambda_rows=lambda_rows,
            embedding_summary_rows=embedding_summary_rows,
            embedding_pair_rows=embedding_pair_rows,
            figures=figures,
        )
        (args.output_dir / 'report.md').write_text(report_text)
        (args.output_dir / 'report_interesting_findings.md').write_text(
            render_interesting_findings(
                comparison_rows=comparison_rows,
                headline_rows=headline_rows,
                geometry_rows=geometry_rows,
                lambda_rows=lambda_rows,
                embedding_summary_rows=embedding_summary_rows,
                embedding_pair_rows=embedding_pair_rows,
                tablefmt=args.tablefmt,
                max_table_rows=args.max_table_rows,
            )
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


def discover_experiments(
    results_dir: Path,
    *,
    include_scrapped: bool,
    requested_experiments: Sequence[str],
    warnings: list[str],
) -> list[ExperimentRecord]:
    candidate_names = (
        _requested_experiment_names(results_dir, requested_experiments, warnings)
        if requested_experiments
        else _artifact_experiment_names(results_dir)
    )
    records: list[ExperimentRecord] = []
    seen: set[str] = set()
    for name in candidate_names:
        if name in seen:
            continue
        seen.add(name)
        parts = Path(name).parts
        if not parts:
            continue
        if parts[0].startswith('_'):
            continue
        if parts[0] == '00_scrapped' and not include_scrapped:
            continue
        if len(parts) > 2:
            warnings.append(f'{name}: skipped because subexperiments support only one child level')
            continue
        stats_path = results_dir / name / 'evaluation_stats.parquet'
        if not stats_path.is_file():
            warnings.append(f'{name}: skipped because evaluation_stats.parquet is missing')
            continue
        records.append(load_experiment_record(results_dir, name, warnings=warnings))
    return sorted(records, key=lambda record: record.name)


def _artifact_experiment_names(results_dir: Path) -> list[str]:
    return sorted(
        path.parent.relative_to(results_dir).as_posix()
        for path in results_dir.glob('**/evaluation_stats.parquet')
        if path.is_file()
    )


def _requested_experiment_names(
    results_dir: Path,
    requested_experiments: Sequence[str],
    warnings: list[str],
) -> list[str]:
    names: list[str] = []
    for raw_name in requested_experiments:
        try:
            resolved = resolve_experiment_name(raw_name, results_dir=results_dir)
        except Exception as exc:
            warnings.append(f'{raw_name}: could not resolve requested experiment ({exc})')
            continue
        if (results_dir / resolved / 'evaluation_stats.parquet').is_file():
            names.append(resolved)
            continue
        children = [
            child
            for child in child_experiment_names(resolved, results_dir=results_dir)
            if (results_dir / child / 'evaluation_stats.parquet').is_file()
        ]
        if children:
            names.extend(children)
        else:
            warnings.append(f'{resolved}: requested experiment has no completed eval artifact')
    return names


def load_experiment_record(
    results_dir: Path,
    name: str,
    *,
    warnings: list[str],
) -> ExperimentRecord:
    cfg: ExperimentCfg | None = None
    config_error: str | None = None
    try:
        cfg = load_config(name)
    except Exception as exc:
        try:
            cfg = _load_config_with_report_compatibility(name)
        except Exception:
            config_error = str(exc)
            warnings.append(
                f'{name}: config could not be loaded, using local artifacts only ({exc})'
            )

    paths = MedicalDatasetGenPaths(
        name,
        result_dir_overrides=cfg.global_.result_dir_overrides if cfg is not None else None,
    )
    parts = Path(name).parts
    is_subexperiment = len(parts) == 2
    distribution_id = parts[0]
    run_label = parts[1] if is_subexperiment else 'parent'
    family_id, family_label = _load_experiment_family(
        results_dir=results_dir,
        name=name,
        distribution_id=distribution_id,
        warnings=warnings,
    )
    return ExperimentRecord(
        name=name,
        experiment_dir=results_dir / name,
        distribution_id=distribution_id,
        run_label=run_label,
        is_subexperiment=is_subexperiment,
        cfg=cfg,
        paths=paths,
        config_error=config_error,
        family_id=family_id,
        family_label=family_label,
    )


def _load_experiment_family(
    *,
    results_dir: Path,
    name: str,
    distribution_id: str,
    warnings: list[str],
) -> tuple[ExperimentFamilyId, str]:
    candidate_paths = [
        results_dir / name / '_exp_family.yaml',
        results_dir / distribution_id / '_exp_family.yaml',
    ]
    for path in candidate_paths:
        if not path.is_file():
            continue
        try:
            raw: object = yaml.safe_load(path.read_text()) or {}
        except Exception as exc:
            warnings.append(f'{name}: could not read experiment family metadata at {path} ({exc})')
            return 'unknown', EXPERIMENT_FAMILY_LABELS['unknown']
        if not isinstance(raw, Mapping):
            warnings.append(f'{name}: experiment family metadata at {path} is not a mapping')
            return 'unknown', EXPERIMENT_FAMILY_LABELS['unknown']
        family_id = raw.get('family_id')
        if not isinstance(family_id, str) or family_id not in EXPERIMENT_FAMILIES:
            warnings.append(
                f'{name}: experiment family metadata at {path} has invalid family_id {family_id!r}'
            )
            return 'unknown', EXPERIMENT_FAMILY_LABELS['unknown']
        typed_family_id = cast(ExperimentFamilyId, family_id)
        family_label = raw.get('family_label')
        return (
            typed_family_id,
            family_label
            if isinstance(family_label, str)
            else EXPERIMENT_FAMILY_LABELS[typed_family_id],
        )
    return 'unknown', EXPERIMENT_FAMILY_LABELS['unknown']


def _load_config_with_report_compatibility(exp_name: str) -> ExperimentCfg:
    paths = MedicalDatasetGenPaths(exp_name)
    raw = _load_raw_experiment_config(paths)
    evaluation = raw.get('evaluation')
    if isinstance(evaluation, Mapping):
        sanitized_evaluation = dict(evaluation)
        sanitized_evaluation.pop('fac_loc_mmr_comparison_kernels', None)
        raw = {**raw, 'evaluation': sanitized_evaluation}
    cfg = ExperimentCfg.model_validate(raw)
    cfg.global_.output_experiment = exp_name
    return cfg


def experiment_manifest_row(record: ExperimentRecord) -> dict[str, object]:
    metadata = embedding_metadata(record)
    return {
        'Experiment': record.name,
        'ShortExperiment': short_experiment_id(record.name),
        'Distribution': record.distribution_id,
        'ShortDistribution': short_token(record.distribution_id),
        'ExperimentFamily': record.family_id,
        'ExperimentFamilyLabel': record.family_label,
        'RunLabel': record.run_label,
        'IsSubexperiment': record.is_subexperiment,
        'ConfigLoaded': record.cfg is not None,
        'ConfigError': record.config_error,
        'EmbeddingModel': metadata.get('model_name') or record.embedding_model,
        'EmbeddingDimension': metadata.get('dimension'),
        'EmbeddingChunks': metadata.get('n_chunks'),
        'EmbeddingQueries': metadata.get('n_queries'),
        'OnlyPassGeometry': record.only_pass_geometry,
        'QueryScope': _query_scope_label(record.only_pass_geometry),
        'CandidatePoolN': record.cfg.retrieval.candidate_pool_n if record.cfg else None,
        'KValues': ','.join(str(k) for k in record.cfg.retrieval.k_values) if record.cfg else None,
        'EvaluationMode': record.cfg.evaluation.mode if record.cfg else None,
        'EvaluationStatsPath': str(record.paths.table_path('evaluation_stats')),
        'QrelsPath': str(record.paths.table_path('qrels')),
        'GeometryStatsPath': str(record.paths.table_path('geometry_stats')),
    }


def dataset_distribution_row(
    record: ExperimentRecord,
    *,
    warnings: list[str],
) -> dict[str, object]:
    qrels_path = record.paths.table_path('qrels')
    base = _base_experiment_row(record)
    cfg = record.cfg
    base.update({
        'QrelsPath': str(qrels_path),
        'ConfiguredGoldChunksPerQuery': cfg.generation.total_gold_chunks() if cfg else None,
        'ConfiguredDistractorChunksPerQuery': cfg.generation.total_distractor_chunks()
        if cfg
        else None,
        'ConfiguredBackgroundOutlierChunksPerQuery': (
            cfg.generation.chunk_pools.background_outlier_chunks_per_query() if cfg else None
        ),
        'ConfiguredNicheClustersPerQuery': (
            cfg.generation.chunk_pools.niche.num_clusters_per_query if cfg else None
        ),
    })
    if not qrels_path.is_file():
        warnings.append(f'{record.name}: qrels missing at {qrels_path}')
        return base

    required_columns = ['query_id', 'cluster_role', 'facet_id', 'is_gold', 'distractor_type']
    try:
        qrels = pl.read_parquet(qrels_path, columns=required_columns)
    except Exception as exc:
        warnings.append(f'{record.name}: could not read qrels ({exc})')
        return base

    if qrels.is_empty():
        warnings.append(f'{record.name}: qrels is empty')
        return base

    cluster_role = pl.col('cluster_role').fill_null('')
    is_gold = pl.col('is_gold').fill_null(False)
    per_query_exprs = [
        pl.len().alias('PoolSize'),
        is_gold.sum().alias('GoldCount'),
        ((~is_gold) & (cluster_role != 'background_outlier'))
        .sum()
        .alias('NearMissDistractorCount'),
        (cluster_role == 'background_outlier').sum().alias('BackgroundOutlierCount'),
        pl.col('facet_id').filter(is_gold).n_unique().alias('GoldFacetCount'),
    ]
    for role, column in ROLE_COUNT_COLUMNS.items():
        per_query_exprs.append((cluster_role == role).sum().alias(column))

    per_query = qrels.group_by('query_id').agg(per_query_exprs)
    summary = _mean_min_max_for_columns(
        per_query,
        columns=[
            'PoolSize',
            'GoldCount',
            'NearMissDistractorCount',
            'BackgroundOutlierCount',
            'GoldFacetCount',
            *ROLE_COUNT_COLUMNS.values(),
        ],
    )
    pool_size = _float_or_none(summary.get('PoolSizeMean'))
    gold_count = _float_or_none(summary.get('GoldCountMean'))
    near_miss_count = _float_or_none(summary.get('NearMissDistractorCountMean'))
    background_count = _float_or_none(summary.get('BackgroundOutlierCountMean'))

    base.update(summary)
    base.update({
        'QueriesInQrels': per_query.height,
        'QrelRows': qrels.height,
        'GoldPercentage': _ratio(gold_count, pool_size),
        'NearMissDistractorPercentage': _ratio(near_miss_count, pool_size),
        'BackgroundOutlierPercentage': _ratio(background_count, pool_size),
        'PrimaryDominanceRatio': _primary_dominance_ratio(summary),
        'DistributionCategory': _distribution_category(summary),
    })
    return base


def geometry_filter_row(
    record: ExperimentRecord,
    *,
    warnings: list[str],
) -> dict[str, object]:
    geometry_path = record.paths.table_path('geometry_stats')
    base = _base_experiment_row(record)
    base.update({'GeometryStatsPath': str(geometry_path)})
    if not geometry_path.is_file():
        warnings.append(f'{record.name}: geometry_stats missing at {geometry_path}')
        return base

    try:
        geometry = pl.read_parquet(geometry_path)
    except Exception as exc:
        warnings.append(f'{record.name}: could not read geometry_stats ({exc})')
        return base

    if geometry.is_empty():
        warnings.append(f'{record.name}: geometry_stats is empty')
        return base

    columns = set(geometry.columns)
    base['GeometryQueries'] = geometry.height
    if 'passes_filter' in columns:
        pass_count = int(geometry.select(pl.col('passes_filter').fill_null(False).sum()).item())
        base['GeometryPassQueries'] = pass_count
        base['GeometryPassRate'] = pass_count / geometry.height if geometry.height else None

    for column in (
        'pool_size',
        'n_distractors_in_pool',
        'n_near_miss_distractors_in_pool',
        'n_background_outliers_in_pool',
        'n_background_outlier_clusters_in_pool',
        'n_topk_retrieved_facets',
        'primary_axis_topk_fraction',
        'dominant_primary_topk_fraction',
        'fac_topk',
        'fac_facloc',
        'avg_cos_topk',
        'avg_cos_facloc',
        'jaccard_topk_facloc',
    ):
        if column in columns:
            base[f'{_title_token(column)}Mean'] = _series_mean(geometry[column])

    failure_rates: list[tuple[str, float]] = []
    for column in sorted(col for col in columns if col.startswith('fail_')):
        rate = _series_mean(geometry[column].fill_null(False))
        if rate is not None:
            base[f'{_title_token(column)}Rate'] = rate
            failure_rates.append((column.removeprefix('fail_'), rate))
    failure_rates.sort(key=lambda item: item[1], reverse=True)
    base['TopFailureModes'] = ', '.join(
        f'{name}:{rate:.3f}' for name, rate in failure_rates[:3] if rate > 0.0
    )
    return base


def selected_strategy_rows(
    record: ExperimentRecord,
    *,
    warnings: list[str],
) -> list[dict[str, object]]:
    stats_path = record.paths.table_path('evaluation_stats')
    if not stats_path.is_file():
        warnings.append(f'{record.name}: evaluation_stats missing at {stats_path}')
        return []

    try:
        stats = pl.read_parquet(stats_path)
    except Exception as exc:
        warnings.append(f'{record.name}: could not read evaluation_stats ({exc})')
        return []

    if stats.is_empty():
        warnings.append(f'{record.name}: evaluation_stats is empty')
        return []
    if LAMBDA_SELECTION_MAXIMIZING_METRIC not in stats.columns:
        warnings.append(
            f'{record.name}: evaluation_stats lacks {LAMBDA_SELECTION_MAXIMIZING_METRIC}'
        )
        return []

    selected, source = _selected_stats_frame(record, stats, warnings=warnings)
    rows: list[dict[str, object]] = []
    for row in selected.iter_rows(named=True):
        strategy = cast(StrategyName, row.get('strategy'))
        lam = _float_or_none(row.get('lam'))
        out = _base_experiment_row(record)
        out.update({
            'SelectionSource': source,
            'strategy': strategy,
            'k': _int_or_none(row.get('k')),
            'lam': lam,
            'lambda_norm': _lambda_norm(record, strategy, lam, stats),
        })
        for metric in EVALUATION_METRICS:
            if metric in row:
                out[metric] = _json_scalar(row[metric])
        rows.append(out)
    return rows


def _selected_stats_frame(
    record: ExperimentRecord,
    stats: pl.DataFrame,
    *,
    warnings: list[str],
) -> tuple[pl.DataFrame, str]:
    if HELDOUT_SELECTION_COLUMNS & set(stats.columns):
        return stats.sort([col for col in ('k', 'strategy', 'lam') if col in stats.columns]), (
            'heldout_selected'
        )

    counts = stats.group_by(['strategy', 'k']).agg(pl.len().alias('n_rows'))
    max_rows_per_group = _int_or_none(counts['n_rows'].max()) or 0
    if max_rows_per_group <= 1:
        return stats.sort([col for col in ('k', 'strategy', 'lam') if col in stats.columns]), (
            'preselected'
        )

    k_values = sorted(int(value) for value in stats['k'].drop_nulls().unique().to_list())
    lambda_cfg = record.cfg.evaluation.lambda_selection if record.cfg else LambdaSelectionCfg()
    frames: list[pl.DataFrame] = []
    top_k = stats.filter(pl.col('strategy') == 'top_k')
    for k in k_values:
        top_k_row = top_k.filter(pl.col('k') == k).head(1)
        if not top_k_row.is_empty():
            frames.append(top_k_row)
    for strategy in DIVERSIFYING_STRATEGIES:
        selected = select_best_lambda_rows(
            stats,
            strategy=strategy,
            k_values=k_values,
            cfg=lambda_cfg,
        )
        if selected.is_empty():
            warnings.append(f'{record.name}: no selected rows for strategy={strategy}')
            continue
        frames.append(selected)
    if not frames:
        return pl.DataFrame(), 'posthoc_selected'
    return pl.concat(frames).sort([
        col for col in ('k', 'strategy', 'lam') if col in stats.columns
    ]), ('posthoc_selected')


def near_optimal_lambda_rows(
    record: ExperimentRecord,
    *,
    epsilon: float,
    warnings: list[str],
) -> list[dict[str, object]]:
    grid_path = _lambda_grid_stats_path(record)
    if not grid_path.is_file():
        return []
    try:
        stats = pl.read_parquet(grid_path)
    except Exception as exc:
        warnings.append(f'{record.name}: could not read lambda grid stats ({exc})')
        return []
    if stats.is_empty() or LAMBDA_SELECTION_MAXIMIZING_METRIC not in stats.columns:
        return []

    rows: list[dict[str, object]] = []
    k_values = sorted(int(value) for value in stats['k'].drop_nulls().unique().to_list())
    for strategy in DIVERSIFYING_STRATEGIES:
        strategy_df = stats.filter(pl.col('strategy') == strategy)
        for k in k_values:
            sub = strategy_df.filter(pl.col('k') == k).drop_nulls(
                subset=['lam', LAMBDA_SELECTION_MAXIMIZING_METRIC]
            )
            if sub.height <= 1:
                continue
            fcp_values = [
                float(value)
                for value in sub[LAMBDA_SELECTION_MAXIMIZING_METRIC].drop_nulls().to_list()
            ]
            lambdas = [float(value) for value in sub['lam'].drop_nulls().to_list()]
            if not fcp_values or not lambdas:
                continue
            best_fcp = max(fcp_values)
            worst_fcp = min(fcp_values)
            threshold = best_fcp - epsilon
            near = sub.filter(pl.col(LAMBDA_SELECTION_MAXIMIZING_METRIC) >= threshold)
            near_lambdas = [float(value) for value in near['lam'].drop_nulls().to_list()]
            full_span = max(lambdas) - min(lambdas)
            near_span = max(near_lambdas) - min(near_lambdas) if near_lambdas else 0.0
            out = _base_experiment_row(record)
            out.update({
                'strategy': strategy,
                'k': k,
                'GridStatsPath': str(grid_path),
                'NearOptimalEpsilon': epsilon,
                'BestFCP': best_fcp,
                'WorstFCP': worst_fcp,
                'FCPRange': best_fcp - worst_fcp,
                'NearOptimalLambdaCount': near.height,
                'TotalLambdaCount': sub.height,
                'NearOptimalLambdaFraction': near.height / sub.height,
                'NearOptimalLambdaSpan': near_span,
                'NearOptimalLambdaSpanNorm': near_span / full_span if full_span else 0.0,
            })
            rows.append(out)
    return rows


def _lambda_grid_stats_path(record: ExperimentRecord) -> Path:
    report_grid_path = record.paths.table_path('evaluation_report_grid_stats')
    if report_grid_path.is_file():
        return report_grid_path
    return record.paths.table_path('evaluation_stats')


def comparison_by_k_rows(strategy_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], dict[str, Mapping[str, object]]] = {}
    for row in strategy_rows:
        experiment = str(row.get('Experiment') or '')
        k = _int_or_none(row.get('k'))
        strategy = row.get('strategy')
        if not experiment or k is None or strategy not in STRATEGIES:
            continue
        grouped.setdefault((experiment, k), {})[cast(str, strategy)] = row

    rows: list[dict[str, object]] = []
    for (experiment, k), by_strategy in sorted(grouped.items()):
        first = next(iter(by_strategy.values()))
        out = {
            'Experiment': experiment,
            'ShortExperiment': first.get('ShortExperiment'),
            'Distribution': first.get('Distribution'),
            'ShortDistribution': first.get('ShortDistribution'),
            'ExperimentFamily': first.get('ExperimentFamily'),
            'ExperimentFamilyLabel': first.get('ExperimentFamilyLabel'),
            'RunLabel': first.get('RunLabel'),
            'EmbeddingModel': first.get('EmbeddingModel'),
            'EmbeddingDimension': first.get('EmbeddingDimension'),
            'OnlyPassGeometry': first.get('OnlyPassGeometry'),
            'QueryScope': first.get('QueryScope'),
            'k': k,
            'SelectionSource': first.get('SelectionSource'),
        }
        for strategy in STRATEGIES:
            row = by_strategy.get(strategy)
            label = _strategy_label(strategy)
            out[f'{label}_lambda'] = row.get('lam') if row else None
            out[f'{label}_lambda_norm'] = row.get('lambda_norm') if row else None
            out[f'{label}_n_queries'] = row.get('n_queries') if row else None
            for metric, metric_label in METRIC_LABELS.items():
                out[f'{label}_{metric_label}'] = row.get(metric) if row else None

        for metric_label in ('FCP', 'FacetCoverage', 'AllFacetCleanRate', 'Precision'):
            fac_loc = _float_or_none(out.get(f'FacLoc_{metric_label}'))
            mmr = _float_or_none(out.get(f'MMR_{metric_label}'))
            top_k = _float_or_none(out.get(f'TopK_{metric_label}'))
            out[f'Delta_FacLoc_MMR_{metric_label}'] = _subtract(fac_loc, mmr)
            out[f'Delta_FacLoc_TopK_{metric_label}'] = _subtract(fac_loc, top_k)
            out[f'Delta_MMR_TopK_{metric_label}'] = _subtract(mmr, top_k)

        out['FacLocVsMMR_FCPOutcome'] = _delta_outcome(
            _float_or_none(out.get('Delta_FacLoc_MMR_FCP')),
            epsilon=FCP_TIE_EPSILON,
        )
        out['FacLocVsMMR_AllFacetCleanRateOutcome'] = _delta_outcome(
            _float_or_none(out.get('Delta_FacLoc_MMR_AllFacetCleanRate')),
            epsilon=FCP_TIE_EPSILON,
        )
        out['FCPWinner'] = _winner_for_metric(out, 'FCP')
        out['AllFacetCleanRateWinner'] = _winner_for_metric(out, 'AllFacetCleanRate')
        rows.append(out)
    return rows


def headline_rows_from_comparisons(
    comparison_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_experiment: dict[str, list[Mapping[str, object]]] = {}
    for row in comparison_rows:
        experiment = str(row.get('Experiment') or '')
        if experiment:
            by_experiment.setdefault(experiment, []).append(row)

    rows: list[dict[str, object]] = []
    for _experiment, group in sorted(by_experiment.items()):
        complete_rows = [
            row
            for row in group
            if all(
                row.get(f'{_strategy_label(strategy)}_FCP') is not None for strategy in STRATEGIES
            )
        ]
        candidates = complete_rows or group
        selected = min(candidates, key=lambda row: int(cast(int, row.get('k') or 0)))
        out = dict(selected)
        out['HeadlineRule'] = 'smallest k with all strategies' if complete_rows else 'smallest k'
        rows.append(out)
    return rows


def lambda_stability_rows(
    strategy_rows: Sequence[Mapping[str, object]],
    near_optimal_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for strategy in DIVERSIFYING_STRATEGIES:
        selected = [row for row in strategy_rows if row.get('strategy') == strategy]
        lambdas = [_float_or_none(row.get('lam')) for row in selected]
        lambda_norms = [_float_or_none(row.get('lambda_norm')) for row in selected]
        lambdas = [value for value in lambdas if value is not None]
        lambda_norms = [value for value in lambda_norms if value is not None]
        near_rows = [row for row in near_optimal_rows if row.get('strategy') == strategy]
        near_fractions = [
            value
            for value in (_float_or_none(row.get('NearOptimalLambdaFraction')) for row in near_rows)
            if value is not None
        ]
        near_spans = [
            value
            for value in (_float_or_none(row.get('NearOptimalLambdaSpanNorm')) for row in near_rows)
            if value is not None
        ]
        row: dict[str, object] = {
            'strategy': strategy,
            'n_selected': len(lambdas),
            'distinct_lambda_count': len(set(round(value, 6) for value in lambdas)),
            'boundary_selection_rate': _boundary_rate(lambda_norms),
            'near_optimal_rows': len(near_rows),
        }
        row.update(_numeric_stats(lambdas, prefix='selected_lambda'))
        row.update(_numeric_stats(lambda_norms, prefix='selected_lambda_norm'))
        row.update(_numeric_stats(near_fractions, prefix='near_optimal_fraction'))
        row.update(_numeric_stats(near_spans, prefix='near_optimal_span_norm'))
        rows.append(row)
    return rows


def embedding_model_summary_rows(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    geometry_rows: Sequence[Mapping[str, object]],
    headline_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    manifest_by_exp = {str(row.get('Experiment')): row for row in manifest_rows}
    geometry_by_exp = {str(row.get('Experiment')): row for row in geometry_rows}
    headline_by_exp = {str(row.get('Experiment')): row for row in headline_rows}
    run_rows: list[Mapping[str, object]] = []
    for experiment, manifest in manifest_by_exp.items():
        geometry = geometry_by_exp.get(experiment, {})
        headline = headline_by_exp.get(experiment, {})
        run_rows.append({
            'EmbeddingModel': manifest.get('EmbeddingModel'),
            'EmbeddingDimension': manifest.get('EmbeddingDimension'),
            'GeometryPassRate': geometry.get('GeometryPassRate'),
            'GeometryQueries': geometry.get('GeometryQueries'),
            'GeometryPassQueries': geometry.get('GeometryPassQueries'),
            'TopK_FCP': headline.get('TopK_FCP'),
            'MMR_FCP': headline.get('MMR_FCP'),
            'FacLoc_FCP': headline.get('FacLoc_FCP'),
            'Delta_FacLoc_MMR_FCP': headline.get('Delta_FacLoc_MMR_FCP'),
            'Delta_FacLoc_TopK_FCP': headline.get('Delta_FacLoc_TopK_FCP'),
            'OnlyPassGeometry': manifest.get('OnlyPassGeometry'),
        })

    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in run_rows:
        model = str(row.get('EmbeddingModel') or 'unknown')
        grouped.setdefault(model, []).append(row)

    rows: list[dict[str, object]] = []
    for model, group in sorted(grouped.items()):
        out: dict[str, object] = {'EmbeddingModel': model, 'Runs': len(group)}
        out.update(
            _numeric_stats(_numeric_values(group, 'EmbeddingDimension'), 'EmbeddingDimension')
        )
        out.update(_numeric_stats(_numeric_values(group, 'GeometryPassRate'), 'GeometryPassRate'))
        out.update(_numeric_stats(_numeric_values(group, 'GeometryQueries'), 'GeometryQueries'))
        out.update(
            _numeric_stats(_numeric_values(group, 'GeometryPassQueries'), 'GeometryPassQueries')
        )
        out.update(_numeric_stats(_numeric_values(group, 'TopK_FCP'), 'TopK_FCP'))
        out.update(_numeric_stats(_numeric_values(group, 'MMR_FCP'), 'MMR_FCP'))
        out.update(_numeric_stats(_numeric_values(group, 'FacLoc_FCP'), 'FacLoc_FCP'))
        out.update(
            _numeric_stats(_numeric_values(group, 'Delta_FacLoc_MMR_FCP'), 'Delta_FacLoc_MMR_FCP')
        )
        out.update(
            _numeric_stats(
                _numeric_values(group, 'Delta_FacLoc_TopK_FCP'),
                'Delta_FacLoc_TopK_FCP',
            )
        )
        out['PassOnlyRuns'] = sum(row.get('OnlyPassGeometry') is True for row in group)
        out['AllQueryRuns'] = sum(row.get('OnlyPassGeometry') is False for row in group)
        rows.append(out)
    return rows


def embedding_query_scope_pair_rows(
    headline_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in headline_rows:
        distribution = str(row.get('Distribution') or '')
        model = str(row.get('EmbeddingModel') or '')
        if distribution and model:
            grouped.setdefault((distribution, model), []).append(row)

    rows: list[dict[str, object]] = []
    for (distribution, model), group in sorted(grouped.items()):
        pass_rows = [row for row in group if row.get('OnlyPassGeometry') is True]
        all_query_rows = [row for row in group if row.get('OnlyPassGeometry') is False]
        if not pass_rows or not all_query_rows:
            continue
        pass_row = pass_rows[0]
        all_row = all_query_rows[0]
        out: dict[str, object] = {
            'Distribution': distribution,
            'ShortDistribution': short_token(distribution),
            'ExperimentFamily': pass_row.get('ExperimentFamily') or all_row.get('ExperimentFamily'),
            'ExperimentFamilyLabel': pass_row.get('ExperimentFamilyLabel')
            or all_row.get('ExperimentFamilyLabel'),
            'EmbeddingModel': model,
            'PassOnlyExperiment': pass_row.get('Experiment'),
            'PassOnlyShortExperiment': pass_row.get('ShortExperiment'),
            'AllQueriesExperiment': all_row.get('Experiment'),
            'AllQueriesShortExperiment': all_row.get('ShortExperiment'),
            'PassOnly_k': pass_row.get('k'),
            'AllQueries_k': all_row.get('k'),
        }
        for label in (
            'TopK_FCP',
            'MMR_FCP',
            'FacLoc_FCP',
            'Delta_FacLoc_MMR_FCP',
            'Delta_FacLoc_TopK_FCP',
        ):
            pass_value = _float_or_none(pass_row.get(label))
            all_value = _float_or_none(all_row.get(label))
            out[f'PassOnly_{label}'] = pass_value
            out[f'AllQueries_{label}'] = all_value
            out[f'AllMinusPassOnly_{label}'] = _subtract(all_value, pass_value)
        rows.append(out)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text('')
        return
    df = pl.DataFrame([dict(row) for row in rows], infer_schema_length=None)
    df.write_csv(path)


def render_report(
    *,
    args: CliArgs,
    records: Sequence[ExperimentRecord],
    dataset_rows: Sequence[Mapping[str, object]],
    geometry_rows: Sequence[Mapping[str, object]],
    comparison_rows: Sequence[Mapping[str, object]],
    headline_rows: Sequence[Mapping[str, object]],
    lambda_rows: Sequence[Mapping[str, object]],
    embedding_summary_rows: Sequence[Mapping[str, object]],
    embedding_pair_rows: Sequence[Mapping[str, object]],
    figures: Sequence[Path],
) -> str:
    lines: list[str] = [
        '# Medical Dataset Experiment Comparison',
        '',
        f'Generated at `{datetime.now(UTC).isoformat()}`.',
        '',
        'This report compares completed experiment folders discovered from persisted '
        '`evaluation_stats.parquet` artifacts. When an experiment already stores held-out '
        'selected-lambda rows, those rows are used directly. Otherwise, lambdas are selected '
        f'post-hoc by `{LAMBDA_SELECTION_MAXIMIZING_METRIC}` within each `strategy x k` grid.',
        '',
        'The headline row for each experiment is the smallest `k` with all three strategies '
        'available, which keeps the summary demanding while preserving the full per-k output in '
        '`comparison_by_k.csv` and `strategy_by_k.csv`.',
        '',
        '## Run Scope',
        '',
        f'- Results dir: `{args.results_dir}`',
        f'- Output dir: `{args.output_dir}`',
        f'- Experiments discovered: `{len(records)}`',
        f'- Scrapped experiments included: `{args.include_scrapped}`',
        f'- Near-optimal lambda epsilon: `{args.near_optimal_epsilon}`',
        '',
    ]

    lines.extend(
        _section_with_table(
            'Headline FacetCoveragePurity',
            _sorted_rows(headline_rows, 'Delta_FacLoc_MMR_FCP'),
            columns=[
                'ShortExperiment',
                'k',
                'EmbeddingModel',
                'QueryScope',
                'ExperimentFamilyLabel',
                'TopK_FCP',
                'MMR_FCP',
                'FacLoc_FCP',
                'Delta_FacLoc_MMR_FCP',
                'Delta_FacLoc_TopK_FCP',
                'FacLocVsMMR_FCPOutcome',
                'TopK_AllFacetCleanRate',
                'MMR_AllFacetCleanRate',
                'FacLoc_AllFacetCleanRate',
                'MMR_lambda',
                'FacLoc_lambda',
            ],
            tablefmt=args.tablefmt,
            max_rows=args.max_table_rows,
        )
    )
    lines.extend(
        _section_with_table(
            'Where FacLoc Is Worse Or Tied With MMR',
            _sorted_rows(
                [
                    row
                    for row in comparison_rows
                    if row.get('FacLocVsMMR_FCPOutcome') in {'facloc_worse', 'tied'}
                ],
                'Delta_FacLoc_MMR_FCP',
                descending=False,
            ),
            columns=[
                'ShortExperiment',
                'k',
                'EmbeddingModel',
                'ExperimentFamilyLabel',
                'TopK_FCP',
                'MMR_FCP',
                'FacLoc_FCP',
                'Delta_FacLoc_MMR_FCP',
                'MMR_AllFacetCleanRate',
                'FacLoc_AllFacetCleanRate',
                'FacLocVsMMR_FCPOutcome',
            ],
            tablefmt=args.tablefmt,
            max_rows=args.max_table_rows,
        )
    )
    lines.extend(
        _section_with_table(
            'Dataset Distributions',
            dataset_rows,
            columns=[
                'ShortExperiment',
                'ExperimentFamilyLabel',
                'DistributionCategory',
                'PoolSizeMean',
                'GoldPercentage',
                'NearMissDistractorPercentage',
                'BackgroundOutlierPercentage',
                'DominantPrimaryGoldCountMean',
                'OtherPrimaryGoldCountMean',
                'SecondaryGoldCountMean',
                'NicheGoldCountMean',
                'HardDistractorCountMean',
            ],
            tablefmt=args.tablefmt,
            max_rows=args.max_table_rows,
        )
    )
    lines.extend(
        _section_with_table(
            'Geometry Filter And Embeddings',
            geometry_rows,
            columns=[
                'ShortExperiment',
                'EmbeddingModel',
                'EmbeddingDimension',
                'GeometryQueries',
                'GeometryPassQueries',
                'GeometryPassRate',
                'NTopkRetrievedFacetsMean',
                'PrimaryAxisTopkFractionMean',
                'DominantPrimaryTopkFractionMean',
                'TopFailureModes',
            ],
            tablefmt=args.tablefmt,
            max_rows=args.max_table_rows,
        )
    )
    lines.extend(
        _section_with_table(
            'Lambda Stability',
            lambda_rows,
            columns=[
                'strategy',
                'n_selected',
                'distinct_lambda_count',
                'selected_lambda_mean',
                'selected_lambda_std',
                'selected_lambda_norm_mean',
                'selected_lambda_norm_std',
                'boundary_selection_rate',
                'near_optimal_fraction_mean',
                'near_optimal_span_norm_mean',
            ],
            tablefmt=args.tablefmt,
            max_rows=args.max_table_rows,
        )
    )
    lines.extend(
        _section_with_table(
            'Embedding Model Summary',
            embedding_summary_rows,
            columns=[
                'EmbeddingModel',
                'Runs',
                'EmbeddingDimension_mean',
                'GeometryPassRate_mean',
                'GeometryPassQueries_mean',
                'TopK_FCP_mean',
                'MMR_FCP_mean',
                'FacLoc_FCP_mean',
                'Delta_FacLoc_MMR_FCP_mean',
                'PassOnlyRuns',
                'AllQueryRuns',
            ],
            tablefmt=args.tablefmt,
            max_rows=args.max_table_rows,
        )
    )
    lines.extend(
        _section_with_table(
            'Pass-Only Vs All-Query Pairs',
            embedding_pair_rows,
            columns=[
                'ShortDistribution',
                'ExperimentFamilyLabel',
                'EmbeddingModel',
                'PassOnlyShortExperiment',
                'AllQueriesShortExperiment',
                'AllMinusPassOnly_TopK_FCP',
                'AllMinusPassOnly_MMR_FCP',
                'AllMinusPassOnly_FacLoc_FCP',
                'AllMinusPassOnly_Delta_FacLoc_MMR_FCP',
            ],
            tablefmt=args.tablefmt,
            max_rows=args.max_table_rows,
        )
    )

    lines.extend([
        '## Output Files',
        '',
        *_bullets(f'`{file_name}`' for file_name in REPORT_FILES),
    ])
    if figures:
        lines.extend([
            '',
            '## Figures',
            '',
            *_bullets(f'`{path.relative_to(args.output_dir)}`' for path in figures),
        ])
    return '\n'.join(lines) + '\n'


def render_interesting_findings(
    *,
    comparison_rows: Sequence[Mapping[str, object]],
    headline_rows: Sequence[Mapping[str, object]],
    geometry_rows: Sequence[Mapping[str, object]],
    lambda_rows: Sequence[Mapping[str, object]],
    embedding_summary_rows: Sequence[Mapping[str, object]],
    embedding_pair_rows: Sequence[Mapping[str, object]],
    tablefmt: str,
    max_table_rows: int,
) -> str:
    fcp_deltas = _numeric_values(comparison_rows, 'Delta_FacLoc_MMR_FCP')
    topk_deltas = _numeric_values(comparison_rows, 'Delta_FacLoc_TopK_FCP')
    complete_fcp_rows = [
        row
        for row in comparison_rows
        if _float_or_none(row.get('Delta_FacLoc_MMR_FCP')) is not None
    ]
    facloc_better_rows = [
        row for row in complete_fcp_rows if row.get('FacLocVsMMR_FCPOutcome') == 'facloc_better'
    ]
    facloc_worse_rows = [
        row for row in complete_fcp_rows if row.get('FacLocVsMMR_FCPOutcome') == 'facloc_worse'
    ]
    facloc_tied_rows = [
        row for row in complete_fcp_rows if row.get('FacLocVsMMR_FCPOutcome') == 'tied'
    ]
    facloc_beats_topk = sum(
        (_float_or_none(row.get('Delta_FacLoc_TopK_FCP')) or 0.0) > 0.0
        for row in comparison_rows
        if _float_or_none(row.get('Delta_FacLoc_TopK_FCP')) is not None
    )

    lines: list[str] = ['# Interesting Findings', '']
    lines.append(
        f'- FacLoc beats MMR on `{LAMBDA_SELECTION_MAXIMIZING_METRIC}` in '
        f'{len(facloc_better_rows)}/{len(complete_fcp_rows)} experiment-k comparisons; '
        f'it is worse in `{len(facloc_worse_rows)}` and tied within '
        f'`±{FCP_TIE_EPSILON:.3f}` FCP in `{len(facloc_tied_rows)}`.'
    )
    lines.append(
        f'- FacLoc beats top-k on `{LAMBDA_SELECTION_MAXIMIZING_METRIC}` in '
        f'{facloc_beats_topk}/{len(topk_deltas)} experiment-k comparisons.'
    )
    if fcp_deltas:
        lines.append(
            f'- Mean FacLoc - MMR FCP delta: `{statistics.fmean(fcp_deltas):.4f}`; '
            f'median: `{statistics.median(fcp_deltas):.4f}`.'
        )
    if topk_deltas:
        lines.append(
            f'- Mean FacLoc - top-k FCP delta: `{statistics.fmean(topk_deltas):.4f}`; '
            f'median: `{statistics.median(topk_deltas):.4f}`.'
        )

    lambda_std = {
        str(row.get('strategy')): _float_or_none(row.get('selected_lambda_norm_std'))
        for row in lambda_rows
    }
    facloc_lambda_std = lambda_std.get('fac_loc')
    mmr_lambda_std = lambda_std.get('mmr')
    if facloc_lambda_std is not None and mmr_lambda_std is not None:
        less_sensitive = 'FacLoc' if facloc_lambda_std <= mmr_lambda_std else 'MMR'
        lines.append(
            f'- Normalized selected-lambda std is lower for `{less_sensitive}` in the '
            'aggregate lambda-stability table.'
        )

    lines.append('')
    lines.extend(
        _section_with_table(
            'Largest FacLoc Over MMR Gains',
            _sorted_rows(headline_rows, 'Delta_FacLoc_MMR_FCP', descending=True),
            columns=[
                'ShortExperiment',
                'k',
                'EmbeddingModel',
                'ExperimentFamilyLabel',
                'TopK_FCP',
                'MMR_FCP',
                'FacLoc_FCP',
                'Delta_FacLoc_MMR_FCP',
                'Delta_FacLoc_TopK_FCP',
            ],
            tablefmt=tablefmt,
            max_rows=max_table_rows,
        )
    )
    lines.extend(
        _section_with_table(
            'FacLoc Worse Or Tied With MMR',
            _sorted_rows(
                [
                    row
                    for row in comparison_rows
                    if row.get('FacLocVsMMR_FCPOutcome') in {'facloc_worse', 'tied'}
                ],
                'Delta_FacLoc_MMR_FCP',
                descending=False,
            ),
            columns=[
                'ShortExperiment',
                'k',
                'EmbeddingModel',
                'ExperimentFamilyLabel',
                'TopK_FCP',
                'MMR_FCP',
                'FacLoc_FCP',
                'Delta_FacLoc_MMR_FCP',
                'MMR_AllFacetCleanRate',
                'FacLoc_AllFacetCleanRate',
                'FacLocVsMMR_FCPOutcome',
            ],
            tablefmt=tablefmt,
            max_rows=max_table_rows,
        )
    )
    lines.extend(
        _section_with_table(
            'Lowest Geometry Pass Rates',
            _sorted_rows(geometry_rows, 'GeometryPassRate', descending=False),
            columns=[
                'ShortExperiment',
                'EmbeddingModel',
                'EmbeddingDimension',
                'GeometryQueries',
                'GeometryPassQueries',
                'GeometryPassRate',
                'TopFailureModes',
            ],
            tablefmt=tablefmt,
            max_rows=max_table_rows,
        )
    )
    lines.extend(
        _section_with_table(
            'Embedding Summary',
            embedding_summary_rows,
            columns=[
                'EmbeddingModel',
                'Runs',
                'GeometryPassRate_mean',
                'GeometryPassRate_min',
                'GeometryPassRate_max',
                'Delta_FacLoc_MMR_FCP_mean',
                'PassOnlyRuns',
                'AllQueryRuns',
            ],
            tablefmt=tablefmt,
            max_rows=max_table_rows,
        )
    )
    lines.extend(
        _section_with_table(
            'Pass-Only Vs All-Query Deltas',
            embedding_pair_rows,
            columns=[
                'ShortDistribution',
                'ExperimentFamilyLabel',
                'EmbeddingModel',
                'AllMinusPassOnly_MMR_FCP',
                'AllMinusPassOnly_FacLoc_FCP',
                'AllMinusPassOnly_Delta_FacLoc_MMR_FCP',
                'AllMinusPassOnly_Delta_FacLoc_TopK_FCP',
            ],
            tablefmt=tablefmt,
            max_rows=max_table_rows,
        )
    )
    return '\n'.join(lines) + '\n'


def write_figures(
    *,
    output_dir: Path,
    plot_format: PlotFormat,
    max_rows: int,
    headline_rows: Sequence[Mapping[str, object]],
    embedding_pair_rows: Sequence[Mapping[str, object]],
    geometry_rows: Sequence[Mapping[str, object]],
    lambda_rows: Sequence[Mapping[str, object]],
    near_optimal_rows: Sequence[Mapping[str, object]],
    dataset_rows: Sequence[Mapping[str, object]],
    warnings: list[str],
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
    except Exception as exc:
        warnings.append(f'plotting skipped because matplotlib could not be imported ({exc})')
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    for obsolete_stem in ('fcp_delta_by_experiment', 'facloc_vs_topk_delta_by_experiment'):
        (output_dir / f'{obsolete_stem}.{plot_format}').unlink(missing_ok=True)
    paths: list[Path] = []

    paths.extend(
        _plot_headline_fcp_delta_columns(
            plt=plt,
            rows=headline_rows,
            output_dir=output_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        _plot_query_scope_delta_columns(
            plt=plt,
            rows=embedding_pair_rows,
            output_dir=output_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        _plot_clean_rate_delta_columns(
            plt=plt,
            rows=headline_rows,
            output_dir=output_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        _plot_geometry_pass_rate(
            plt=plt,
            rows=geometry_rows,
            output_dir=output_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        _plot_lambda_stability(
            plt=plt,
            rows=lambda_rows,
            output_dir=output_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        _plot_near_optimal_width(
            plt=plt,
            rows=near_optimal_rows,
            output_dir=output_dir,
            plot_format=plot_format,
        )
    )
    paths.extend(
        _plot_dataset_composition(
            plt=plt,
            rows=dataset_rows,
            output_dir=output_dir,
            plot_format=plot_format,
        )
    )
    return paths


def _plot_headline_fcp_delta_columns(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    value_columns = ('Delta_FacLoc_MMR_FCP', 'Delta_FacLoc_TopK_FCP')
    plot_rows = [
        row
        for row in rows
        if all(_float_or_none(row.get(column)) is not None for column in value_columns)
    ]
    if not plot_rows:
        return []
    plot_rows = _query_scope_paired_rows(plot_rows, value_columns[0])
    labels = [str(row.get('ShortExperiment') or row.get('Experiment')) for row in plot_rows]
    colors = [_family_color_for_row(row) for row in plot_rows]
    series = [
        (
            'FacLoc - MMR FCP',
            [_float_or_none(row.get(value_columns[0])) or 0.0 for row in plot_rows],
        ),
        (
            'FacLoc - top-k FCP',
            [_float_or_none(row.get(value_columns[1])) or 0.0 for row in plot_rows],
        ),
    ]
    fig_height = max(5.0, 0.28 * len(labels) + 1.6)
    fig, axes_obj = plt.subplots(  # type: ignore[attr-defined]
        ncols=2,
        sharey=True,
        figsize=(15.0, fig_height),
    )
    axes = cast(Sequence[Any], axes_obj)
    try:
        for ax, (title, values) in zip(axes, series, strict=True):
            _draw_family_delta_bars(ax=ax, labels=labels, values=values, colors=colors)
            ax.set_title(title)
            ax.set_xlabel('Headline FCP delta')
        _add_family_legend(fig=fig, rows=plot_rows)
        fig.suptitle('Headline FCP deltas by experiment', y=0.995)
        fig.tight_layout(rect=(0, 0.03, 1, 0.985))
        path = output_dir / f'headline_fcp_deltas_by_experiment.{plot_format}'
        fig.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def _plot_query_scope_delta_columns(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    value_columns = (
        'AllMinusPassOnly_Delta_FacLoc_MMR_FCP',
        'AllMinusPassOnly_Delta_FacLoc_TopK_FCP',
    )
    plot_rows = [
        row
        for row in rows
        if all(_float_or_none(row.get(column)) is not None for column in value_columns)
    ]
    if not plot_rows:
        return []
    # The y-axis is inverted, so ascending data order renders larger scores lower.
    plot_rows = _sorted_rows(plot_rows, value_columns[0], descending=False)
    labels = [
        f'{row.get("ShortDistribution")}/{_short_model_label(str(row.get("EmbeddingModel") or ""))}'
        for row in plot_rows
    ]
    colors = [_family_color_for_row(row) for row in plot_rows]
    series = [
        (
            '(FacLoc - MMR) all-query minus pass-only',
            [_float_or_none(row.get(value_columns[0])) or 0.0 for row in plot_rows],
        ),
        (
            '(FacLoc - top-k) all-query minus pass-only',
            [_float_or_none(row.get(value_columns[1])) or 0.0 for row in plot_rows],
        ),
    ]
    fig_height = max(5.0, 0.32 * len(labels) + 1.6)
    fig, axes_obj = plt.subplots(  # type: ignore[attr-defined]
        ncols=2,
        sharey=True,
        figsize=(16.0, fig_height),
    )
    axes = cast(Sequence[Any], axes_obj)
    try:
        for ax, (title, values) in zip(axes, series, strict=True):
            _draw_family_delta_bars(ax=ax, labels=labels, values=values, colors=colors)
            ax.set_title(title)
            ax.set_xlabel('Headline FCP delta change')
        _add_family_legend(fig=fig, rows=plot_rows)
        fig.suptitle('All-query sensitivity of headline FacLoc margins', y=0.995)
        fig.tight_layout(rect=(0, 0.03, 1, 0.985))
        path = output_dir / f'query_scope_headline_delta_shift_by_experiment.{plot_format}'
        fig.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def _draw_family_delta_bars(
    *,
    ax: Any,
    labels: Sequence[str],
    values: Sequence[float],
    colors: Sequence[str],
) -> None:
    positions = list(range(len(labels)))
    ax.barh(positions, values, color=colors)
    ax.axvline(0.0, color='#303030', linewidth=0.9)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.25)
    _annotate_horizontal_values(ax=ax, values=values)


def _annotate_horizontal_values(*, ax: Any, values: Sequence[float]) -> None:
    if not values:
        return
    min_value = min([0.0, *values])
    max_value = max([0.0, *values])
    span = max(max_value - min_value, 0.05)
    padding = span * 0.22
    ax.set_xlim(min_value - padding, max_value + padding)
    offset = max(span * 0.015, 0.003)
    for position, value in enumerate(values):
        if value >= 0:
            x_position = value + offset
            alignment = 'left'
        else:
            x_position = value - offset
            alignment = 'right'
        ax.text(
            x_position,
            position,
            f'{value:+.3f}',
            va='center',
            ha=alignment,
            fontsize=7,
            color='#202020',
        )


def _query_scope_paired_rows(
    rows: Sequence[Mapping[str, object]],
    value_column: str,
) -> list[Mapping[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        distribution = str(row.get('Distribution') or '')
        embedding_model = str(row.get('EmbeddingModel') or '')
        if distribution and embedding_model:
            grouped.setdefault((distribution, embedding_model), []).append(row)

    ordered_groups: list[tuple[float, tuple[str, str], list[Mapping[str, object]]]] = []
    for key, group in grouped.items():
        values = [
            value
            for value in (_float_or_none(row.get(value_column)) for row in group)
            if value is not None
        ]
        if not values:
            continue
        ordered_groups.append((statistics.fmean(values), key, group))

    ordered_rows: list[Mapping[str, object]] = []
    # Horizontal plots use an inverted y-axis, so ascending row order places
    # higher-scoring pair groups lower in the rendered figure.
    for _mean_value, _key, group in sorted(ordered_groups, key=lambda item: item[0]):
        ordered_rows.extend(
            sorted(
                group,
                key=lambda row: (
                    1 if row.get('OnlyPassGeometry') is False else 0,
                    str(row.get('ShortExperiment') or row.get('Experiment') or ''),
                ),
            )
        )
    return ordered_rows


def _sorted_pair_rows_by_source_mean(
    rows: Sequence[Mapping[str, object]],
    *,
    pass_column: str,
    all_query_column: str,
) -> list[Mapping[str, object]]:
    return sorted(
        rows,
        key=lambda row: _mean_available(
            _float_or_none(row.get(pass_column)),
            _float_or_none(row.get(all_query_column)),
        ),
    )


def _mean_available(*values: float | None) -> float:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else float('-inf')


def _representative_distribution_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        distribution = str(row.get('Distribution') or '')
        if distribution:
            grouped.setdefault(distribution, []).append(row)

    representatives: list[Mapping[str, object]] = []
    for _distribution, group in grouped.items():
        representatives.append(
            sorted(
                group,
                key=lambda row: (
                    0 if row.get('OnlyPassGeometry') is True else 1,
                    str(row.get('ShortExperiment') or row.get('Experiment') or ''),
                ),
            )[0]
        )
    return _sorted_rows(representatives, 'GoldPercentage', descending=True)


def _family_color_for_row(row: Mapping[str, object]) -> str:
    family_id = row.get('ExperimentFamily')
    if isinstance(family_id, str) and family_id in EXPERIMENT_FAMILIES:
        return EXPERIMENT_FAMILY_COLORS[cast(ExperimentFamilyId, family_id)]
    return EXPERIMENT_FAMILY_COLORS['unknown']


def _add_family_legend(*, fig: Any, rows: Sequence[Mapping[str, object]]) -> None:
    from matplotlib.patches import Patch

    present = {
        cast(ExperimentFamilyId, row.get('ExperimentFamily'))
        for row in rows
        if isinstance(row.get('ExperimentFamily'), str)
        and row.get('ExperimentFamily') in EXPERIMENT_FAMILIES
    }
    if not present:
        return
    ordered: list[ExperimentFamilyId] = [
        family_id for family_id in EXPERIMENT_FAMILIES if family_id in present
    ]
    handles = [
        Patch(
            facecolor=EXPERIMENT_FAMILY_COLORS[family_id],
            label=EXPERIMENT_FAMILY_LABELS[family_id],
        )
        for family_id in ordered
    ]
    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=min(4, len(handles)),
        frameon=False,
        fontsize=8,
    )


def _color_tick_labels_by_family(*, ax: Any, rows: Sequence[Mapping[str, object]]) -> None:
    for tick_label, row in zip(ax.get_yticklabels(), rows, strict=False):
        tick_label.set_color(_family_color_for_row(row))


def _plot_clean_rate_delta_columns(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    value_columns = (
        'Delta_FacLoc_MMR_AllFacetCleanRate',
        'Delta_FacLoc_TopK_AllFacetCleanRate',
    )
    plot_rows = [
        row
        for row in rows
        if all(_float_or_none(row.get(column)) is not None for column in value_columns)
    ]
    if not plot_rows:
        return []
    plot_rows = _query_scope_paired_rows(plot_rows, value_columns[0])
    labels = [str(row.get('ShortExperiment') or row.get('Experiment')) for row in plot_rows]
    colors = [_family_color_for_row(row) for row in plot_rows]
    series = [
        (
            'FacLoc - MMR AllFacetCleanRate',
            [_float_or_none(row.get(value_columns[0])) or 0.0 for row in plot_rows],
        ),
        (
            'FacLoc - top-k AllFacetCleanRate',
            [_float_or_none(row.get(value_columns[1])) or 0.0 for row in plot_rows],
        ),
    ]
    fig_height = max(5.0, 0.28 * len(labels) + 1.6)
    fig, axes_obj = plt.subplots(  # type: ignore[attr-defined]
        ncols=2,
        sharey=True,
        figsize=(15.0, fig_height),
    )
    axes = cast(Sequence[Any], axes_obj)
    try:
        for ax, (title, values) in zip(axes, series, strict=True):
            _draw_family_delta_bars(ax=ax, labels=labels, values=values, colors=colors)
            ax.set_title(title)
            ax.set_xlabel('Headline AllFacetCleanRate delta')
        _add_family_legend(fig=fig, rows=plot_rows)
        fig.suptitle('Headline AllFacetCleanRate deltas by experiment', y=0.995)
        fig.tight_layout(rect=(0, 0.03, 1, 0.985))
        path = output_dir / f'all_facet_clean_rate_by_experiment.{plot_format}'
        fig.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def _plot_geometry_pass_rate(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    plot_rows = _sorted_rows(
        [
            row
            for row in rows
            if _float_or_none(row.get('GeometryPassRate')) is not None
            and 'Q' not in str(row.get('ShortExperiment') or row.get('Experiment') or '')
        ],
        'GeometryPassRate',
        descending=True,
    )
    if not plot_rows:
        return []
    labels = [
        f'{row.get("ShortExperiment") or _short_experiment_label(str(row.get("Experiment")))}/'
        f'{_short_model_label(str(row.get("EmbeddingModel")))}'
        for row in plot_rows
    ]
    values = [_float_or_none(row.get('GeometryPassRate')) or 0.0 for row in plot_rows]
    colors = [_family_color_for_row(row) for row in plot_rows]
    fig_height = max(5.0, 0.28 * len(labels) + 1.6)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))  # type: ignore[attr-defined]
    try:
        ax.barh(range(len(labels)), values, color=colors)
        ax.set_title('Geometry filter pass rate by experiment and embedding')
        ax.set_xlabel('Pass rate')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.25)
        _annotate_horizontal_values(ax=ax, values=values)
        ax.set_xlim(0, 1.08)
        _add_family_legend(fig=fig, rows=plot_rows)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        path = output_dir / f'geometry_pass_rate_by_embedding.{plot_format}'
        fig.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def _plot_lambda_stability(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    plot_rows = [row for row in rows if row.get('strategy') in DIVERSIFYING_STRATEGIES]
    if not plot_rows:
        return []
    labels = [str(row.get('strategy')) for row in plot_rows]
    means = [_float_or_none(row.get('selected_lambda_norm_mean')) or 0.0 for row in plot_rows]
    stds = [_float_or_none(row.get('selected_lambda_norm_std')) or 0.0 for row in plot_rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))  # type: ignore[attr-defined]
    try:
        ax.bar(labels, means, yerr=stds, color=['#C47A3A', '#287C8E'], capsize=5)
        ax.set_title('Selected lambda stability')
        ax.set_ylabel('Normalized lambda mean ± std')
        ax.set_ylim(0, min(1.0, max(means + stds + [0.1]) + 0.15))
        ax.grid(axis='y', alpha=0.25)
        fig.tight_layout()
        path = output_dir / f'lambda_stability_boxplot.{plot_format}'
        fig.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def _plot_near_optimal_width(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    data: list[list[float]] = []
    labels: list[str] = []
    for strategy in DIVERSIFYING_STRATEGIES:
        values = [
            value
            for value in (
                _float_or_none(row.get('NearOptimalLambdaSpanNorm'))
                for row in rows
                if row.get('strategy') == strategy
            )
            if value is not None
        ]
        if values:
            data.append(values)
            labels.append(strategy)
    if not data:
        return []
    fig, ax = plt.subplots(figsize=(6.5, 4.5))  # type: ignore[attr-defined]
    try:
        ax.boxplot(data, tick_labels=labels, patch_artist=True)
        ax.set_title('Near-optimal lambda width')
        ax.set_ylabel('Normalized lambda span within epsilon of best FCP')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.25)
        fig.tight_layout()
        path = output_dir / f'near_optimal_lambda_width.{plot_format}'
        fig.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def _plot_dataset_composition(
    *,
    plt: object,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    plot_rows = [
        row
        for row in rows
        if all(
            _float_or_none(row.get(column)) is not None
            for column in (
                'GoldPercentage',
                'NearMissDistractorPercentage',
                'BackgroundOutlierPercentage',
            )
        )
    ]
    plot_rows = _representative_distribution_rows(plot_rows)
    if not plot_rows:
        return []
    labels = [str(row.get('ShortDistribution') or row.get('Distribution')) for row in plot_rows]
    gold = [_float_or_none(row.get('GoldPercentage')) or 0.0 for row in plot_rows]
    near = [_float_or_none(row.get('NearMissDistractorPercentage')) or 0.0 for row in plot_rows]
    background = [
        _float_or_none(row.get('BackgroundOutlierPercentage')) or 0.0 for row in plot_rows
    ]
    positions = list(range(len(labels)))
    fig_height = max(5.0, 0.28 * len(labels) + 1.6)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))  # type: ignore[attr-defined]
    try:
        ax.barh(positions, gold, label='Gold', color='#287C8E')
        ax.barh(positions, near, left=gold, label='Near-miss distractors', color='#C47A3A')
        bottoms = [g + n for g, n in zip(gold, near, strict=True)]
        ax.barh(
            positions,
            background,
            left=bottoms,
            label='Background outliers',
            color='#6F7890',
        )
        ax.set_title('Candidate-pool composition')
        ax.set_xlabel('Share of qrel pool')
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.25)
        _color_tick_labels_by_family(ax=ax, rows=plot_rows)
        ax.legend()
        _add_family_legend(fig=fig, rows=plot_rows)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        path = output_dir / f'dataset_composition_stacked.{plot_format}'
        fig.savefig(path, dpi=180)
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]


def _base_experiment_row(record: ExperimentRecord) -> dict[str, object]:
    metadata = embedding_metadata(record)
    return {
        'Experiment': record.name,
        'ShortExperiment': short_experiment_id(record.name),
        'Distribution': record.distribution_id,
        'ShortDistribution': short_token(record.distribution_id),
        'ExperimentFamily': record.family_id,
        'ExperimentFamilyLabel': record.family_label,
        'RunLabel': record.run_label,
        'IsSubexperiment': record.is_subexperiment,
        'EmbeddingModel': metadata.get('model_name') or record.embedding_model,
        'EmbeddingDimension': metadata.get('dimension'),
        'OnlyPassGeometry': record.only_pass_geometry,
        'QueryScope': _query_scope_label(record.only_pass_geometry),
    }


def embedding_metadata(record: ExperimentRecord) -> dict[str, object]:
    path = record.paths.embeddings_paths('metadata')
    fallback_path = record.experiment_dir / 'embeddings_metadata.json'
    metadata_path = path if path.is_file() else fallback_path
    if not metadata_path.is_file():
        return {}
    try:
        raw = json.loads(metadata_path.read_text())
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items() if _is_jsonish_scalar(value)}


def _mean_min_max_for_columns(df: pl.DataFrame, *, columns: Sequence[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for column in columns:
        if column not in df.columns:
            continue
        values = [
            value
            for value in (_float_or_none(value) for value in df[column].to_list())
            if value is not None
        ]
        if not values:
            continue
        out[f'{column}Mean'] = statistics.fmean(values)
        out[f'{column}Min'] = min(values)
        out[f'{column}Max'] = max(values)
    return out


def _primary_dominance_ratio(summary: Mapping[str, object]) -> float | None:
    dominant = _float_or_none(summary.get('DominantPrimaryGoldCountMean'))
    other_primary = _float_or_none(summary.get('OtherPrimaryGoldCountMean'))
    secondary = _float_or_none(summary.get('SecondaryGoldCountMean'))
    reference_values = [value for value in (other_primary, secondary) if value and value > 0.0]
    if dominant is None or not reference_values:
        return None
    return dominant / statistics.fmean(reference_values)


def _distribution_category(summary: Mapping[str, object]) -> str | None:
    pool_size = _float_or_none(summary.get('PoolSizeMean'))
    if pool_size is None:
        return None
    size_label = 'small' if pool_size <= 80 else 'medium' if pool_size <= 160 else 'large'
    dominance = _primary_dominance_ratio(summary)
    if dominance is None:
        dominance_label = 'unknown-dominance'
    elif dominance < 1.25:
        dominance_label = 'balanced-primary'
    elif dominance < 1.75:
        dominance_label = 'mild-primary-skew'
    else:
        dominance_label = 'strong-primary-skew'
    background = _float_or_none(summary.get('BackgroundOutlierCountMean')) or 0.0
    background_label = 'no-bg' if background == 0.0 else 'bg-outliers'
    niche = _float_or_none(summary.get('NicheGoldCountMean')) or 0.0
    niche_label = 'niche' if niche > 0.0 else 'no-niche'
    near_miss = _float_or_none(summary.get('NearMissDistractorCountMean')) or 0.0
    distractor_label = 'low-distractors' if near_miss <= 12 else 'hard-distractors'
    return '/'.join([size_label, dominance_label, niche_label, background_label, distractor_label])


def _lambda_norm(
    record: ExperimentRecord,
    strategy: StrategyName,
    lam: float | None,
    stats: pl.DataFrame,
) -> float | None:
    if strategy == 'top_k' or lam is None:
        return None
    start: float | None = None
    stop: float | None = None
    if record.cfg is not None:
        grid = (
            record.cfg.retrieval.lambdas_mmr
            if strategy == 'mmr'
            else record.cfg.retrieval.lambdas_fac_loc
        )
        start = float(grid.start)
        stop = float(grid.stop)
    else:
        sub = stats.filter(pl.col('strategy') == strategy).drop_nulls(subset=['lam'])
        if not sub.is_empty():
            start = _float_or_none(sub['lam'].min())
            stop = _float_or_none(sub['lam'].max())
    if start is None or stop is None:
        return None
    denominator = stop - start
    if denominator == 0:
        return 0.0
    return (lam - start) / denominator


def _winner_for_metric(row: Mapping[str, object], metric_label: str) -> str | None:
    values = {
        'top_k': _float_or_none(row.get(f'TopK_{metric_label}')),
        'mmr': _float_or_none(row.get(f'MMR_{metric_label}')),
        'fac_loc': _float_or_none(row.get(f'FacLoc_{metric_label}')),
    }
    present = {key: value for key, value in values.items() if value is not None}
    if not present:
        return None
    return max(present.items(), key=lambda item: item[1])[0]


def _numeric_stats(values: Sequence[float], prefix: str) -> dict[str, object]:
    if not values:
        return {
            f'{prefix}_mean': None,
            f'{prefix}_std': None,
            f'{prefix}_min': None,
            f'{prefix}_max': None,
            f'{prefix}_median': None,
            f'{prefix}_iqr': None,
        }
    sorted_values = sorted(values)
    return {
        f'{prefix}_mean': statistics.fmean(values),
        f'{prefix}_std': statistics.stdev(values) if len(values) > 1 else 0.0,
        f'{prefix}_min': min(values),
        f'{prefix}_max': max(values),
        f'{prefix}_median': statistics.median(values),
        f'{prefix}_iqr': _quantile(sorted_values, 0.75) - _quantile(sorted_values, 0.25),
    }


def _numeric_values(rows: Sequence[Mapping[str, object]], column: str) -> list[float]:
    return [
        value for value in (_float_or_none(row.get(column)) for row in rows) if value is not None
    ]


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _boundary_rate(values: Sequence[float]) -> float | None:
    if not values:
        return None
    boundary_count = sum(value <= 0.02 or value >= 0.98 for value in values)
    return boundary_count / len(values)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _delta_outcome(delta: float | None, *, epsilon: float) -> str | None:
    if delta is None:
        return None
    if delta < -epsilon:
        return 'facloc_worse'
    if abs(delta) <= epsilon:
        return 'tied'
    return 'facloc_better'


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    if isinstance(value, ScalarItem):
        return _int_or_none(value.item())
    return None


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        numeric = float(value)
        return None if math.isnan(numeric) or math.isinf(numeric) else numeric
    if isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError:
            return None
        return None if math.isnan(numeric) or math.isinf(numeric) else numeric
    if isinstance(value, ScalarItem):
        return _float_or_none(value.item())
    return None


def _json_scalar(value: object) -> object:
    if _is_jsonish_scalar(value):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, ScalarItem):
        item = value.item()
        return item if _is_jsonish_scalar(item) else str(item)
    return str(value)


def _series_mean(series: pl.Series) -> float | None:
    values = [_float_or_none(value) for value in series.to_list()]
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def _is_jsonish_scalar(value: object) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _strategy_label(strategy: StrategyName) -> str:
    return {'top_k': 'TopK', 'mmr': 'MMR', 'fac_loc': 'FacLoc'}[strategy]


def _query_scope_label(only_pass_geometry: bool | None) -> str:
    if only_pass_geometry is True:
        return 'pass-only'
    if only_pass_geometry is False:
        return 'all-query'
    return 'unknown'


def _title_token(value: str) -> str:
    return ''.join(part.capitalize() for part in value.split('_') if part)


def _short_experiment_label(value: str) -> str:
    if len(value) <= 42:
        return value
    if '/' in value:
        parent, child = value.split('/', 1)
        parent = parent[:24]
        child = child[:17]
        return f'{parent}/{child}'
    return value[:39] + '...'


def short_experiment_id(value: str) -> str:
    return '/'.join(short_token(part) for part in Path(value).parts)


def short_token(value: str) -> str:
    return value.split('_', 1)[0]


def _short_model_label(value: str) -> str:
    return value.rsplit('/', 1)[-1].replace('Embedding-', '').replace('-cos-v1', '')


def _sorted_rows(
    rows: Sequence[Mapping[str, object]],
    column: str,
    *,
    descending: bool = True,
) -> list[Mapping[str, object]]:
    with_values = [row for row in rows if _float_or_none(row.get(column)) is not None]
    without_values = [row for row in rows if _float_or_none(row.get(column)) is None]
    return (
        sorted(
            with_values,
            key=lambda row: cast(float, _float_or_none(row.get(column))),
            reverse=descending,
        )
        + without_values
    )


def _select_extreme_rows(
    rows: Sequence[Mapping[str, object]],
    column: str,
    max_rows: int,
) -> list[Mapping[str, object]]:
    if len(rows) <= max_rows:
        return _sorted_rows(rows, column, descending=True)
    half = max(1, max_rows // 2)
    best = _sorted_rows(rows, column, descending=True)[:half]
    worst = _sorted_rows(rows, column, descending=False)[: max_rows - half]
    selected: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for row in [*best, *worst]:
        experiment = str(row.get('Experiment'))
        if experiment in seen:
            continue
        seen.add(experiment)
        selected.append(row)
    return _sorted_rows(selected, column, descending=True)


def _section_with_table(
    title: str,
    rows: Sequence[Mapping[str, object]],
    *,
    columns: Sequence[str],
    tablefmt: str,
    max_rows: int,
) -> list[str]:
    lines = [f'## {title}', '']
    if not rows:
        lines.extend(['No rows available.', ''])
        return lines
    shown = rows[:max_rows]
    table_rows = [
        [_format_table_value(row.get(column), column=column) for column in columns] for row in shown
    ]
    lines.append(
        tabulate(
            table_rows,
            headers=[_table_header(column) for column in columns],
            tablefmt=tablefmt,
            maxcolwidths=[_table_col_width(column) for column in columns],
            disable_numparse=True,
        )
    )
    if len(rows) > max_rows:
        lines.append('')
        lines.append(f'Showing {max_rows}/{len(rows)} rows. Full data is in the CSV outputs.')
    lines.append('')
    return lines


def _format_table_value(value: object, *, column: str) -> object:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value if len(value) <= 70 else value[:67] + '...'
    numeric = _float_or_none(value)
    if numeric is not None:
        if column in INTEGER_TABLE_COLUMNS:
            return f'{numeric:.0f}'
        if abs(numeric) >= 1000:
            return f'{numeric:.0f}'
        return f'{numeric:.4f}'
    if value is None:
        return ''
    text = str(value)
    return text if len(text) <= 70 else text[:67] + '...'


def _table_header(column: str) -> str:
    return TABLE_HEADERS.get(column, column)


def _table_col_width(column: str) -> int:
    return TABLE_COL_WIDTHS.get(column, DEFAULT_TABLE_COL_WIDTH)


def _bullets(items: Iterable[str]) -> list[str]:
    return [f'- {item}' for item in items]


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
        default=12,
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
    )


if __name__ == '__main__':
    outputs = run_report(parse_args())
    print(f'wrote report files to {outputs.output_dir}')
    print(
        f'experiments: {outputs.experiments_loaded}/{outputs.experiments_discovered} configs loaded; '
        f'warnings: {outputs.warnings_count}; figures: {outputs.figures_count}'
    )
