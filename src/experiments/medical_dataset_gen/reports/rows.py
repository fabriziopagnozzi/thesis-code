from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import polars as pl

from experiments.medical_dataset_gen.evaluation.lambda_selection import (
    LAMBDA_SELECTION_MAXIMIZING_METRIC,
    select_best_lambda_rows,
)
from experiments.medical_dataset_gen.reports.analysis_constants import (
    DIVERSIFYING_STRATEGIES,
    EVALUATION_METRICS,
    HELDOUT_SELECTION_COLUMNS,
    ROLE_COUNT_COLUMNS,
    StrategyName,
)
from experiments.medical_dataset_gen.reports.helpers import (
    base_experiment_row,
    distribution_category,
    embedding_metadata,
    float_or_none,
    int_or_none,
    json_scalar,
    lambda_norm,
    mean_min_max_for_columns,
    numeric_values,
    primary_dominance_ratio,
    query_scope_label,
    ratio,
    series_mean,
    short_experiment_id,
    short_token,
    title_token,
)
from experiments.medical_dataset_gen.reports.models import ExperimentRecord
from experiments.medical_dataset_gen.utils.global_schemas import LambdaSelectionCfg


def experiment_manifest_row(record: ExperimentRecord) -> dict[str, object]:
    metadata = embedding_metadata(record)
    return {
        'Experiment': record.name,
        'ShortExperiment': short_experiment_id(record.name),
        'Distribution': record.distribution_id,
        'DistributionBase': record.distribution_base_id,
        'ShortDistribution': short_token(record.distribution_id),
        'ExperimentFamily': record.family_id,
        'ExperimentFamilyLabel': record.family_label,
        'RunLabel': record.run_label,
        'ArtifactOrigin': record.origin,
        'DatasetSchemaVersion': record.dataset_schema_version,
        'EvaluationSchemaVersion': record.evaluation_schema_version,
        'IncludeInCausalSummaries': record.include_in_causal_summaries,
        'IncludeInFamilySummary': record.include_in_family_summary,
        'SuiteTags': '|'.join(record.tags),
        'AnalysisBlocks': '|'.join(record.analysis_blocks),
        'AnalysisTier': record.analysis_tier,
        'RunProfileFactors': json_scalar(record.run_profile_factors),
        'IsSubexperiment': record.is_subexperiment,
        'ConfigLoaded': record.cfg is not None,
        'ConfigError': record.config_error,
        'EmbeddingModel': metadata.get('model_name') or record.embedding_model,
        'EmbeddingDimension': metadata.get('dimension'),
        'EmbeddingChunks': metadata.get('n_chunks'),
        'EmbeddingQueries': metadata.get('n_queries'),
        'OnlyPassGeometry': record.only_pass_geometry,
        'QueryScope': query_scope_label(record.only_pass_geometry),
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
    base = base_experiment_row(record)
    cfg = record.cfg
    base.update(
        {
            'QrelsPath': str(qrels_path),
            'ConfiguredGoldChunksPerQuery': cfg.generation.total_gold_chunks() if cfg else None,
            'ConfiguredDistractorChunksPerQuery': cfg.generation.total_distractor_chunks()
            if cfg
            else None,
            'ConfiguredBackgroundOutlierChunksPerQuery': (
                cfg.generation.chunk_pools.background_outliers_per_query() if cfg else None
            ),
            'ConfiguredNicheClustersPerQuery': (
                cfg.generation.chunk_pools.niche.num_clusters_per_query if cfg else None
            ),
        }
    )
    if not qrels_path.is_file():
        warnings.append(f'{record.name}: qrels missing at {qrels_path}')
        return base

    required_columns = [
        'query_id',
        'cluster_id',
        'cluster_role',
        'facet_id',
        'is_gold',
        'distractor_type',
    ]
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
    summary = mean_min_max_for_columns(
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
    pool_size = float_or_none(summary.get('PoolSizeMean'))
    gold_count = float_or_none(summary.get('GoldCountMean'))
    near_miss_count = float_or_none(summary.get('NearMissDistractorCountMean'))
    background_count = float_or_none(summary.get('BackgroundOutlierCountMean'))

    base.update(summary)
    base.update(
        {
            'QueriesInQrels': per_query.height,
            'QrelRows': qrels.height,
            'GoldPercentage': ratio(gold_count, pool_size),
            'NearMissDistractorPercentage': ratio(near_miss_count, pool_size),
            'BackgroundOutlierPercentage': ratio(background_count, pool_size),
            'PrimaryDominanceRatio': primary_dominance_ratio(summary),
            'DistributionCategory': distribution_category(summary),
            'RealizedPurityMean': ratio(gold_count, pool_size),
            'RealizedGoldMassVectorJson': json.dumps(
                [
                    summary.get('DominantPrimaryGoldCountMean'),
                    summary.get('OtherPrimaryGoldCountMean'),
                    summary.get('SecondaryGoldCountMean'),
                    summary.get('NicheGoldCountMean'),
                ]
            ),
            'RealizedNearMissTypeMassJson': _mean_near_miss_type_masses(qrels),
            'RealizedClusterTopologyJson': _mean_cluster_topology(qrels),
        }
    )
    return base


def _mean_near_miss_type_masses(qrels: pl.DataFrame) -> str:
    non_gold = qrels.filter(
        (~pl.col('is_gold').fill_null(False))
        & (pl.col('cluster_role').fill_null('') != 'background_outlier')
    )
    if non_gold.is_empty():
        return '{}'
    counts = non_gold.group_by('query_id', 'distractor_type').len()
    means = counts.group_by('distractor_type').agg(pl.col('len').mean().alias('mean_mass'))
    payload = {
        str(row['distractor_type']): float(row['mean_mass'])
        for row in means.iter_rows(named=True)
        if row['distractor_type'] is not None
    }
    return json.dumps(payload, sort_keys=True)


def _mean_cluster_topology(qrels: pl.DataFrame) -> str:
    clusters = qrels.group_by('query_id', 'cluster_role', 'cluster_id').len()
    if clusters.is_empty():
        return '{}'
    # First aggregate within query.  Averaging directly over clusters would
    # make query instances with more clusters contribute more weight.
    per_query = clusters.group_by('query_id', 'cluster_role').agg(
        pl.len().alias('clusters_per_query'),
        pl.col('len').mean().alias('chunks_per_cluster'),
    )
    summary = per_query.group_by('cluster_role').agg(
        pl.col('clusters_per_query').mean().alias('clusters_per_query'),
        pl.col('chunks_per_cluster').mean().alias('chunks_per_cluster'),
    )
    payload = {
        str(row['cluster_role']): {
            'clusters_per_query': float(row['clusters_per_query']),
            'chunks_per_cluster': float(row['chunks_per_cluster']),
        }
        for row in summary.iter_rows(named=True)
    }
    return json.dumps(payload, sort_keys=True)


def geometry_filter_row(
    record: ExperimentRecord,
    *,
    warnings: list[str],
) -> dict[str, object]:
    geometry_path = record.paths.table_path('geometry_stats')
    base = base_experiment_row(record)
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
        'primary_axis_stress_fraction',
        'dominant_primary_topk_fraction',
        'fac_topk',
        'fac_facloc',
        'avg_cos_topk',
        'avg_cos_facloc',
        'jaccard_topk_facloc',
        'query_to_gold_mean',
        'query_to_near_miss_mean',
        'query_to_background_outlier_mean',
        'gold_minus_near_miss_similarity_margin',
        'gold_minus_background_outlier_similarity_margin',
    ):
        if column in columns:
            base[f'{title_token(column)}Mean'] = series_mean(geometry[column])

    failure_rates: list[tuple[str, float]] = []
    for column in sorted(col for col in columns if col.startswith('fail_')):
        rate = series_mean(geometry[column].fill_null(False))
        if rate is not None:
            base[f'{title_token(column)}Rate'] = rate
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
        lam = float_or_none(row.get('lam'))
        out = base_experiment_row(record)
        out.update(
            {
                'SelectionSource': source,
                'strategy': strategy,
                'k': int_or_none(row.get('k')),
                'lam': lam,
                'lambda_norm': lambda_norm(record, strategy, lam, stats),
                'CandidatePoolMass': (
                    record.factors.get('pool_mass') if record.factors is not None else None
                ),
            }
        )
        pool_mass = out['CandidatePoolMass']
        out['k_over_pool'] = (
            float(out['k']) / float(pool_mass)
            if isinstance(out['k'], int) and isinstance(pool_mass, int | float) and pool_mass
            else None
        )
        for metric in EVALUATION_METRICS:
            if metric in row:
                out[metric] = json_scalar(row[metric])
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
    max_rows_per_group = int_or_none(counts['n_rows'].max()) or 0
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
    return pl.concat(frames).sort(
        [col for col in ('k', 'strategy', 'lam') if col in stats.columns]
    ), ('posthoc_selected')


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
            out = base_experiment_row(record)
            out.update(
                {
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
                }
            )
            rows.append(out)
    return rows


def lambda_grid_fcp_delta_rows(
    records: Sequence[ExperimentRecord],
    *,
    warnings: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        grid_path = _lambda_validation_grid_stats_path(record)
        if not grid_path.is_file():
            continue
        try:
            stats = pl.read_parquet(grid_path)
        except Exception as exc:
            warnings.append(f'{record.name}: could not read validation lambda grid stats ({exc})')
            continue
        if (
            stats.is_empty()
            or 'strategy' not in stats.columns
            or 'k' not in stats.columns
            or 'lam' not in stats.columns
            or LAMBDA_SELECTION_MAXIMIZING_METRIC not in stats.columns
        ):
            continue

        topk_by_k: dict[int, float] = {}
        topk = stats.filter(pl.col('strategy') == 'top_k')
        for topk_row in topk.iter_rows(named=True):
            k = int_or_none(topk_row.get('k'))
            fcp = float_or_none(topk_row.get(LAMBDA_SELECTION_MAXIMIZING_METRIC))
            if k is not None and fcp is not None:
                topk_by_k[k] = fcp

        for row in stats.filter(pl.col('strategy').is_in(DIVERSIFYING_STRATEGIES)).iter_rows(
            named=True
        ):
            strategy = row.get('strategy')
            if strategy not in DIVERSIFYING_STRATEGIES:
                continue
            typed_strategy = cast(StrategyName, strategy)
            k = int_or_none(row.get('k'))
            lam = float_or_none(row.get('lam'))
            fcp = float_or_none(row.get(LAMBDA_SELECTION_MAXIMIZING_METRIC))
            if k is None or lam is None or fcp is None:
                continue
            topk_fcp = topk_by_k.get(k)
            if topk_fcp is None:
                continue

            out = base_experiment_row(record)
            out.update(
                {
                    'DataSplit': 'validation',
                    'GridStatsPath': str(grid_path),
                    'strategy': typed_strategy,
                    'k': k,
                    'lam': lam,
                    'lambda_norm': lambda_norm(record, typed_strategy, lam, stats),
                    'TopK_FCP': topk_fcp,
                    'Strategy_FCP': fcp,
                    'DeltaStrategyTopK_FCP': fcp - topk_fcp,
                }
            )
            rows.append(out)
    return rows


def lambda_safety_summary_rows(
    lambda_grid_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, StrategyName, int], list[Mapping[str, object]]] = {}
    for row in lambda_grid_rows:
        experiment = str(row.get('Experiment') or '')
        strategy = row.get('strategy')
        k = int_or_none(row.get('k'))
        if not experiment or strategy not in DIVERSIFYING_STRATEGIES or k is None:
            continue
        grouped.setdefault((experiment, cast(StrategyName, strategy), k), []).append(row)

    rows: list[dict[str, object]] = []
    for (_experiment, strategy, k), group in sorted(grouped.items()):
        deltas = numeric_values(group, 'DeltaStrategyTopK_FCP')
        if not deltas:
            continue
        lambda_count = len(deltas)
        nonnegative_count = sum(value >= 0.0 for value in deltas)
        best_row = max(
            group, key=lambda row: float_or_none(row.get('DeltaStrategyTopK_FCP')) or 0.0
        )
        worst_row = min(
            group,
            key=lambda row: float_or_none(row.get('DeltaStrategyTopK_FCP')) or 0.0,
        )
        first = dict(group[0])
        out = {
            key: first.get(key)
            for key in (
                'Experiment',
                'ShortExperiment',
                'Distribution',
                'ShortDistribution',
                'ExperimentFamily',
                'ExperimentFamilyLabel',
                'RunLabel',
                'IsSubexperiment',
                'EmbeddingModel',
                'EmbeddingDimension',
                'OnlyPassGeometry',
                'QueryScope',
                'DataSplit',
            )
        }
        out.update(
            {
                'strategy': strategy,
                'k': k,
                'LambdaCount': lambda_count,
                'NonnegativeDeltaLambdaCount': nonnegative_count,
                'SafeLambdaFraction': nonnegative_count / lambda_count,
                'WorstDeltaStrategyTopK_FCP': min(deltas),
                'BestDeltaStrategyTopK_FCP': max(deltas),
                'MeanDeltaStrategyTopK_FCP': statistics.fmean(deltas),
                'MedianDeltaStrategyTopK_FCP': statistics.median(deltas),
                'DeltaStrategyTopK_FCPRange': max(deltas) - min(deltas),
                'WorstLambda': worst_row.get('lam'),
                'WorstLambdaNorm': worst_row.get('lambda_norm'),
                'BestLambda': best_row.get('lam'),
                'BestLambdaNorm': best_row.get('lambda_norm'),
            }
        )
        rows.append(out)
    return rows


def _lambda_grid_stats_path(record: ExperimentRecord) -> Path:
    report_grid_path = record.paths.table_path('evaluation_report_grid_stats')
    if report_grid_path.is_file():
        return report_grid_path
    return record.paths.table_path('evaluation_stats')


def _lambda_validation_grid_stats_path(record: ExperimentRecord) -> Path:
    selection_path = record.paths.table_path('evaluation_selection_stats')
    if selection_path.is_file():
        return selection_path
    return _lambda_grid_stats_path(record)
