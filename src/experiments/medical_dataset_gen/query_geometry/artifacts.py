"""Build and store the geometry-analysis artifacts.

This module exists to keep the embedding-geometry stage's file handling in one
place, separate from the numerical reduction logic. It uses the shared
experiment paths and plain JSON/parquet serialization so artifact lookup stays
consistent with the rest of the pipeline.
"""

from __future__ import annotations

import json
from typing import cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from experiments.medical_dataset_gen.evaluation.retrieval_utils import (
    get_candidate_pool_indices,
    run_topn_cosine_retrieval,
    select_indices,
)
from experiments.medical_dataset_gen.global_config import ExperimentCfg
from experiments.medical_dataset_gen.query_geometry.dim_reduction import (
    cluster_features,
    hdbscan_labels,
    reduce_for_plot,
)
from experiments.medical_dataset_gen.schemas.query_geometry_schemas import (
    GeometryArtifact,
    GeometryGlobalLambdaKey,
    GeometryIndexMaps,
    GeometryQueryLambdaKey,
    GeometrySelection,
)
from experiments.medical_dataset_gen.schemas.retrieval_schemas import (
    ChunkDocumentRecord,
    QrelRecord,
    QueryRecord,
    RetrievalStrategy,
)

_STRATEGY_ORDER = ['top_k', 'mmr', 'fac_loc']
_STATS_BEST_SORT = [
    'CleanFacetF1@k',
    'MeanFacetHitRate@k',
    'Precision@k',
    'DistractorRate',
    'MeanFacetRecall@k',
    'alpha-nDCG@k',
    'CFF1',
    'FC',
    'GP',
    'DR',
    'WFC',
    'alpha_nDCG',
]
_STATS_BEST_DESC = [True, True, True, False, True, True, True, True, False, True, True, True]
_QUERY_BEST_SORT = [
    'clean_facet_f1',
    'facet_coverage',
    'gold_precision',
    'distractor_rate',
    'weighted_facet_coverage',
    'alpha_ndcg',
    'gold_recall',
]
_QUERY_BEST_DESC = [True, True, True, False, True, True, True]

_QUERY_SELECTION_SORT = [
    'selection_score',
    'passes_filter',
    'fac_loc_fc_gain',
    'topk_dominant_count',
    'query_id',
]
_QUERY_SELECTION_BEST_DESC = [True, True, True, True, False]
_QUERY_SELECTION_WORST_DESC = [False, False, False, False, False]


def choose_query_ids(
    cfg: ExperimentCfg,
    queries: pl.DataFrame,
    geometry_filter: pl.DataFrame,
    eval_results: pl.DataFrame,
) -> list[str]:
    groups = choose_query_groups(cfg, queries, geometry_filter, eval_results)
    return [query_id for query_ids in groups.values() for query_id in query_ids]


def choose_query_groups(
    cfg: ExperimentCfg,
    queries: pl.DataFrame,
    geometry_filter: pl.DataFrame,
    eval_results: pl.DataFrame,
) -> dict[str, list[str]]:
    if cfg.query_geometry.query_ids:
        return {'manual': cfg.query_geometry.query_ids[: cfg.query_geometry.n_queries]}

    ranked = ranked_queries_for_query_geometry(cfg, queries, geometry_filter, eval_results)
    if cfg.query_geometry.query_selection == 'best':
        return {'best': ranked['query_id'].head(cfg.query_geometry.n_queries).to_list()}
    return mixed_query_groups(ranked, cfg.query_geometry.n_queries)


def query_directory_names_for_groups(
    cfg: ExperimentCfg,
    queries: pl.DataFrame,
    geometry_filter: pl.DataFrame,
    eval_results: pl.DataFrame,
    groups: dict[str, list[str]],
) -> dict[str, str]:
    ranked = ranked_queries_for_query_geometry(cfg, queries, geometry_filter, eval_results)
    best_rank_by_qid = _rank_position_map(ranked['query_id'].to_list())
    query_dir_name_by_id: dict[str, str] = {}
    for _, query_ids in groups.items():
        for fallback_rank, qid in enumerate(query_ids, start=1):
            # Directory prefixes should reflect the query's absolute position in
            # the full best-to-worst ranking, even for the "bad" tail slice.
            rank = best_rank_by_qid.get(qid, fallback_rank)
            query_dir_name_by_id[qid] = f'{rank:04d}_{qid}'
    return query_dir_name_by_id


def ranked_queries_for_query_geometry(
    cfg: ExperimentCfg,
    queries: pl.DataFrame,
    geometry_filter: pl.DataFrame,
    eval_results: pl.DataFrame,
) -> pl.DataFrame:
    base = queries.select('query_id')
    if geometry_filter.height > 0:
        gcols = [
            col
            for col in [
                'query_id',
                'passes_filter',
                'topk_dominant_count',
                'in_minus_cross_similarity',
                'n_distractors_in_pool',
            ]
            if col in geometry_filter.columns
        ]
        base = base.join(geometry_filter.select(gcols), on='query_id', how='left')
    else:
        base = base.with_columns(
            pl.lit(True).alias('passes_filter'),
            pl.lit(0).alias('topk_dominant_count'),
            pl.lit(0.0).alias('in_minus_cross_similarity'),
            pl.lit(0).alias('n_distractors_in_pool'),
        )

    if eval_results.height > 0:
        gains = evaluation_gain_table(eval_results, cfg.query_geometry.plot_k)
        if gains.height > 0:
            base = base.join(gains, on='query_id', how='left')

    for col, default in [
        ('passes_filter', True),
        ('topk_dominant_count', 0),
        ('in_minus_cross_similarity', 0.0),
        ('n_distractors_in_pool', 0),
        ('fac_loc_fc_gain', 0.0),
        ('mmr_fc_gain', 0.0),
    ]:
        if col not in base.columns:
            base = base.with_columns(pl.lit(default).alias(col))

    ranked = (
        base
        .with_columns(
            pl.col('passes_filter').fill_null(False),
            pl.col('topk_dominant_count').fill_null(0),
            pl.col('in_minus_cross_similarity').fill_null(0.0),
            pl.col('n_distractors_in_pool').fill_null(0),
            pl.col('fac_loc_fc_gain').fill_null(0.0),
            pl.col('mmr_fc_gain').fill_null(0.0),
        )
        .with_columns(
            (
                pl.col('passes_filter').cast(pl.Int64) * 10.0
                + pl.col('fac_loc_fc_gain') * 5.0
                + pl.col('topk_dominant_count') * 0.3
                + pl.col('in_minus_cross_similarity') * 3.0
                + pl.col('n_distractors_in_pool') * 0.02
            ).alias('selection_score')
        )
        .sort(_QUERY_SELECTION_SORT, descending=_QUERY_SELECTION_BEST_DESC)
    )
    return ranked


def mixed_query_ids(ranked: pl.DataFrame, n_queries: int) -> list[str]:
    groups = mixed_query_groups(ranked, n_queries)
    return [query_id for query_ids in groups.values() for query_id in query_ids]


def mixed_query_groups(ranked: pl.DataFrame, n_queries: int) -> dict[str, list[str]]:
    n_best, n_mid, n_bad = mixed_group_sizes(min(n_queries, ranked.height))
    best_ids = ranked['query_id'].head(n_best).to_list()
    bad_ids = (
        ranked
        .sort(_QUERY_SELECTION_SORT, descending=_QUERY_SELECTION_WORST_DESC)['query_id']
        .head(n_bad)
        .to_list()
    )
    excluded_ids = [*best_ids, *bad_ids]
    remaining = ranked.filter(~pl.col('query_id').is_in(excluded_ids))
    mid_start = max(0, (remaining.height - n_mid) // 2)
    mid_ids = remaining['query_id'].slice(mid_start, n_mid).to_list()

    return {
        group: query_ids
        for group, query_ids in [('best', best_ids), ('mid', mid_ids), ('worst', bad_ids)]
        if query_ids
    }


def mixed_group_sizes(n_queries: int) -> tuple[int, int, int]:
    base = n_queries // 3
    remainder = n_queries % 3
    n_best = base + int(remainder >= 1)
    n_mid = base + int(remainder >= 2)
    n_bad = base
    return n_best, n_mid, n_bad


def _rank_position_map(query_ids: list[str]) -> dict[str, int]:
    return {qid: rank for rank, qid in enumerate(query_ids, start=1)}


def evaluation_gain_table(eval_results: pl.DataFrame, k: int) -> pl.DataFrame:
    if eval_results.height == 0:
        return pl.DataFrame()
    available_k = sorted(eval_results['k'].unique().to_list())
    if k not in available_k:
        k = available_k[len(available_k) // 2]

    topk = eval_results.filter((pl.col('strategy') == 'top_k') & (pl.col('k') == k)).select(
        'query_id', pl.col('facet_coverage').alias('topk_fc')
    )
    rows = [topk]
    for strategy in ['fac_loc', 'mmr']:
        sub = eval_results.filter((pl.col('strategy') == strategy) & (pl.col('k') == k))
        if sub.height == 0:
            continue
        agg_exprs = [
            pl.col('facet_coverage').mean().alias('fc'),
            pl.col('gold_precision').mean().alias('gp'),
            pl.col('distractor_rate').mean().alias('dr'),
            pl.col('weighted_facet_coverage').mean().alias('wfc'),
        ]
        sort_cols = ['query_id', 'fc', 'gp', 'dr', 'wfc']
        descending = [False, True, True, False, True]
        if 'alpha_ndcg' in sub.columns:
            agg_exprs.append(pl.col('alpha_ndcg').mean().alias('alpha_ndcg'))
            sort_cols.append('alpha_ndcg')
            descending.append(True)
        best = (
            sub
            .group_by('query_id', 'lam')
            .agg(agg_exprs)
            .sort(sort_cols, descending=descending)
            .group_by('query_id')
            .first()
            .select('query_id', pl.col('fc').alias(f'{strategy}_fc'))
        )
        rows.append(best)

    joined = rows[0]
    for row in rows[1:]:
        joined = joined.join(row, on='query_id', how='left')
    if 'fac_loc_fc' not in joined.columns:
        joined = joined.with_columns(pl.lit(None).alias('fac_loc_fc'))
    if 'mmr_fc' not in joined.columns:
        joined = joined.with_columns(pl.lit(None).alias('mmr_fc'))
    return joined.with_columns(
        (pl.col('fac_loc_fc').fill_null(pl.col('topk_fc')) - pl.col('topk_fc')).alias(
            'fac_loc_fc_gain'
        ),
        (pl.col('mmr_fc').fill_null(pl.col('topk_fc')) - pl.col('topk_fc')).alias('mmr_fc_gain'),
    ).select('query_id', 'fac_loc_fc_gain', 'mmr_fc_gain')


def build_query_artifact(
    cfg: ExperimentCfg,
    qid: str,
    query: QueryRecord,
    query_qrels: dict[str, QrelRecord],
    chunk_vectors: NDArray[np.float32],
    query_vectors: NDArray[np.float32],
    chunk_ids: list[str],
    maps: GeometryIndexMaps,
    query_best_lambdas: dict[GeometryQueryLambdaKey, float],
    global_best_lambdas: dict[GeometryGlobalLambdaKey, float],
) -> GeometryArtifact | None:
    qidx = maps['query_id_to_idx'][qid]
    pool_n = cfg.query_geometry.candidate_pool_n or cfg.retrieval.candidate_pool_n

    candidate_idx = get_candidate_pool_indices(
        query_id=qid,
        chunks_by_source_query=maps['chunks_by_source_query'],
    )
    topn_global, topn_sims = run_topn_cosine_retrieval(
        candidate_indices=candidate_idx,
        chunk_vectors=chunk_vectors,
        query_vector=query_vectors[qidx],
        n=pool_n,
    )
    if len(topn_global) == 0:
        return None

    candidate_vectors = chunk_vectors[topn_global]
    query_vector = query_vectors[qidx]
    sim_matrix = candidate_vectors @ candidate_vectors.T
    k = min(cfg.query_geometry.plot_k, len(topn_global))
    candidate_chunk_ids = [chunk_ids[int(i)] for i in topn_global]
    labels, label_ids, roles, is_gold = candidate_labels(
        candidate_chunk_ids,
        query_qrels,
        maps['chunk_by_id'],
        query,
    )
    selection_variants = strategy_selection_variants(cfg, topn_sims, sim_matrix, k)
    selections = strategy_selections(
        cfg,
        query_best_lambdas,
        global_best_lambdas,
        qid,
        selection_variants,
        k,
    )
    coords, reduction_method = reduce_for_plot(
        cfg,
        np.vstack([candidate_vectors, query_vector[None, :]]).astype(np.float32),
    )
    cluster_labels = hdbscan_labels(cfg, cluster_features(cfg, candidate_vectors))

    return GeometryArtifact(
        query_id=qid,
        query=query,
        pool_scope=cfg.retrieval.pool_scope,
        candidate_chunk_ids=candidate_chunk_ids,
        candidate_vectors=candidate_vectors,
        query_vector=query_vector,
        sim_to_query=topn_sims.astype(np.float32),
        sim_matrix=sim_matrix,
        coords=coords[:-1],
        query_coord=coords[-1],
        reduction_method=reduction_method,
        labels=labels,
        label_ids=label_ids,
        roles=roles,
        is_gold=is_gold,
        facets_by_id=facet_label_map(query),
        cluster_labels=cluster_labels,
        selections=selections,
        selection_variants=selection_variants,
        lambda_values_by_strategy={
            'mmr': [
                float(lam)
                for lam in cfg.retrieval.lambda_values_for_strategy('mmr')
                if lam is not None
            ],
            'fac_loc': [
                float(lam)
                for lam in cfg.retrieval.lambda_values_for_strategy('fac_loc')
                if lam is not None
            ],
        },
        mmr_window=cfg.retrieval.mmr_window,
        k=k,
        chunk_by_id=maps['chunk_by_id'],
        qrel_by_chunk_id=query_qrels,
    )


def candidate_labels(
    candidate_chunk_ids: list[str],
    query_qrels: dict[str, QrelRecord],
    chunk_by_id: dict[str, ChunkDocumentRecord],
    query: QueryRecord,
) -> tuple[list[str], list[str], list[str], list[bool]]:
    facet_labels = facet_label_map(query)
    labels: list[str] = []
    label_ids: list[str] = []
    roles: list[str] = []
    gold_flags: list[bool] = []
    for chunk_id in candidate_chunk_ids:
        qrel = query_qrels.get(chunk_id)
        if qrel is None:
            raise RuntimeError(
                f'query-local candidate {chunk_id!r} has no qrel for query {query.query_id!r}'
            )
        roles.append(str(qrel.cluster_role or 'unknown'))
        gold = qrel.is_gold
        gold_flags.append(gold)
        if gold and qrel.facet_id:
            facet_id = qrel.facet_id
            label_ids.append(facet_id)
            labels.append(facet_labels.get(facet_id, facet_id))
        else:
            dtype = qrel.distractor_type or 'hard_distractor'
            label_ids.append(dtype)
            labels.append(distractor_label(qrel, chunk_by_id.get(chunk_id)))
    return labels, label_ids, roles, gold_flags


def distractor_label(qrel: QrelRecord, chunk: ChunkDocumentRecord | None) -> str:
    target = distractor_target_label(chunk)
    if qrel.cluster_role == 'background_outlier':
        cluster_suffix = background_cluster_suffix(qrel.cluster_id)
        return f'background outlier: {target}{cluster_suffix}'
    return target


def distractor_target_label(chunk: ChunkDocumentRecord | None) -> str:
    if chunk is None:
        return facet_surface_label(
            condition='unknown condition',
            subgroup='unknown subgroup',
            axis='unknown axis',
        )
    condition = chunk.condition_display or chunk.condition_id or 'unknown condition'
    subgroup = chunk.subgroup_label or chunk.subgroup_id or 'unknown subgroup'
    axis = str(chunk.axis or 'unknown axis')
    return facet_surface_label(condition=condition, subgroup=subgroup, axis=axis)


def background_cluster_suffix(cluster_id: str | None) -> str:
    if cluster_id is None:
        return ''
    suffix = cluster_id.rsplit('_', 1)[-1]
    if suffix.startswith('bg') and suffix[2:].isdigit():
        return f' (cluster {int(suffix[2:])})'
    return f' ({suffix.replace("_", " ")})'


def facet_label_map(query: QueryRecord) -> dict[str, str]:
    if query.facets_json is None:
        return {}
    facets = json.loads(query.facets_json)
    result = {}
    condition = query.condition_display or query.condition_id or 'unknown condition'
    for facet in facets:
        subgroup = str(facet['subgroup_label'])
        axis = str(facet['axis'])
        result[facet['facet_id']] = facet_surface_label(
            condition=condition,
            subgroup=subgroup,
            axis=axis,
        )
    return result


def facet_surface_label(*, condition: str, subgroup: str, axis: str) -> str:
    return f'{condition} / {subgroup} / {axis.replace("_", " ")}'


def strategy_selections(
    cfg: ExperimentCfg,
    query_best_lambdas: dict[GeometryQueryLambdaKey, float],
    global_best_lambdas: dict[GeometryGlobalLambdaKey, float],
    qid: str,
    selection_variants: dict[RetrievalStrategy, list[GeometrySelection]],
    k: int,
) -> dict[RetrievalStrategy, GeometrySelection]:
    selections: dict[RetrievalStrategy, GeometrySelection] = {}
    if 'top_k' in cfg.retrieval.strategies and 'top_k' in selection_variants:
        topk = selection_variants['top_k'][0]
        selections['top_k'] = GeometrySelection(local_indices=topk.local_indices, lam=topk.lam)

    for strategy in cfg.retrieval.strategies.difference({'top_k'}):
        variants = selection_variants.get(strategy, [])
        if not variants:
            continue
        lam = best_lambda(
            strategy,
            k,
            cfg,
            query_best_lambdas=query_best_lambdas,
            global_best_lambdas=global_best_lambdas,
            query_id=qid,
        )
        local = _selection_for_lambda(variants, lam)
        if local is None:
            local = variants[0].local_indices
        selections[strategy] = GeometrySelection(local_indices=local, lam=lam)
    return selections


def strategy_selection_variants(
    cfg: ExperimentCfg, sim_to_query: NDArray[np.float32], sim_matrix: NDArray[np.float32], k: int
) -> dict[RetrievalStrategy, list[GeometrySelection]]:
    variants: dict[RetrievalStrategy, list[GeometrySelection]] = {
        'top_k': [
            GeometrySelection(
                local_indices=select_indices(
                    strategy='top_k',
                    sim_to_query=sim_to_query,
                    sim_matrix=sim_matrix,
                    k=k,
                    lam=None,
                    mmr_window=cfg.retrieval.mmr_window,
                ),
                lam=None,
            )
        ]
    }

    for strategy in ['mmr', 'fac_loc']:
        if strategy not in cfg.retrieval.strategies:
            continue

        variants[strategy] = [
            GeometrySelection(
                local_indices=select_indices(
                    strategy=strategy,
                    sim_to_query=sim_to_query,
                    sim_matrix=sim_matrix,
                    k=k,
                    lam=float(lam),
                    mmr_window=cfg.retrieval.mmr_window,
                ),
                lam=float(lam),
            )
            for lam in cfg.retrieval.lambda_values_for_strategy(strategy)
            if lam is not None
        ]

    return variants


def _selection_for_lambda(
    variants: list[GeometrySelection],
    lam: float,
) -> NDArray[np.intp] | None:
    for variant in variants:
        variant_lam = variant.lam
        if variant_lam is None:
            continue
        if abs(float(variant_lam) - lam) < 1e-12:
            return variant.local_indices
    return None


def best_lambda(
    strategy: RetrievalStrategy,
    k: int,
    cfg: ExperimentCfg | None,
    *,
    query_best_lambdas: dict[GeometryQueryLambdaKey, float] | None = None,
    global_best_lambdas: dict[GeometryGlobalLambdaKey, float] | None = None,
    query_id: str | None = None,
) -> float:
    if query_best_lambdas is not None and query_id is not None:
        query_lam = query_best_lambdas.get((query_id, strategy, k))
        if query_lam is not None:
            return query_lam

    if global_best_lambdas is not None:
        global_lam = global_best_lambdas.get((strategy, k))
        if global_lam is not None:
            return global_lam

    lambda_values = []
    if cfg is not None and strategy in {'fac_loc', 'mmr'}:
        lambda_values = [
            float(lam)
            for lam in cfg.retrieval.lambda_values_for_strategy(strategy)
            if lam is not None
        ]
    if not lambda_values:
        return 0.5
    if strategy == 'fac_loc':
        return min(lambda_values)
    if strategy == 'mmr':
        return max(lambda_values)
    return 0.5


def build_best_lambda_maps(
    eval_stats: pl.DataFrame,
    eval_results: pl.DataFrame,
) -> tuple[
    dict[GeometryQueryLambdaKey, float],
    dict[GeometryGlobalLambdaKey, float],
]:
    return _query_best_lambda_map(eval_results), _global_best_lambda_map(eval_stats)


def _query_best_lambda_map(eval_results: pl.DataFrame) -> dict[GeometryQueryLambdaKey, float]:
    result: dict[GeometryQueryLambdaKey, float] = {}
    required_columns = {'query_id', 'strategy', 'k', 'lam'}
    if eval_results.height == 0 or not required_columns.issubset(eval_results.columns):
        return result

    best_rows = _first_sorted_lambdas(
        eval_results.filter(pl.col('strategy') != 'top_k'),
        group_cols=['query_id', 'strategy', 'k'],
        preferred_sort_cols=_QUERY_BEST_SORT,
        preferred_sort_desc=_QUERY_BEST_DESC,
    )
    for row in best_rows.iter_rows(named=True):
        strategy = cast(RetrievalStrategy, str(row['strategy']))
        result[(str(row['query_id']), strategy, int(row['k']))] = float(row['lam'])
    return result


def _global_best_lambda_map(eval_stats: pl.DataFrame) -> dict[GeometryGlobalLambdaKey, float]:
    result: dict[GeometryGlobalLambdaKey, float] = {}
    required_columns = {'strategy', 'k', 'lam'}
    if eval_stats.height == 0 or not required_columns.issubset(eval_stats.columns):
        return result

    best_rows = _first_sorted_lambdas(
        eval_stats.filter(pl.col('strategy') != 'top_k'),
        group_cols=['strategy', 'k'],
        preferred_sort_cols=_STATS_BEST_SORT,
        preferred_sort_desc=_STATS_BEST_DESC,
    )
    for row in best_rows.iter_rows(named=True):
        strategy = cast(RetrievalStrategy, str(row['strategy']))
        result[(strategy, int(row['k']))] = float(row['lam'])
    return result


def _first_sorted_lambdas(
    df: pl.DataFrame,
    *,
    group_cols: list[str],
    preferred_sort_cols: list[str],
    preferred_sort_desc: list[bool],
) -> pl.DataFrame:
    if df.height == 0:
        return df
    sort_cols, desc = _available_sort(df, preferred_sort_cols, preferred_sort_desc)
    return (
        df
        .sort(
            [*group_cols, *sort_cols],
            descending=[False] * len(group_cols) + desc,
        )
        .group_by(*group_cols, maintain_order=True)
        .first()
    )


def _available_sort(
    df: pl.DataFrame, preferred_cols: list[str], preferred_desc: list[bool]
) -> tuple[list[str], list[bool]]:
    pairs = [
        (col, desc)
        for col, desc in zip(preferred_cols, preferred_desc, strict=True)
        if col in df.columns
    ]
    if not pairs:
        return ['lam'], [False]
    cols, desc = zip(*pairs, strict=True)
    return list(cols), list(desc)
