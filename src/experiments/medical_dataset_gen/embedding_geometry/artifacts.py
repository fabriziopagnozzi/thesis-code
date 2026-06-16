"""Build and store the geometry-analysis artifacts.

This module exists to keep the embedding-geometry stage's file handling in one
place, separate from the numerical reduction logic. It uses the shared
experiment paths and plain JSON/parquet serialization so artifact lookup stays
consistent with the rest of the pipeline.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from experiments.medical_dataset_gen.embedding_geometry.reduction import (
    cluster_features,
    hdbscan_labels,
    reduce_for_plot,
)
from experiments.medical_dataset_gen.global_configs import ExperimentCfg
from experiments.medical_dataset_gen.retrieval.utils import (
    candidate_pool_indices,
    select_indices,
    topn_by_query,
)

_STRATEGY_ORDER = ['top_k', 'mmr', 'fac_loc']
_STATS_BEST_SORT = [
    'FacetCoverage@k',
    'Precision@k',
    'DistractorRate',
    'MeanFacetRecall@k',
    'alpha-nDCG@k',
    'FC',
    'GP',
    'DR',
    'WFC',
    'alpha_nDCG',
]
_STATS_BEST_DESC = [True, True, False, True, True, True, True, False, True, True]
_QUERY_BEST_SORT = [
    'facet_coverage',
    'gold_precision',
    'distractor_rate',
    'weighted_facet_coverage',
    'alpha_ndcg',
    'gold_recall',
]
_QUERY_BEST_DESC = [True, True, False, True, True, True]

_DISTRACTOR_LABELS = {
    'same_condition_wrong_subgroup': 'soft distractor: same condition, wrong subgroup',
    'same_subgroup_wrong_condition': 'hard distractor: wrong condition, same subgroup',
    'same_axis_wrong_condition': 'hard distractor: wrong condition, same answer axis',
    'hard_distractor': 'hard distractor',
}
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
    geometry: pl.DataFrame,
    eval_results: pl.DataFrame,
) -> list[str]:
    groups = choose_query_groups(cfg, queries, geometry, eval_results)
    return [query_id for query_ids in groups.values() for query_id in query_ids]


def choose_query_groups(
    cfg: ExperimentCfg,
    queries: pl.DataFrame,
    geometry: pl.DataFrame,
    eval_results: pl.DataFrame,
) -> dict[str, list[str]]:
    if cfg.embedding_geometry.query_ids:
        return {'manual': cfg.embedding_geometry.query_ids[: cfg.embedding_geometry.n_queries]}

    ranked = ranked_queries_for_embedding_geometry(cfg, queries, geometry, eval_results)
    if cfg.embedding_geometry.query_selection == 'best':
        return {'good': ranked['query_id'].head(cfg.embedding_geometry.n_queries).to_list()}
    return mixed_query_groups(ranked, cfg.embedding_geometry.n_queries)


def ranked_queries_for_embedding_geometry(
    cfg: ExperimentCfg,
    queries: pl.DataFrame,
    geometry: pl.DataFrame,
    eval_results: pl.DataFrame,
) -> pl.DataFrame:
    base = queries.select('query_id')
    if geometry.height > 0:
        gcols = [
            col
            for col in [
                'query_id',
                'passes_filter',
                'topk_dominant_count',
                'in_minus_cross_similarity',
                'n_distractors_in_pool',
            ]
            if col in geometry.columns
        ]
        base = base.join(geometry.select(gcols), on='query_id', how='left')
    else:
        base = base.with_columns(
            pl.lit(True).alias('passes_filter'),
            pl.lit(0).alias('topk_dominant_count'),
            pl.lit(0.0).alias('in_minus_cross_similarity'),
            pl.lit(0).alias('n_distractors_in_pool'),
        )

    if eval_results.height > 0:
        gains = evaluation_gain_table(eval_results, cfg.embedding_geometry.plot_k)
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
        base.with_columns(
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
    n_good, n_mid, n_bad = mixed_group_sizes(min(n_queries, ranked.height))
    good_ids = ranked['query_id'].head(n_good).to_list()
    bad_ids = (
        ranked.sort(_QUERY_SELECTION_SORT, descending=_QUERY_SELECTION_WORST_DESC)['query_id']
        .head(n_bad)
        .to_list()
    )
    excluded_ids = [*good_ids, *bad_ids]
    remaining = ranked.filter(~pl.col('query_id').is_in(excluded_ids))
    mid_start = max(0, (remaining.height - n_mid) // 2)
    mid_ids = remaining['query_id'].slice(mid_start, n_mid).to_list()

    return {
        group: query_ids
        for group, query_ids in [('good', good_ids), ('mid', mid_ids), ('bad', bad_ids)]
        if query_ids
    }


def mixed_group_sizes(n_queries: int) -> tuple[int, int, int]:
    base = n_queries // 3
    remainder = n_queries % 3
    n_good = base + int(remainder >= 1)
    n_mid = base + int(remainder >= 2)
    n_bad = base
    return n_good, n_mid, n_bad


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
            sub.group_by('query_id', 'lam')
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
    queries: pl.DataFrame,
    qrels: pl.DataFrame,
    chunk_vectors: NDArray[np.float32],
    query_vectors: NDArray[np.float32],
    chunk_ids: list[str],
    maps: dict[str, Any],
    eval_stats: pl.DataFrame,
    eval_results: pl.DataFrame,
) -> dict[str, Any] | None:
    query = queries.filter(pl.col('query_id') == qid).row(0, named=True)
    qidx = maps['query_id_to_idx'][qid]
    candidate_idx = candidate_pool_indices(
        query_id=qid,
        pool_scope=cfg.retrieval.pool_scope,
        n_chunks=len(chunk_ids),
        chunks_by_source_query=maps['chunks_by_source_query'],
        chunks_by_condition=maps['chunks_by_condition'],
        query_condition_id=query.get('condition_id'),
    )
    pool_n = cfg.embedding_geometry.candidate_pool_n or cfg.retrieval.candidate_pool_n
    topn_global, topn_sims = topn_by_query(
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
    k = min(cfg.embedding_geometry.plot_k, len(topn_global))
    candidate_chunk_ids = [chunk_ids[int(i)] for i in topn_global]
    labels, label_ids, roles, is_gold = candidate_labels(
        qid, candidate_chunk_ids, maps['chunk_by_id'], query
    )
    selection_variants = strategy_selection_variants(cfg, topn_sims, sim_matrix, k)
    selections = strategy_selections(cfg, eval_stats, eval_results, qid, selection_variants, k)
    coords, reduction_method = reduce_for_plot(
        cfg,
        np.vstack([candidate_vectors, query_vector[None, :]]).astype(np.float32),
    )
    cluster_labels = hdbscan_labels(cfg, cluster_features(cfg, candidate_vectors))

    return {
        'query_id': qid,
        'query': query,
        'pool_scope': cfg.retrieval.pool_scope,
        'candidate_chunk_ids': candidate_chunk_ids,
        'candidate_vectors': candidate_vectors,
        'query_vector': query_vector,
        'sim_to_query': topn_sims.astype(np.float32),
        'sim_matrix': sim_matrix,
        'coords': coords[:-1],
        'query_coord': coords[-1],
        'reduction_method': reduction_method,
        'labels': labels,
        'label_ids': label_ids,
        'roles': roles,
        'is_gold': is_gold,
        'facets_by_id': facet_label_map(query),
        'cluster_labels': cluster_labels,
        'selections': selections,
        'selection_variants': selection_variants,
        'lambda_values': [float(lam) for lam in cfg.retrieval.lambda_values],
        'mmr_window': cfg.retrieval.mmr_window,
        'k': k,
        'qrels': qrels.filter(pl.col('query_id') == qid),
        'chunk_by_id': maps['chunk_by_id'],
    }


def candidate_labels(
    qid: str,
    candidate_chunk_ids: list[str],
    chunk_by_id: dict[str, dict[str, Any]],
    query: dict[str, Any],
) -> tuple[list[str], list[str], list[str], list[bool]]:
    facet_labels = facet_label_map(query)
    labels: list[str] = []
    label_ids: list[str] = []
    roles: list[str] = []
    gold_flags: list[bool] = []
    for chunk_id in candidate_chunk_ids:
        row = chunk_by_id[chunk_id]
        roles.append(str(row.get('cluster_role') or 'unknown'))
        gold = bool(row.get('is_gold'))
        gold_flags.append(gold)
        if row.get('source_query_id') != qid:
            row_condition_id = row.get('condition_id')
            query_condition_id = query.get('condition_id')
            if row_condition_id != query_condition_id:
                label_ids.append('other_condition')
                labels.append('off-query wrong-condition chunks')
                continue
            label_ids.append('other_same_condition_query')
            labels.append('other same-condition queries')
        elif gold and row.get('facet_id'):
            facet_id = str(row['facet_id'])
            label_ids.append(facet_id)
            labels.append(facet_labels.get(facet_id, facet_id))
        else:
            dtype = str(row.get('distractor_type') or 'hard_distractor')
            label_ids.append(dtype)
            labels.append(distractor_label(dtype))
    return labels, label_ids, roles, gold_flags


def distractor_label(distractor_type: str) -> str:
    return _DISTRACTOR_LABELS.get(distractor_type, distractor_type.replace('_', ' '))


def facet_label_map(query: dict[str, Any]) -> dict[str, str]:
    facets = json.loads(query['facets_json'])
    result = {}
    for facet in facets:
        subgroup = str(facet['subgroup_label']).replace('patients ', '')
        axis = 'duration' if facet['axis'] == 'treatment_duration' else 'rehab'
        result[facet['facet_id']] = f'{subgroup} / {axis}'
    return result


def strategy_selections(
    cfg: ExperimentCfg,
    eval_stats: pl.DataFrame,
    eval_results: pl.DataFrame,
    qid: str,
    selection_variants: dict[str, list[dict[str, Any]]],
    k: int,
) -> dict[str, dict[str, Any]]:
    selections = {}
    if 'top_k' in cfg.retrieval.strategies and 'top_k' in selection_variants:
        selections['top_k'] = selection_variants['top_k'][0]
    for strategy in ['mmr', 'fac_loc']:
        if strategy not in cfg.retrieval.strategies:
            continue
        variants = selection_variants.get(strategy, [])
        if not variants:
            continue
        lam = best_lambda(
            eval_stats,
            strategy,
            k,
            cfg,
            eval_results=eval_results,
            query_id=qid,
        )
        local = _selection_for_lambda(variants, lam)
        if local is None:
            local = variants[0]['local_indices']
        selections[strategy] = {'local_indices': local, 'lam': lam}
    return selections


def strategy_selection_variants(
    cfg: ExperimentCfg,
    sim_to_query: NDArray[np.float32],
    sim_matrix: NDArray[np.float32],
    k: int,
) -> dict[str, list[dict[str, Any]]]:
    variants: dict[str, list[dict[str, Any]]] = {
        'top_k': [
            {
                'local_indices': select_indices(
                    strategy='top_k',
                    sim_to_query=sim_to_query,
                    sim_matrix=sim_matrix,
                    k=k,
                    lam=None,
                    mmr_window=cfg.retrieval.mmr_window,
                ),
                'lam': None,
            }
        ]
    }
    for strategy in ['mmr', 'fac_loc']:
        if strategy not in cfg.retrieval.strategies:
            continue
        variants[strategy] = [
            {
                'local_indices': select_indices(
                    strategy=strategy,
                    sim_to_query=sim_to_query,
                    sim_matrix=sim_matrix,
                    k=k,
                    lam=float(lam),
                    mmr_window=cfg.retrieval.mmr_window,
                ),
                'lam': float(lam),
            }
            for lam in cfg.retrieval.lambda_values
        ]
    return variants


def _selection_for_lambda(
    variants: list[dict[str, Any]],
    lam: float,
) -> NDArray[np.intp] | None:
    for variant in variants:
        variant_lam = variant.get('lam')
        if variant_lam is None:
            continue
        if abs(float(variant_lam) - lam) < 1e-12:
            return variant['local_indices']
    return None


def best_lambda(
    eval_stats: pl.DataFrame,
    strategy: str,
    k: int,
    cfg: ExperimentCfg | None,
    *,
    eval_results: pl.DataFrame | None = None,
    query_id: str | None = None,
) -> float:
    if eval_results is not None and query_id is not None:
        query_lam = _best_query_lambda(eval_results, query_id, strategy, k)
        if query_lam is not None:
            return query_lam

    if (
        eval_stats.height > 0
        and {'strategy', 'k', 'lam'}.issubset(eval_stats.columns)
        and strategy in eval_stats['strategy'].unique().to_list()
    ):
        sub = eval_stats.filter((pl.col('strategy') == strategy) & (pl.col('k') == k))
        if sub.height > 0:
            sort_cols, desc = _available_sort(sub, _STATS_BEST_SORT, _STATS_BEST_DESC)
            return float(sub.sort(sort_cols, descending=desc)['lam'][0])
    lambda_values = [] if cfg is None else cfg.retrieval.lambda_values
    if not lambda_values:
        return 0.5
    if strategy == 'fac_loc':
        return min(lambda_values)
    if strategy == 'mmr':
        return max(lambda_values)
    return 0.5


def _best_query_lambda(
    eval_results: pl.DataFrame,
    query_id: str,
    strategy: str,
    k: int,
) -> float | None:
    if not {'query_id', 'strategy', 'k', 'lam'}.issubset(eval_results.columns):
        return None
    sub = eval_results.filter(
        (pl.col('query_id') == query_id) & (pl.col('strategy') == strategy) & (pl.col('k') == k)
    )
    if sub.height == 0:
        return None
    sort_cols, desc = _available_sort(sub, _QUERY_BEST_SORT, _QUERY_BEST_DESC)
    return float(sub.sort(sort_cols, descending=desc)['lam'][0])


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
