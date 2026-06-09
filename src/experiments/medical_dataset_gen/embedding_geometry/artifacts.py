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
_BEST_SORT = ['FC', 'WFC', 'GP', 'DR']
_BEST_DESC = [True, True, True, False]


def choose_query_ids(
    cfg: ExperimentCfg,
    queries: pl.DataFrame,
    geometry: pl.DataFrame,
    eval_results: pl.DataFrame,
) -> list[str]:
    if cfg.embedding_geometry.query_ids:
        return cfg.embedding_geometry.query_ids[: cfg.embedding_geometry.n_queries]

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
        .sort(
            ['selection_score', 'passes_filter', 'fac_loc_fc_gain', 'topk_dominant_count', 'query_id'],
            descending=[True, True, True, True, False],
        )
    )
    return ranked['query_id'].head(cfg.embedding_geometry.n_queries).to_list()


def evaluation_gain_table(eval_results: pl.DataFrame, k: int) -> pl.DataFrame:
    if eval_results.height == 0:
        return pl.DataFrame()
    available_k = sorted(eval_results['k'].unique().to_list())
    if k not in available_k:
        k = available_k[len(available_k) // 2]

    topk = (
        eval_results
        .filter((pl.col('strategy') == 'top_k') & (pl.col('k') == k))
        .select('query_id', pl.col('facet_coverage').alias('topk_fc'))
    )
    rows = [topk]
    for strategy in ['fac_loc', 'mmr']:
        sub = eval_results.filter((pl.col('strategy') == strategy) & (pl.col('k') == k))
        if sub.height == 0:
            continue
        best = (
            sub
            .group_by('query_id', 'lam')
            .agg(
                pl.col('facet_coverage').mean().alias('fc'),
                pl.col('weighted_facet_coverage').mean().alias('wfc'),
                pl.col('gold_precision').mean().alias('gp'),
                pl.col('distractor_rate').mean().alias('dr'),
            )
            .sort(['query_id', 'fc', 'wfc', 'gp', 'dr'], descending=[False, True, True, True, False])
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
        (pl.col('fac_loc_fc').fill_null(pl.col('topk_fc')) - pl.col('topk_fc')).alias('fac_loc_fc_gain'),
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
    labels, label_ids, roles, is_gold = candidate_labels(qid, candidate_chunk_ids, maps['chunk_by_id'], query)
    selections = strategy_selections(cfg, eval_stats, topn_sims, sim_matrix, k)
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
        row_condition_id = row.get('condition_id')
        query_condition_id = query.get('condition_id')
        if row_condition_id != query_condition_id:
            label_ids.append('other_condition')
            labels.append('other conditions')
        elif row.get('source_query_id') != qid:
            label_ids.append('other_same_condition_query')
            labels.append('other same-condition queries')
        elif gold and row.get('facet_id'):
            facet_id = str(row['facet_id'])
            label_ids.append(facet_id)
            labels.append(facet_labels.get(facet_id, facet_id))
        else:
            dtype = str(row.get('distractor_type') or 'hard_distractor')
            label_ids.append(dtype)
            labels.append(dtype.replace('_', ' '))
    return labels, label_ids, roles, gold_flags


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
    sim_to_query: NDArray[np.float32],
    sim_matrix: NDArray[np.float32],
    k: int,
) -> dict[str, dict[str, Any]]:
    selections = {}
    for strategy in _STRATEGY_ORDER:
        if strategy not in cfg.retrieval.strategies:
            continue
        lam = None if strategy == 'top_k' else best_lambda(eval_stats, strategy, k, cfg)
        local = select_indices(
            strategy=strategy,
            sim_to_query=sim_to_query,
            sim_matrix=sim_matrix,
            k=k,
            lam=lam,
            mmr_window=cfg.retrieval.mmr_window,
        )
        selections[strategy] = {'local_indices': local, 'lam': lam}
    return selections


def best_lambda(
    eval_stats: pl.DataFrame,
    strategy: str,
    k: int,
    cfg: ExperimentCfg,
) -> float:
    if eval_stats.height > 0 and strategy in eval_stats['strategy'].unique().to_list():
        sub = eval_stats.filter((pl.col('strategy') == strategy) & (pl.col('k') == k))
        if sub.height > 0:
            return float(sub.sort(_BEST_SORT, descending=_BEST_DESC)['lam'][0])
    if strategy == 'fac_loc':
        return min(cfg.retrieval.lambda_values)
    if strategy == 'mmr':
        return max(cfg.retrieval.lambda_values)
    return 0.5
