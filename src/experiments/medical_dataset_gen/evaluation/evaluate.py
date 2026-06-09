from collections import Counter, defaultdict
from typing import Any

import numpy as np
import polars as pl
from tqdm import tqdm

from experiments.medical_dataset_gen.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
    read_parquet,
    write_parquet,
)
from experiments.medical_dataset_gen.retrieval.embed import load_embedding_arrays
from experiments.medical_dataset_gen.retrieval.utils import (
    build_index_maps,
    candidate_pool_indices,
    retrieval_diagnostics,
    select_indices,
    topn_by_query,
)


def run_evaluate(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    chunks = read_parquet(paths, 'chunks')
    queries = read_parquet(paths, 'queries')
    qrels = read_parquet(paths, 'qrels')
    geometry = read_parquet(paths, 'geometry_stats')
    _assert_pool_scope_match(geometry, cfg.retrieval.pool_scope, table_name='geometry_stats')
    chunk_vectors, query_vectors, chunk_ids, query_ids = load_embedding_arrays(paths)
    maps = build_index_maps(chunks, queries, chunk_ids, query_ids)

    facet_gold = _facet_gold_map(qrels)
    gold_by_query = {
        qid: {chunk_id for ids in facet_map.values() for chunk_id in ids}
        for qid, facet_map in facet_gold.items()
    }
    pass_map = dict(
        zip(geometry['query_id'].to_list(), geometry['passes_filter'].to_list(), strict=True)
    )

    rows: list[dict[str, Any]] = []
    for query in tqdm(
        queries.iter_rows(named=True), total=len(queries), desc='Evaluating', dynamic_ncols=True
    ):
        qid = query['query_id']
        if cfg.retrieval.only_pass_geometry and not bool(pass_map.get(qid, False)):
            continue

        query_facet_gold = facet_gold.get(qid)
        query_all_gold = gold_by_query.get(qid)
        if not query_facet_gold or not query_all_gold:
            continue

        qidx = maps['query_id_to_idx'][qid]
        candidate_idx = candidate_pool_indices(
            query_id=qid,
            pool_scope=cfg.retrieval.pool_scope,
            n_chunks=len(chunk_ids),
            chunks_by_source_query=maps['chunks_by_source_query'],
            chunks_by_condition=maps['chunks_by_condition'],
            query_condition_id=query.get('condition_id'),
        )
        topn_global, topn_sims = topn_by_query(
            candidate_indices=candidate_idx,
            chunk_vectors=chunk_vectors,
            query_vector=query_vectors[qidx],
            n=cfg.retrieval.candidate_pool_n,
        )
        if len(topn_global) == 0:
            continue

        candidate_vectors = chunk_vectors[topn_global]
        sim_matrix = candidate_vectors @ candidate_vectors.T
        sim_to_query = topn_sims.astype(np.float32)
        candidate_chunk_ids = [chunk_ids[i] for i in topn_global]
        topk_by_k: dict[int, np.ndarray] = {}

        for k in cfg.retrieval.k_values:
            if k > len(candidate_chunk_ids):
                continue
            topk_by_k[k] = select_indices('top_k', sim_to_query, sim_matrix, k=k, lam=None)

        for strategy in cfg.retrieval.strategies:
            lam_values: list[None] | list[float] = (
                [None] if strategy == 'top_k' else cfg.retrieval.lambda_values
            )
            for lam in lam_values:
                for k in cfg.retrieval.k_values:
                    if k > len(candidate_chunk_ids):
                        continue
                    selected_local = select_indices(
                        strategy=strategy,
                        sim_to_query=sim_to_query,
                        sim_matrix=sim_matrix,
                        k=k,
                        lam=lam,
                        mmr_window=cfg.retrieval.mmr_window,
                    )
                    selected_chunk_ids = [candidate_chunk_ids[int(i)] for i in selected_local]
                    topk_ref = topk_by_k[k]
                    row = {
                        'query_id': qid,
                        'query_type': query['query_type'],
                        'condition_id': query['condition_id'],
                        'split': query['split'],
                        'strategy': strategy,
                        'k': k,
                        'lam': lam,
                        'pool_scope': cfg.retrieval.pool_scope,
                        'pool_size': len(candidate_chunk_ids),
                        **_retrieval_metrics(
                            selected_chunk_ids=selected_chunk_ids,
                            chunk_by_id=maps['chunk_by_id'],
                            facet_to_gold=query_facet_gold,
                            all_gold_ids=query_all_gold,
                            dominant_facet_id=query['dominant_facet_id'],
                        ),
                        **retrieval_diagnostics(
                            selected_local,
                            sim_to_query,
                            sim_matrix,
                            topk_local_indices=topk_ref if strategy != 'top_k' else None,
                        ),
                    }
                    rows.append(row)

    results = pl.DataFrame(rows)
    write_parquet(paths, 'evaluation_results', results)
    stats = summarize_results(results)
    write_parquet(paths, 'evaluation_stats', stats)
    print(stats)
    return results


def _assert_pool_scope_match(
    df: pl.DataFrame,
    expected_pool_scope: str,
    table_name: str,
) -> None:
    if 'pool_scope' not in df.columns or df.is_empty():
        return
    scopes = sorted({str(value) for value in df['pool_scope'].drop_nulls().to_list()})
    if not scopes:
        return
    if scopes != [expected_pool_scope]:
        raise ValueError(
            f'{table_name} was generated with pool_scope={scopes}, '
            f'but the current config expects pool_scope={expected_pool_scope!r}. '
            'Rerun from the geometry stage, or use a config matching the stored artifacts.'
        )


def summarize_results(results: pl.DataFrame) -> pl.DataFrame:
    if len(results) == 0:
        return pl.DataFrame()
    return (
        results
        .group_by('strategy', 'lam', 'k')
        .agg(
            pl.col('facet_coverage').mean().alias('FC'),
            pl.col('weighted_facet_coverage').mean().alias('WFC'),
            pl.col('gold_precision').mean().alias('GP'),
            pl.col('gold_recall').mean().alias('GR'),
            pl.col('distractor_rate').mean().alias('DR'),
            pl.col('dominant_cluster_concentration').mean().alias('DCC'),
            pl.col('fac_cov_score').mean().alias('fac'),
            pl.col('avg_cos').mean().alias('avg_cos'),
            pl.col('jaccard_vs_topk').mean().alias('jac'),
            pl.col('query_id').n_unique().alias('n_queries'),
        )
        .sort('k', 'strategy', 'lam')
    )


def _facet_gold_map(qrels: pl.DataFrame) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in qrels.filter(pl.col('is_gold')).iter_rows(named=True):
        result[row['query_id']][row['facet_id']].append(row['chunk_id'])
    return result


def _retrieval_metrics(
    selected_chunk_ids: list[str],
    chunk_by_id: dict[str, dict[str, Any]],
    facet_to_gold: dict[str, list[str]],
    all_gold_ids: set[str],
    dominant_facet_id: str,
) -> dict[str, float | int]:
    selected = set(selected_chunk_ids)
    selected_gold = selected & all_gold_ids

    facet_hits = {
        facet_id for facet_id, gold_ids in facet_to_gold.items() if selected & set(gold_ids)
    }
    facet_coverage = len(facet_hits) / len(facet_to_gold) if facet_to_gold else 0.0
    weighted = (
        np.mean([
            len(selected & set(gold_ids)) / len(gold_ids)
            for gold_ids in facet_to_gold.values()
            if gold_ids
        ])
        if facet_to_gold
        else 0.0
    )

    selected_rows = [chunk_by_id[cid] for cid in selected_chunk_ids]
    non_gold_count = sum(1 for cid in selected_chunk_ids if cid not in all_gold_ids)
    dominant_count = sum(
        1 for cid in selected_chunk_ids if chunk_by_id[cid].get('facet_id') == dominant_facet_id
    )
    selected_facet_counts = Counter(
        chunk_by_id[cid].get('facet_id')
        for cid in selected_chunk_ids
        if cid in all_gold_ids and chunk_by_id[cid].get('facet_id')
    )
    max_facet_concentration = (
        selected_facet_counts.most_common(1)[0][1] / len(selected_chunk_ids)
        if selected_chunk_ids and selected_facet_counts
        else 0.0
    )

    return {
        'facet_coverage': float(facet_coverage),
        'weighted_facet_coverage': float(weighted),
        'gold_precision': len(selected_gold) / len(selected_chunk_ids)
        if selected_chunk_ids
        else 0.0,
        'gold_recall': len(selected_gold) / len(all_gold_ids) if all_gold_ids else 0.0,
        'distractor_rate': non_gold_count / len(selected_chunk_ids) if selected_chunk_ids else 0.0,
        'dominant_cluster_concentration': dominant_count / len(selected_chunk_ids)
        if selected_chunk_ids
        else 0.0,
        'max_facet_concentration': float(max_facet_concentration),
        'n_selected': len(selected_chunk_ids),
        'n_selected_gold': len(selected_gold),
        'n_selected_non_gold': non_gold_count,
        'n_unique_hadms': len({row['admission_id'] for row in selected_rows}),
    }


if __name__ == '__main__':
    from experiments.medical_dataset_gen.global_configs import (
        dump_effective_config,
        load_config_from_cli,
        paths_for,
        setup_logging,
    )

    cfg = load_config_from_cli()
    paths = paths_for(cfg)
    setup_logging(paths)
    dump_effective_config(cfg, paths)
    run_evaluate(cfg, paths)
