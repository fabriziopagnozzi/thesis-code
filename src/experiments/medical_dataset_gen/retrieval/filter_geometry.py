"""Filter retrieval candidates by embedding geometry before evaluation.

This module exists to measure whether the synthetic benchmark produces the
intended query-local geometry, including facet coverage and dominance effects,
before retrieval metrics are computed. It uses nearest-neighbor scoring,
pool-scope filtering, and per-query diagnostic aggregation to decide which
queries are valid for later evaluation.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from tqdm import tqdm

from experiments.medical_dataset_gen.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
    read_parquet,
    write_parquet,
)

from .embed import load_embedding_arrays
from .utils import (
    build_index_maps,
    compute_retrieval_diagnostics,
    get_candidate_pool_indices,
    run_topn_cosine_retrieval,
    select_indices,
)


def run_filter_geometry(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    chunk_documents = read_parquet(paths, 'chunk_documents')
    chunk_memberships = read_parquet(paths, 'chunk_memberships')
    queries = read_parquet(paths, 'queries')
    qrels = read_parquet(paths, 'qrels')
    chunk_vectors, query_vectors, chunk_ids, query_ids = load_embedding_arrays(paths)
    maps = build_index_maps(chunk_documents, chunk_memberships, queries, chunk_ids, query_ids)

    facet_gold = _facet_gold_map(qrels)
    qrels_by_query_chunk = _qrels_by_query_chunk(qrels)
    primary_k = int(cfg.geometry.primary_topk_dominance_k)
    diagnostic_k_values = _diagnostic_k_values(cfg)
    rows = []
    for query in tqdm(
        queries.iter_rows(named=True), total=len(queries), desc='Geometry', dynamic_ncols=True
    ):
        qid = query['query_id']
        qidx = maps['query_id_to_idx'][qid]
        candidate_idx = get_candidate_pool_indices(
            query_id=qid,
            pool_scope=cfg.retrieval.pool_scope,
            n_chunks=len(chunk_ids),
            chunks_by_source_query=maps['chunks_by_source_query'],
            chunks_by_condition=maps['chunks_by_condition'],
            query_condition_id=query.get('condition_id'),
        )
        topn_global, topn_sims = run_topn_cosine_retrieval(
            candidate_indices=candidate_idx,
            chunk_vectors=chunk_vectors,
            query_vector=query_vectors[qidx],
            n=cfg.retrieval.candidate_pool_n,
        )
        topn_chunk_ids = [chunk_ids[i] for i in topn_global]
        topn_set = set(topn_chunk_ids)

        query_facets = facet_gold.get(qid, {})
        query_qrels = qrels_by_query_chunk.get(qid, {})
        if not query_facets:
            continue
        dominant_facet_id = str(query.get('dominant_facet_id') or '')
        facets_present = {
            facet_id: bool(topn_set & set(gold_ids)) for facet_id, gold_ids in query_facets.items()
        }
        n_facets_present = sum(facets_present.values())

        topk_by_k = _topk_diagnostics_by_k(
            topn_chunk_ids=topn_chunk_ids,
            query_qrels=query_qrels,
            query_facets=query_facets,
            dominant_facet_id=dominant_facet_id,
            k_values=diagnostic_k_values,
        )
        primary_topk = topk_by_k[primary_k]
        topk_dominant = _topk_dominant_count(
            topn_chunk_ids=topn_chunk_ids,
            query_qrels=query_qrels,
            k=primary_k,
        )
        topk_retrieved_facets = _topk_retrieved_facets(
            topn_chunk_ids=topn_chunk_ids,
            query_qrels=query_qrels,
            k=primary_k,
        )
        n_topk_retrieved_facets = len(topk_retrieved_facets)
        planned_topk_dominant = int(primary_topk['planned_dominant_count'])
        all_facet_rank = _rank_where_all_facets_first_covered(
            topn_chunk_ids=topn_chunk_ids,
            query_qrels=query_qrels,
            query_facets=query_facets,
        )
        n_distractors = sum(
            1
            for chunk_id in topn_chunk_ids
            if not bool(query_qrels.get(chunk_id, {}).get('is_gold'))
        )
        n_near_miss_distractors = sum(
            1
            for chunk_id in topn_chunk_ids
            if _is_query_near_miss_distractor(query_qrels, chunk_id)
        )
        in_sim, cross_sim = _facet_separation(
            qid=qid,
            query_facets=query_facets,
            chunk_id_to_idx=maps['chunk_id_to_idx'],
            chunk_vectors=chunk_vectors,
        )
        background_diagnostics = _background_outlier_diagnostics(
            topn_chunk_ids=topn_chunk_ids,
            topn_sims=topn_sims,
            query_qrels=query_qrels,
            chunk_id_to_idx=maps['chunk_id_to_idx'],
            chunk_vectors=chunk_vectors,
            expected_background_chunks=(
                cfg.generation.background_outlier_clusters_per_query
                * cfg.generation.background_outlier_cluster_size
            ),
        )

        diagnostics = _topk_vs_facloc_diagnostics(
            topn_global=topn_global,
            topn_sims=topn_sims,
            chunk_vectors=chunk_vectors,
            k=primary_k,
        )

        missing_facet = n_facets_present != len(query_facets)
        weak_topk_dominance = planned_topk_dominant < cfg.geometry.min_topk_dominant_count
        too_many_topk_facets = (
            cfg.geometry.max_topk_retrieved_facets is not None
            and n_topk_retrieved_facets > cfg.geometry.max_topk_retrieved_facets
        )
        weak_facet_separation = (in_sim - cross_sim) < cfg.geometry.min_in_minus_cross_similarity
        too_few_near_miss_distractors = (
            n_near_miss_distractors < cfg.geometry.min_distractors_in_pool
        )
        missing_or_malformed_background_outlier = not bool(
            background_diagnostics['background_outlier_complete']
        )

        passes = not (
            missing_facet
            or weak_topk_dominance
            or too_many_topk_facets
            or weak_facet_separation
            or too_few_near_miss_distractors
            or missing_or_malformed_background_outlier
        )

        rows.append(
            {
                'query_id': qid,
                'pool_scope': cfg.retrieval.pool_scope,
                'pool_size': len(topn_global),
                'topk_dominance_k': primary_k,
                'primary_topk_dominance_k': primary_k,
                'n_facets': len(query_facets),
                'n_facets_present': n_facets_present,
                'all_facets_present': n_facets_present == len(query_facets),
                'topk_dominant_count': topk_dominant,
                'planned_dominant_facet_id': dominant_facet_id,
                'planned_topk_dominant_count': planned_topk_dominant,
                'planned_topk_dominant_fraction': primary_topk['planned_dominant_fraction'],
                'n_topk_retrieved_facets': n_topk_retrieved_facets,
                'max_topk_retrieved_facets': cfg.geometry.max_topk_retrieved_facets,
                'rank_where_all_facets_first_covered': all_facet_rank,
                'all_facets_covered_before_primary_k': (
                    all_facet_rank is not None and all_facet_rank <= primary_k
                ),
                'n_distractors_in_pool': n_distractors,
                'n_near_miss_distractors_in_pool': n_near_miss_distractors,
                'mean_in_facet_similarity': in_sim,
                'mean_cross_facet_similarity': cross_sim,
                'in_minus_cross_similarity': in_sim - cross_sim,
                'passes_filter': passes,
                'fail_missing_facet': missing_facet,
                'fail_weak_topk_dominance': weak_topk_dominance,
                'fail_too_many_topk_facets': too_many_topk_facets,
                'fail_weak_facet_separation': weak_facet_separation,
                'fail_too_few_near_miss_distractors': too_few_near_miss_distractors,
                'fail_missing_or_malformed_background_outlier': (
                    missing_or_malformed_background_outlier
                ),
                'facets_present_json': _json_bool_map(facets_present),
                'topk_retrieved_facets_json': _json_str_list(topk_retrieved_facets),
            }
            | _flatten_topk_diagnostics(topk_by_k)
            | background_diagnostics
            | diagnostics
        )

    df = pl.DataFrame(rows)
    write_parquet(paths, 'geometry_stats', df)
    n_pass = int(df['passes_filter'].sum()) if len(df) else 0
    print(f'[geometry] {n_pass:,}/{len(df):,} queries pass')
    return df


def _facet_gold_map(qrels: pl.DataFrame) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in qrels.filter(pl.col('is_gold')).iter_rows(named=True):
        result[row['query_id']][row['facet_id']].append(row['chunk_id'])
    return result


def _qrels_by_query_chunk(qrels: pl.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in qrels.iter_rows(named=True):
        result[str(row['query_id'])][str(row['chunk_id'])] = row
    return result


def _diagnostic_k_values(cfg: ExperimentCfg) -> list[int]:
    return sorted({
        int(k)
        for k in [
            *cfg.retrieval.k_values,
            cfg.geometry.topk_dominance_k,
            cfg.geometry.primary_topk_dominance_k,
        ]
    })


def _topk_diagnostics_by_k(
    *,
    topn_chunk_ids: list[str],
    query_qrels: dict[str, dict[str, Any]],
    query_facets: dict[str, list[str]],
    dominant_facet_id: str,
    k_values: list[int],
) -> dict[int, dict[str, int | float | list[str]]]:
    rows = {}
    for k in k_values:
        counts = _topk_facet_counts(
            topn_chunk_ids=topn_chunk_ids,
            query_qrels=query_qrels,
            k=k,
        )
        retrieved_facets = sorted(counts)
        n_selected = min(k, len(topn_chunk_ids))
        denominator = max(n_selected, 1)
        most_common_count = counts.most_common(1)[0][1] if counts else 0
        planned_count = counts.get(dominant_facet_id, 0)
        n_facets = len(query_facets)
        rows[k] = {
            'dominant_count': most_common_count,
            'dominant_fraction': most_common_count / denominator,
            'planned_dominant_count': planned_count,
            'planned_dominant_fraction': planned_count / denominator,
            'n_retrieved_facets': len(retrieved_facets),
            'facet_coverage': len(retrieved_facets) / n_facets if n_facets else 0.0,
            'all_facets_covered': len(retrieved_facets) == n_facets,
            'retrieved_facets': retrieved_facets,
        }
    return rows


def _flatten_topk_diagnostics(topk_by_k: dict[int, dict[str, object]]) -> dict[str, object]:
    flat: dict[str, object] = {}
    for k, row in topk_by_k.items():
        prefix = f'topk_{k}'
        flat[f'{prefix}_dominant_count'] = row['dominant_count']
        flat[f'{prefix}_dominant_fraction'] = row['dominant_fraction']
        flat[f'{prefix}_planned_dominant_count'] = row['planned_dominant_count']
        flat[f'{prefix}_planned_dominant_fraction'] = row['planned_dominant_fraction']
        flat[f'{prefix}_n_retrieved_facets'] = row['n_retrieved_facets']
        flat[f'{prefix}_facet_coverage'] = row['facet_coverage']
        flat[f'{prefix}_all_facets_covered'] = row['all_facets_covered']
        flat[f'{prefix}_retrieved_facets_json'] = _json_str_list(list(row['retrieved_facets']))
    return flat


def _topk_facet_counts(
    topn_chunk_ids: list[str],
    query_qrels: dict[str, dict[str, Any]],
    k: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for chunk_id in topn_chunk_ids[:k]:
        row = query_qrels.get(chunk_id, {})
        if row.get('is_gold') and row.get('facet_id'):
            counts[str(row['facet_id'])] += 1
    return counts


def _topk_dominant_count(
    topn_chunk_ids: list[str],
    query_qrels: dict[str, dict[str, Any]],
    k: int,
) -> int:
    counts = _topk_facet_counts(
        topn_chunk_ids=topn_chunk_ids,
        query_qrels=query_qrels,
        k=k,
    )
    return counts.most_common(1)[0][1] if counts else 0


def _topk_retrieved_facets(
    topn_chunk_ids: list[str],
    query_qrels: dict[str, dict[str, Any]],
    k: int,
) -> list[str]:
    facets = {
        str(row['facet_id'])
        for chunk_id in topn_chunk_ids[:k]
        if (row := query_qrels.get(chunk_id, {})).get('is_gold') and row.get('facet_id')
    }
    return sorted(facets)


def _rank_where_all_facets_first_covered(
    *,
    topn_chunk_ids: list[str],
    query_qrels: dict[str, dict[str, Any]],
    query_facets: dict[str, list[str]],
) -> int | None:
    expected = set(query_facets)
    seen: set[str] = set()
    for rank, chunk_id in enumerate(topn_chunk_ids, start=1):
        row = query_qrels.get(chunk_id, {})
        if row.get('is_gold') and row.get('facet_id') in expected:
            seen.add(str(row['facet_id']))
            if seen == expected:
                return rank
    return None


def _facet_separation(
    qid: str,
    query_facets: dict[str, list[str]],
    chunk_id_to_idx: dict[str, int],
    chunk_vectors: NDArray[np.float32],
) -> tuple[float, float]:
    _ = qid
    gold_ids = []
    labels = []
    for facet_id, ids in query_facets.items():
        for chunk_id in ids:
            if chunk_id in chunk_id_to_idx:
                gold_ids.append(chunk_id)
                labels.append(facet_id)
    if len(gold_ids) < 2:
        return 0.0, 0.0

    vectors = chunk_vectors[[chunk_id_to_idx[chunk_id] for chunk_id in gold_ids]]
    sim = vectors @ vectors.T
    labels_arr = np.array(labels)
    same = labels_arr[:, None] == labels_arr[None, :]
    not_self = ~np.eye(len(labels_arr), dtype=bool)
    in_vals = sim[same & not_self]
    cross_vals = sim[~same & not_self]
    in_sim = float(in_vals.mean()) if len(in_vals) else 0.0
    cross_sim = float(cross_vals.mean()) if len(cross_vals) else 0.0
    return in_sim, cross_sim


def _background_outlier_diagnostics(
    topn_chunk_ids: list[str],
    topn_sims: NDArray[np.float32],
    query_qrels: dict[str, dict[str, Any]],
    chunk_id_to_idx: dict[str, int],
    chunk_vectors: NDArray[np.float32],
    expected_background_chunks: int,
) -> dict[str, float | int | bool | None]:
    background_positions = [
        idx
        for idx, chunk_id in enumerate(topn_chunk_ids)
        if _is_background_outlier(query_qrels, chunk_id)
    ]
    background_ids = [topn_chunk_ids[idx] for idx in background_positions]
    background_clusters = {
        str(query_qrels[chunk_id].get('cluster_id')) for chunk_id in background_ids
    }
    gold_positions = [
        idx for idx, chunk_id in enumerate(topn_chunk_ids) if _is_query_gold(query_qrels, chunk_id)
    ]

    query_to_background = (
        float(np.asarray(topn_sims)[background_positions].mean()) if background_positions else None
    )
    query_to_gold = float(np.asarray(topn_sims)[gold_positions].mean()) if gold_positions else None
    margin = (
        float(query_to_gold - query_to_background)
        if query_to_gold is not None and query_to_background is not None
        else None
    )

    background_in_cluster_similarity = _mean_same_cluster_similarity(
        chunk_ids=background_ids,
        query_qrels=query_qrels,
        chunk_id_to_idx=chunk_id_to_idx,
        chunk_vectors=chunk_vectors,
    )
    ranks = [pos + 1 for pos in background_positions]
    expected = int(expected_background_chunks)
    complete = len(background_ids) >= expected if expected > 0 else True

    return {
        'n_background_outliers_in_pool': len(background_ids),
        'n_background_outlier_clusters_in_pool': len(background_clusters),
        'background_outlier_complete': complete,
        'background_outlier_mean_in_cluster_similarity': background_in_cluster_similarity,
        'query_to_background_outlier_mean': query_to_background,
        'query_to_gold_mean': query_to_gold,
        'gold_minus_background_outlier_similarity_margin': margin,
        'background_outlier_first_rank': min(ranks) if ranks else None,
        'background_outlier_median_rank': float(np.median(ranks)) if ranks else None,
    }


def _is_query_gold(query_qrels: dict[str, dict[str, Any]], chunk_id: str) -> bool:
    return bool(query_qrels.get(chunk_id, {}).get('is_gold'))


def _is_background_outlier(query_qrels: dict[str, dict[str, Any]], chunk_id: str) -> bool:
    return query_qrels.get(chunk_id, {}).get('cluster_role') == 'background_outlier'


def _is_query_near_miss_distractor(query_qrels: dict[str, dict[str, Any]], chunk_id: str) -> bool:
    row = query_qrels.get(chunk_id)
    return (
        bool(row)
        and not bool(row.get('is_gold'))
        and row.get('cluster_role') != 'background_outlier'
    )


def _mean_same_cluster_similarity(
    chunk_ids: list[str],
    query_qrels: dict[str, dict[str, Any]],
    chunk_id_to_idx: dict[str, int],
    chunk_vectors: NDArray[np.float32],
) -> float | None:
    ids_by_cluster: dict[str, list[str]] = defaultdict(list)
    for chunk_id in chunk_ids:
        if chunk_id in chunk_id_to_idx:
            ids_by_cluster[str(query_qrels[chunk_id].get('cluster_id'))].append(chunk_id)

    values = []
    for cluster_ids in ids_by_cluster.values():
        if len(cluster_ids) < 2:
            continue
        vectors = chunk_vectors[[chunk_id_to_idx[chunk_id] for chunk_id in cluster_ids]]
        sim = vectors @ vectors.T
        not_self = ~np.eye(len(cluster_ids), dtype=bool)
        values.extend(float(value) for value in sim[not_self])
    return float(np.mean(values)) if values else None


def _topk_vs_facloc_diagnostics(
    topn_global: NDArray[np.intp],
    topn_sims: NDArray[np.float32],
    chunk_vectors: NDArray[np.float32],
    k: int,
) -> dict[str, float]:
    if len(topn_global) == 0:
        return {
            'fac_topk': 0.0,
            'fac_facloc': 0.0,
            'avg_cos_topk': 0.0,
            'avg_cos_facloc': 0.0,
            'jaccard_topk_facloc': 0.0,
        }
    candidate_vectors = chunk_vectors[topn_global]
    sim_matrix = candidate_vectors @ candidate_vectors.T
    sim_to_query = topn_sims.astype(np.float32)
    topk = select_indices('top_k', sim_to_query, sim_matrix, k=k, lam=None)
    fl = select_indices('fac_loc', sim_to_query, sim_matrix, k=k, lam=0.3)
    topk_diag = compute_retrieval_diagnostics(topk, sim_to_query, sim_matrix)
    fl_diag = compute_retrieval_diagnostics(fl, sim_to_query, sim_matrix, topk_local_indices=topk)
    return {
        'fac_topk': topk_diag['fac_cov_score'],
        'fac_facloc': fl_diag['fac_cov_score'],
        'avg_cos_topk': topk_diag['avg_cos'],
        'avg_cos_facloc': fl_diag['avg_cos'],
        'jaccard_topk_facloc': fl_diag['jaccard_vs_topk'],
    }


def _json_bool_map(value: dict[str, bool]) -> str:
    return json.dumps(value, sort_keys=True)


def _json_str_list(value: list[str]) -> str:
    return json.dumps(value, sort_keys=True)


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
    run_filter_geometry(cfg, paths)
