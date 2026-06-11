"""Filter retrieval candidates by embedding geometry before evaluation.

This module exists to measure whether the synthetic benchmark produces the
intended query-local geometry, including facet coverage and dominance effects,
before retrieval metrics are computed. It uses nearest-neighbor scoring,
pool-scope filtering, and per-query diagnostic aggregation to decide which
queries are valid for later evaluation.
"""

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
from experiments.medical_dataset_gen.retrieval.embed import load_embedding_arrays
from experiments.medical_dataset_gen.retrieval.utils import (
    build_index_maps,
    candidate_pool_indices,
    retrieval_diagnostics,
    select_indices,
    topn_by_query,
)


def run_filter_geometry(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    chunks = read_parquet(paths, 'chunks')
    queries = read_parquet(paths, 'queries')
    qrels = read_parquet(paths, 'qrels')
    chunk_vectors, query_vectors, chunk_ids, query_ids = load_embedding_arrays(paths)
    maps = build_index_maps(chunks, queries, chunk_ids, query_ids)

    facet_gold = _facet_gold_map(qrels)
    rows = []
    for query in tqdm(
        queries.iter_rows(named=True), total=len(queries), desc='Geometry', dynamic_ncols=True
    ):
        qid = query['query_id']
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
        topn_chunk_ids = [chunk_ids[i] for i in topn_global]
        topn_set = set(topn_chunk_ids)

        query_facets = facet_gold.get(qid, {})
        if not query_facets:
            continue
        facets_present = {
            facet_id: bool(topn_set & set(gold_ids)) for facet_id, gold_ids in query_facets.items()
        }
        n_facets_present = sum(facets_present.values())

        topk_dominant = _topk_dominant_count(
            topn_chunk_ids=topn_chunk_ids,
            chunk_by_id=maps['chunk_by_id'],
            k=cfg.geometry.topk_dominance_k,
        )
        n_distractors = sum(
            1 for chunk_id in topn_chunk_ids if not bool(maps['chunk_by_id'][chunk_id]['is_gold'])
        )
        in_sim, cross_sim = _facet_separation(
            qid=qid,
            query_facets=query_facets,
            chunk_id_to_idx=maps['chunk_id_to_idx'],
            chunk_vectors=chunk_vectors,
        )

        diagnostics = _topk_vs_facloc_diagnostics(
            topn_global=topn_global,
            topn_sims=topn_sims,
            chunk_vectors=chunk_vectors,
            k=cfg.geometry.topk_dominance_k,
        )

        passes = (
            n_facets_present == len(query_facets)
            and topk_dominant >= cfg.geometry.min_topk_dominant_count
            and (in_sim - cross_sim) >= cfg.geometry.min_in_minus_cross_similarity
            and n_distractors >= cfg.geometry.min_distractors_in_pool
        )

        rows.append(
            {
                'query_id': qid,
                'pool_scope': cfg.retrieval.pool_scope,
                'pool_size': len(topn_global),
                'n_facets': len(query_facets),
                'n_facets_present': n_facets_present,
                'all_facets_present': n_facets_present == len(query_facets),
                'topk_dominant_count': topk_dominant,
                'n_distractors_in_pool': n_distractors,
                'mean_in_facet_similarity': in_sim,
                'mean_cross_facet_similarity': cross_sim,
                'in_minus_cross_similarity': in_sim - cross_sim,
                'passes_filter': passes,
                'facets_present_json': _json_bool_map(facets_present),
            }
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


def _topk_dominant_count(
    topn_chunk_ids: list[str],
    chunk_by_id: dict[str, dict[str, Any]],
    k: int,
) -> int:
    counts: Counter[str] = Counter()
    for chunk_id in topn_chunk_ids[:k]:
        row = chunk_by_id[chunk_id]
        if row['is_gold'] and row['facet_id']:
            counts[row['facet_id']] += 1
    return counts.most_common(1)[0][1] if counts else 0


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
    topk_diag = retrieval_diagnostics(topk, sim_to_query, sim_matrix)
    fl_diag = retrieval_diagnostics(fl, sim_to_query, sim_matrix, topk_local_indices=topk)
    return {
        'fac_topk': topk_diag['fac_cov_score'],
        'fac_facloc': fl_diag['fac_cov_score'],
        'avg_cos_topk': topk_diag['avg_cos'],
        'avg_cos_facloc': fl_diag['avg_cos'],
        'jaccard_topk_facloc': fl_diag['jaccard_vs_topk'],
    }


def _json_bool_map(value: dict[str, bool]) -> str:
    import json

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
