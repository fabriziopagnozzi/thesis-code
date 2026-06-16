"""Evaluate retrieval strategies on the synthetic medical benchmark.

This module exists to score top-k, MMR, and facility-location against the gold
facets and distractors generated earlier in the pipeline. It uses shared
candidate-pool logic, per-query metric aggregation, and redundancy-aware
ranking metrics so the benchmark can expose coverage differences rather than
just nearest-neighbor accuracy.
"""

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

ALPHA_NDCG_REDUNDANCY = 0.5


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
    stats = (
        results
        .group_by('strategy', 'lam', 'k')
        .agg(
            pl.col('query_id').n_unique().alias('n_queries'),
            pl.col('gold_precision').mean().alias('Precision@k'),
            pl.col('gold_recall').mean().alias('Recall@k'),
            pl.col('gold_f1').mean().alias('F1@k'),
            pl.col('average_precision_at_k').mean().alias('MAP@k'),
            pl.col('facet_coverage').mean().alias('FacetCoverage@k'),
            pl.col('weighted_facet_coverage').mean().alias('MeanFacetRecall@k'),
            pl.col('facet_mrr_at_k').mean().alias('FacetMRR@k'),
            pl.col('alpha_ndcg').mean().alias('alpha-nDCG@k'),
            pl.col('distractor_rate').mean().alias('DistractorRate'),
            pl.col('near_miss_distractor_rate').mean().alias('NearMissDistractorRate'),
            pl.col('background_outlier_rate').mean().alias('BackgroundOutlierRate'),
            pl.col('any_distractor_rate').mean().alias('AnyDistractorRate'),
            pl.col('dominant_facet_rate').mean().alias('DominantFacetRate'),
            pl.col('redundant_gold_rate').mean().alias('RedundantGoldRate'),
            pl.col('fac_cov_score').mean().alias('fac'),
            pl.col('avg_cos').mean().alias('avg_cos'),
            pl.col('jaccard_vs_topk').mean().alias('jac'),
        )
        .sort('k', 'strategy', 'lam')
    )
    return stats.select(
        'strategy',
        'lam',
        'k',
        'n_queries',
        'Precision@k',
        'Recall@k',
        'F1@k',
        'MAP@k',
        'FacetCoverage@k',
        'MeanFacetRecall@k',
        'FacetMRR@k',
        'alpha-nDCG@k',
        'DistractorRate',
        'NearMissDistractorRate',
        'BackgroundOutlierRate',
        'AnyDistractorRate',
        'DominantFacetRate',
        'RedundantGoldRate',
        'fac',
        'avg_cos',
        'jac',
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
    relevance = _relevance_metrics(
        selected_chunk_ids=selected_chunk_ids,
        all_gold_ids=all_gold_ids,
    )
    facet_coverage = _facet_coverage_metrics(
        selected_chunk_ids=selected_chunk_ids,
        facet_to_gold=facet_to_gold,
    )
    diversified_ranking = _diversified_ranking_metrics(
        selected_chunk_ids=selected_chunk_ids,
        chunk_by_id=chunk_by_id,
        facet_to_gold=facet_to_gold,
        all_gold_ids=all_gold_ids,
    )
    redundancy = _redundancy_metrics(
        selected_chunk_ids=selected_chunk_ids,
        chunk_by_id=chunk_by_id,
        all_gold_ids=all_gold_ids,
        dominant_facet_id=dominant_facet_id,
        n_selected_gold=int(relevance['n_selected_gold']),
        n_facet_hits=int(facet_coverage['n_unique_gold_facets']),
    )
    selected_rows = [chunk_by_id[cid] for cid in selected_chunk_ids]

    return {
        **relevance,
        **facet_coverage,
        **diversified_ranking,
        **redundancy,
        'n_unique_hadms': len({
            row.get('admission_id') for row in selected_rows if row.get('admission_id')
        }),
    }


def _relevance_metrics(
    selected_chunk_ids: list[str],
    all_gold_ids: set[str],
) -> dict[str, float | int]:
    n_selected = len(selected_chunk_ids)
    n_selected_gold = sum(1 for chunk_id in selected_chunk_ids if chunk_id in all_gold_ids)
    gold_precision = n_selected_gold / n_selected if n_selected else 0.0
    gold_recall = n_selected_gold / len(all_gold_ids) if all_gold_ids else 0.0
    return {
        'gold_precision': float(gold_precision),
        'gold_recall': float(gold_recall),
        'gold_f1': float(_harmonic_mean(gold_precision, gold_recall)),
        'average_precision_at_k': average_precision_at_k(
            selected_chunk_ids=selected_chunk_ids,
            all_gold_ids=all_gold_ids,
        ),
        'n_selected': n_selected,
        'n_selected_gold': n_selected_gold,
    }


def _facet_coverage_metrics(
    selected_chunk_ids: list[str],
    facet_to_gold: dict[str, list[str]],
) -> dict[str, float | int]:
    selected = set(selected_chunk_ids)
    facet_gold_sets = {
        facet_id: set(gold_ids) for facet_id, gold_ids in facet_to_gold.items() if gold_ids
    }
    facet_hits = {facet_id for facet_id, gold_ids in facet_gold_sets.items() if selected & gold_ids}
    n_facets = len(facet_to_gold)
    n_facet_hits = len(facet_hits)
    facet_coverage = n_facet_hits / n_facets if n_facets else 0.0
    mean_facet_recall = (
        np.mean([
            len(selected & gold_ids) / len(gold_ids)
            for gold_ids in facet_gold_sets.values()
            if gold_ids
        ])
        if facet_gold_sets
        else 0.0
    )
    facet_hit_density = n_facet_hits / len(selected_chunk_ids) if selected_chunk_ids else 0.0
    facet_f1 = _harmonic_mean(facet_hit_density, facet_coverage)

    return {
        'facet_coverage': float(facet_coverage),
        'weighted_facet_coverage': float(mean_facet_recall),
        'facet_hit_density': float(facet_hit_density),
        'unique_facet_rate': float(facet_hit_density),
        'facet_f1': float(facet_f1),
        # Backward-compatible raw names; summary tables no longer label these as AP/AF1.
        'aspect_precision': float(facet_hit_density),
        'aspect_f1': float(facet_f1),
        'n_unique_gold_facets': n_facet_hits,
        'n_total_facets': n_facets,
    }


def _diversified_ranking_metrics(
    selected_chunk_ids: list[str],
    chunk_by_id: dict[str, dict[str, Any]],
    facet_to_gold: dict[str, list[str]],
    all_gold_ids: set[str],
) -> dict[str, float]:
    facet_mrr_at_k = _facet_mrr(
        selected_chunk_ids=selected_chunk_ids,
        chunk_by_id=chunk_by_id,
        facet_ids=list(facet_to_gold),
        all_gold_ids=all_gold_ids,
    )
    return {
        'alpha_ndcg': _alpha_ndcg(
            selected_chunk_ids=selected_chunk_ids,
            chunk_by_id=chunk_by_id,
            facet_to_gold=facet_to_gold,
            all_gold_ids=all_gold_ids,
            alpha=ALPHA_NDCG_REDUNDANCY,
        ),
        'facet_mrr_at_k': facet_mrr_at_k,
        'facet_mrr': facet_mrr_at_k,
    }


def _redundancy_metrics(
    selected_chunk_ids: list[str],
    chunk_by_id: dict[str, dict[str, Any]],
    all_gold_ids: set[str],
    dominant_facet_id: str,
    n_selected_gold: int,
    n_facet_hits: int,
) -> dict[str, float | int]:
    n_selected = len(selected_chunk_ids)
    non_gold_rows = [
        chunk_by_id[chunk_id] for chunk_id in selected_chunk_ids if chunk_id not in all_gold_ids
    ]
    non_gold_count = len(non_gold_rows)
    background_outlier_count = sum(
        1 for row in non_gold_rows if row.get('cluster_role') == 'background_outlier'
    )
    near_miss_distractor_count = non_gold_count - background_outlier_count
    dominant_count = sum(
        1
        for chunk_id in selected_chunk_ids
        if chunk_by_id[chunk_id].get('facet_id') == dominant_facet_id
    )
    selected_facet_counts = Counter(
        chunk_by_id[chunk_id].get('facet_id')
        for chunk_id in selected_chunk_ids
        if chunk_id in all_gold_ids and chunk_by_id[chunk_id].get('facet_id')
    )
    max_facet_concentration = (
        selected_facet_counts.most_common(1)[0][1] / n_selected
        if n_selected and selected_facet_counts
        else 0.0
    )
    redundant_gold_count = max(n_selected_gold - n_facet_hits, 0)
    dominant_facet_rate = dominant_count / n_selected if n_selected else 0.0

    return {
        'distractor_rate': non_gold_count / n_selected if n_selected else 0.0,
        'any_distractor_rate': non_gold_count / n_selected if n_selected else 0.0,
        'near_miss_distractor_rate': (
            near_miss_distractor_count / n_selected if n_selected else 0.0
        ),
        'background_outlier_rate': background_outlier_count / n_selected if n_selected else 0.0,
        'dominant_facet_rate': float(dominant_facet_rate),
        'dominant_cluster_concentration': float(dominant_facet_rate),
        'max_facet_concentration': float(max_facet_concentration),
        'redundant_gold_rate': redundant_gold_count / n_selected if n_selected else 0.0,
        'n_selected_non_gold': non_gold_count,
        'n_selected_near_miss_distractors': near_miss_distractor_count,
        'n_selected_background_outliers': background_outlier_count,
        'n_redundant_gold': redundant_gold_count,
    }


def average_precision_at_k(
    selected_chunk_ids: list[str],
    all_gold_ids: set[str],
    k: int | None = None,
) -> float:
    rank_cutoff = len(selected_chunk_ids) if k is None else k
    denominator = min(len(all_gold_ids), rank_cutoff)
    if denominator <= 0:
        return 0.0

    n_hits = 0
    precision_sum = 0.0
    for rank, chunk_id in enumerate(selected_chunk_ids[:rank_cutoff], start=1):
        if chunk_id not in all_gold_ids:
            continue
        n_hits += 1
        precision_sum += n_hits / rank
    return float(precision_sum / denominator)


def _harmonic_mean(left: float, right: float) -> float:
    denom = left + right
    return 0.0 if denom <= 0 else 2 * left * right / denom


def _alpha_ndcg(
    selected_chunk_ids: list[str],
    chunk_by_id: dict[str, dict[str, Any]],
    facet_to_gold: dict[str, list[str]],
    all_gold_ids: set[str],
    alpha: float,
) -> float:
    """alpha-nDCG with facet_id as the subtopic label.

    Repeated gold chunks from the same facet receive diminishing gain, which
    makes this a ranking-sensitive coverage metric for the synthetic benchmark.
    """
    selected_dcg = _alpha_dcg(
        selected_chunk_ids=selected_chunk_ids,
        chunk_by_id=chunk_by_id,
        all_gold_ids=all_gold_ids,
        alpha=alpha,
    )
    ideal_labels = _ideal_alpha_labels(facet_to_gold, k=len(selected_chunk_ids), alpha=alpha)
    ideal_dcg = _alpha_dcg_from_labels(ideal_labels, alpha=alpha)
    return float(selected_dcg / ideal_dcg) if ideal_dcg > 0 else 0.0


def _alpha_dcg(
    selected_chunk_ids: list[str],
    chunk_by_id: dict[str, dict[str, Any]],
    all_gold_ids: set[str],
    alpha: float,
) -> float:
    labels = [
        str(chunk_by_id[chunk_id].get('facet_id'))
        if chunk_id in all_gold_ids and chunk_by_id[chunk_id].get('facet_id')
        else None
        for chunk_id in selected_chunk_ids
    ]
    return _alpha_dcg_from_labels(labels, alpha=alpha)


def _alpha_dcg_from_labels(labels: list[str | None], alpha: float) -> float:
    counts: Counter[str] = Counter()
    total = 0.0
    for rank, facet_id in enumerate(labels, start=1):
        if facet_id is None:
            continue
        gain = (1 - alpha) ** counts[facet_id]
        counts[facet_id] += 1
        total += gain / np.log2(rank + 1)
    return float(total)


def _ideal_alpha_labels(
    facet_to_gold: dict[str, list[str]],
    k: int,
    alpha: float,
) -> list[str]:
    remaining = {facet_id: len(gold_ids) for facet_id, gold_ids in facet_to_gold.items()}
    counts: Counter[str] = Counter()
    labels = []
    for _ in range(k):
        candidates = [
            (facet_id, (1 - alpha) ** counts[facet_id])
            for facet_id, n_remaining in remaining.items()
            if n_remaining > 0
        ]
        if not candidates:
            break
        facet_id, _ = max(candidates, key=lambda item: item[1])
        labels.append(facet_id)
        remaining[facet_id] -= 1
        counts[facet_id] += 1
    return labels


def _facet_mrr(
    selected_chunk_ids: list[str],
    chunk_by_id: dict[str, dict[str, Any]],
    facet_ids: list[str],
    all_gold_ids: set[str],
) -> float:
    if not facet_ids:
        return 0.0
    first_rank: dict[str, int] = {}
    for rank, chunk_id in enumerate(selected_chunk_ids, start=1):
        if chunk_id not in all_gold_ids:
            continue
        facet_id = chunk_by_id[chunk_id].get('facet_id')
        if facet_id:
            first_rank.setdefault(str(facet_id), rank)
    reciprocal_ranks = [
        1 / first_rank[facet_id] if facet_id in first_rank else 0.0 for facet_id in facet_ids
    ]
    return float(np.mean(reciprocal_ranks))


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
