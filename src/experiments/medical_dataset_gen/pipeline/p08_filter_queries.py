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
from typing import TypedDict

import numpy as np
import polars as pl
from numpy.typing import NDArray
from tqdm import tqdm

from experiments.medical_dataset_gen.evaluation.retrieval_utils import (
    build_index_maps,
    build_query_to_facet_gold_map,
    compute_retrieval_diagnostics,
    get_candidate_pool_indices,
    get_qrels_by_query_chunk,
    is_query_gold,
    load_embedding_arrays,
    run_topn_cosine_retrieval,
    select_indices,
)
from experiments.medical_dataset_gen.global_config import (
    ExperimentCfg,
    GeometryFilterCfg,
    MedicalDatasetGenPaths,
)
from experiments.medical_dataset_gen.schemas.query_geometry_schemas import (
    GeometryFilterStatsRow,
)
from experiments.medical_dataset_gen.schemas.retrieval_schemas import (
    BackgroundOutlierDiagnostics,
    FacetIdToGoldChunks,
    QueryIdToQrels,
    QueryRecord,
    TopKDiagnosticsByK,
)
from experiments.medical_dataset_gen.utils.io_utils import (
    json_dumps,
    read_parquet,
    write_json,
    write_parquet,
)


class StrictGeometryFailures(TypedDict):
    fail_missing_facet: bool
    fail_weak_primary_axis_dominance: bool
    fail_too_many_topk_facets: bool
    fail_weak_facet_separation: bool
    fail_weak_same_axis_cohort_separation: bool
    fail_weak_same_cohort_axis_separation: bool
    fail_too_few_near_miss_distractors: bool
    fail_missing_or_malformed_background_outlier: bool


def run_filter_queries(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    chunk_documents = read_parquet(paths, 'chunk_documents')
    chunk_memberships = read_parquet(paths, 'chunk_memberships')
    queries = read_parquet(paths, 'queries')
    qrels = read_parquet(paths, 'qrels')
    calibration_path = paths.table_path('query_plan_calibration')
    calibration_warning_by_query = (
        {
            str(query_id): bool(warning)
            for query_id, warning in zip(
                pl.read_parquet(calibration_path)['query_id'].to_list(),
                pl.read_parquet(calibration_path)['calibration_warning'].to_list(),
                strict=True,
            )
        }
        if calibration_path.exists()
        else {}
    )
    chunk_vectors, query_vectors, chunk_ids, query_ids = load_embedding_arrays(paths)
    maps = build_index_maps(chunk_documents, chunk_memberships, queries, chunk_ids, query_ids)

    facet_gold = build_query_to_facet_gold_map(qrels)
    qrels_by_query_chunk = get_qrels_by_query_chunk(qrels)
    primary_k = int(cfg.geometry_filter.topk_k)
    diagnostic_k_values = _diagnostic_k_values(cfg)
    rows: list[dict[str, object]] = []

    for query_row in tqdm(
        queries.iter_rows(named=True), total=len(queries), desc='Geometry', dynamic_ncols=True
    ):
        query = QueryRecord.model_validate(query_row)
        qid = query.query_id
        qidx = maps['query_id_to_idx'][qid]

        candidate_idx = get_candidate_pool_indices(
            query_id=qid,
            chunks_by_source_query=maps['chunks_by_source_query'],
        )
        topn_global, topn_sims = run_topn_cosine_retrieval(
            candidate_indices=candidate_idx,
            chunk_vectors=chunk_vectors,
            query_vector=query_vectors[qidx],
            n=cfg.retrieval.candidate_pool_n,
        )
        topn_chunk_ids = [chunk_ids[int(i)] for i in topn_global]
        topn_set = set(topn_chunk_ids)

        query_facets = facet_gold.get(qid, {})
        query_qrels = qrels_by_query_chunk.get(qid, {})
        if not query_facets:
            continue
        calibrated_facet_id = query.calibrated_primary_facet_id
        facet_meta = {
            str(facet['facet_id']): (str(facet['subgroup_id']), str(facet['axis']))
            for facet in json.loads(query.facets_json or '[]')
        }
        facets_present = {
            facet_id: bool(topn_set & set(gold_ids)) for facet_id, gold_ids in query_facets.items()
        }
        n_facets_present = sum(facets_present.values())

        topk_by_k = _topk_diagnostics_by_k(
            topn_chunk_ids=topn_chunk_ids,
            query_qrels=query_qrels,
            query_facets=query_facets,
            calibrated_facet_id=calibrated_facet_id,
            primary_axis=query.primary_axis,
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
        calibrated_topk_count = int(primary_topk['calibrated_primary_count'])
        primary_axis_topk_count = int(primary_topk['primary_axis_count'])
        all_facet_rank = _rank_where_all_facets_first_covered(
            topn_chunk_ids=topn_chunk_ids,
            query_qrels=query_qrels,
            query_facets=query_facets,
        )
        n_distractors = sum(
            1
            for chunk_id in topn_chunk_ids
            if not ((qrel := query_qrels.get(chunk_id)) and qrel.is_gold)
        )
        n_near_miss_distractors = sum(
            1
            for chunk_id in topn_chunk_ids
            if _is_query_near_miss_distractor(query_qrels, chunk_id)
        )
        separation = _facet_separation(
            qid=qid,
            query_facets=query_facets,
            facet_meta=facet_meta,
            chunk_id_to_idx=maps['chunk_id_to_idx'],
            chunk_vectors=chunk_vectors,
        )
        background_diagnostics = _background_outlier_diagnostics(
            topn_chunk_ids=topn_chunk_ids,
            topn_sims=topn_sims,
            query_qrels=query_qrels,
            chunk_id_to_idx=maps['chunk_id_to_idx'],
            chunk_vectors=chunk_vectors,
            expected_background_chunks=cfg.generation.chunk_pools.background_outlier_chunks_per_query(),
        )

        diagnostics = _topk_vs_facloc_diagnostics(
            topn_global=topn_global,
            topn_sims=topn_sims,
            chunk_vectors=chunk_vectors,
            k=primary_k,
        )

        failures = _strict_gate_failures(
            cfg.geometry_filter,
            n_facets_present=n_facets_present,
            n_facets=len(query_facets),
            primary_axis_topk_count=primary_axis_topk_count,
            n_topk_retrieved_facets=n_topk_retrieved_facets,
            in_minus_cross_similarity=separation['in_minus_cross_similarity'],
            same_axis_cohort_gap=separation['same_axis_cohort_gap'],
            same_cohort_axis_gap=separation['same_cohort_axis_gap'],
            n_near_miss_distractors=n_near_miss_distractors,
            background_outlier_complete=bool(background_diagnostics['background_outlier_complete']),
        )
        passes = not any(failures.values())

        geometry_row_data: dict[str, object] = (
            {
                'query_id': qid,
                'evidence_profile_id': query.evidence_profile_id,
                'pool_id': query.pool_id,
                'condition_id': query.condition_id,
                'cohort_contrast_id': query.cohort_contrast_id,
                'cohort_dimension_id': query.cohort_dimension_id,
                'template_id': query.template_id,
                'calibration_warning': calibration_warning_by_query.get(qid, False),
                'pool_scope': cfg.retrieval.pool_scope,
                'pool_size': len(topn_global),
                'topk_k': primary_k,
                'n_facets': len(query_facets),
                'n_facets_present': n_facets_present,
                'all_facets_present': n_facets_present == len(query_facets),
                'topk_dominant_count': topk_dominant,
                'calibrated_primary_facet_id': calibrated_facet_id,
                'calibrated_primary_topk_count': calibrated_topk_count,
                'calibrated_primary_topk_fraction': primary_topk['calibrated_primary_fraction'],
                'primary_axis': query.primary_axis,
                'secondary_axis': query.secondary_axis,
                'primary_axis_topk_count': primary_axis_topk_count,
                'primary_axis_topk_fraction': primary_topk['primary_axis_fraction'],
                'n_topk_retrieved_facets': n_topk_retrieved_facets,
                'max_topk_retrieved_facets': cfg.geometry_filter.max_topk_retrieved_facets,
                'rank_where_all_facets_first_covered': all_facet_rank,
                'all_facets_covered_before_primary_k': (
                    all_facet_rank is not None and all_facet_rank <= primary_k
                ),
                'n_distractors_in_pool': n_distractors,
                'n_near_miss_distractors_in_pool': n_near_miss_distractors,
                **separation,
                'passes_filter': passes,
                **failures,
                'facets_present_json': json_dumps(facets_present),
                'topk_retrieved_facets_json': json_dumps(topk_retrieved_facets),
            }
            | _flatten_topk_diagnostics(topk_by_k)
            | background_diagnostics
            | diagnostics
        )
        geometry_row = GeometryFilterStatsRow.model_validate(geometry_row_data)
        rows.append(geometry_row.model_dump(mode='python'))

    df = pl.DataFrame(rows)
    write_parquet(paths, 'geometry_stats', df)
    slice_stats = (
        df
        .group_by(
            'condition_id',
            'cohort_dimension_id',
            'cohort_contrast_id',
            'primary_axis',
            'secondary_axis',
            'template_id',
            'calibration_warning',
            'n_topk_retrieved_facets',
        )
        .agg(
            pl.len().alias('n_queries'),
            pl.col('passes_filter').sum().alias('n_eligible'),
            pl.col('passes_filter').mean().alias('eligible_rate'),
        )
        .sort('condition_id', 'cohort_contrast_id', 'primary_axis', 'secondary_axis')
    )
    write_parquet(paths, 'geometry_slice_stats', slice_stats)
    n_pass = int(df['passes_filter'].sum()) if len(df) else 0
    fail_columns = [column for column in df.columns if column.startswith('fail_')]
    combinations = df.select(fail_columns).to_dicts() if len(df) else []
    failure_combinations = Counter(
        '+'.join(column.removeprefix('fail_') for column, failed in row.items() if failed) or 'pass'
        for row in combinations
    )
    write_json(
        paths,
        'geometry_summary.json',
        {
            'dataset_schema_version': cfg.dataset_schema_version,
            'raw_query_count': len(df),
            'eligible_query_count': n_pass,
            'eligible_target_minimum': 3001,
            'target_met': n_pass > 3000,
            'failure_combinations': dict(sorted(failure_combinations.items())),
        },
    )
    print(f'[geometry] {n_pass:,}/{len(df):,} queries pass')
    return df


def _strict_gate_failures(
    cfg: GeometryFilterCfg,
    *,
    n_facets_present: int,
    n_facets: int,
    primary_axis_topk_count: int,
    n_topk_retrieved_facets: int,
    in_minus_cross_similarity: float,
    same_axis_cohort_gap: float,
    same_cohort_axis_gap: float,
    n_near_miss_distractors: int,
    background_outlier_complete: bool,
) -> StrictGeometryFailures:
    max_facets = cfg.max_topk_retrieved_facets
    return {
        'fail_missing_facet': n_facets_present != n_facets,
        'fail_weak_primary_axis_dominance': (primary_axis_topk_count < cfg.min_primary_axis_count),
        'fail_too_many_topk_facets': (
            max_facets is not None and n_topk_retrieved_facets > max_facets
        ),
        'fail_weak_facet_separation': (
            in_minus_cross_similarity < cfg.min_in_minus_cross_similarity
        ),
        'fail_weak_same_axis_cohort_separation': (
            same_axis_cohort_gap < cfg.min_same_axis_cohort_gap
        ),
        'fail_weak_same_cohort_axis_separation': (
            same_cohort_axis_gap < cfg.min_same_cohort_axis_gap
        ),
        'fail_too_few_near_miss_distractors': (
            n_near_miss_distractors < cfg.min_distractors_in_pool
        ),
        'fail_missing_or_malformed_background_outlier': not background_outlier_complete,
    }


def _diagnostic_k_values(cfg: ExperimentCfg) -> list[int]:
    return sorted({
        int(k)
        for k in [
            *cfg.retrieval.k_values,
            cfg.geometry_filter.topk_k,
        ]
    })


def _topk_diagnostics_by_k(
    *,
    topn_chunk_ids: list[str],
    query_qrels: QueryIdToQrels,
    query_facets: FacetIdToGoldChunks,
    calibrated_facet_id: str,
    primary_axis: str,
    k_values: list[int],
) -> TopKDiagnosticsByK:
    rows: TopKDiagnosticsByK = {}
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
        calibrated_count = counts.get(calibrated_facet_id, 0)
        primary_axis_count = sum(
            count
            for facet_id, count in counts.items()
            if (
                qrel := next(
                    (
                        row
                        for row in query_qrels.values()
                        if row.facet_id == facet_id and row.is_gold
                    ),
                    None,
                )
            )
            is not None
            and qrel.axis == primary_axis
        )
        n_facets = len(query_facets)
        rows[k] = {
            'dominant_count': most_common_count,
            'dominant_fraction': most_common_count / denominator,
            'primary_axis_count': primary_axis_count,
            'primary_axis_fraction': primary_axis_count / denominator,
            'calibrated_primary_count': calibrated_count,
            'calibrated_primary_fraction': calibrated_count / denominator,
            'n_retrieved_facets': len(retrieved_facets),
            'facet_coverage': len(retrieved_facets) / n_facets if n_facets else 0.0,
            'all_facets_covered': len(retrieved_facets) == n_facets,
            'retrieved_facets': retrieved_facets,
        }
    return rows


def _flatten_topk_diagnostics(topk_by_k: TopKDiagnosticsByK) -> dict[str, object]:
    flat: dict[str, object] = {}
    for k, row in topk_by_k.items():
        prefix = f'topk_{k}'
        flat[f'{prefix}_dominant_count'] = row['dominant_count']
        flat[f'{prefix}_dominant_fraction'] = row['dominant_fraction']
        flat[f'{prefix}_primary_axis_count'] = row['primary_axis_count']
        flat[f'{prefix}_primary_axis_fraction'] = row['primary_axis_fraction']
        flat[f'{prefix}_calibrated_primary_count'] = row['calibrated_primary_count']
        flat[f'{prefix}_calibrated_primary_fraction'] = row['calibrated_primary_fraction']
        flat[f'{prefix}_n_retrieved_facets'] = row['n_retrieved_facets']
        flat[f'{prefix}_facet_coverage'] = row['facet_coverage']
        flat[f'{prefix}_all_facets_covered'] = row['all_facets_covered']
        flat[f'{prefix}_retrieved_facets_json'] = json_dumps(row['retrieved_facets'])
    return flat


def _topk_facet_counts(
    topn_chunk_ids: list[str],
    query_qrels: QueryIdToQrels,
    k: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for chunk_id in topn_chunk_ids[:k]:
        row = query_qrels.get(chunk_id)
        if row is not None and row.is_gold and (facet_id := row.facet_id):
            counts[facet_id] += 1
    return counts


def _topk_dominant_count(
    topn_chunk_ids: list[str],
    query_qrels: QueryIdToQrels,
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
    query_qrels: QueryIdToQrels,
    k: int,
) -> list[str]:
    facets: set[str] = set()
    for chunk_id in topn_chunk_ids[:k]:
        row = query_qrels.get(chunk_id)
        if row is not None and row.is_gold and (facet_id := row.facet_id):
            facets.add(facet_id)
    return sorted(facets)


def _rank_where_all_facets_first_covered(
    *,
    topn_chunk_ids: list[str],
    query_qrels: QueryIdToQrels,
    query_facets: FacetIdToGoldChunks,
) -> int | None:
    expected = set(query_facets)
    seen: set[str] = set()
    for rank, chunk_id in enumerate(topn_chunk_ids, start=1):
        row = query_qrels.get(chunk_id)
        facet_id = row.facet_id if row is not None else None
        if row is not None and row.is_gold and facet_id in expected:
            seen.add(facet_id)
            if seen == expected:
                return rank
    return None


def _facet_separation(
    qid: str,
    query_facets: FacetIdToGoldChunks,
    facet_meta: dict[str, tuple[str, str]],
    chunk_id_to_idx: dict[str, int],
    chunk_vectors: NDArray[np.float32],
) -> dict[str, float]:
    _ = qid
    gold_ids = list[str]()
    labels = list[str]()

    for facet_id, ids in query_facets.items():
        for chunk_id in ids:
            if chunk_id in chunk_id_to_idx:
                gold_ids.append(chunk_id)
                labels.append(facet_id)
    if len(gold_ids) < 2:
        return {
            'mean_in_facet_similarity': 0.0,
            'mean_cross_facet_similarity': 0.0,
            'in_minus_cross_similarity': 0.0,
            'mean_same_axis_different_cohort_similarity': 0.0,
            'mean_same_cohort_different_axis_similarity': 0.0,
            'mean_different_axis_cohort_similarity': 0.0,
            'same_axis_cohort_gap': 0.0,
            'same_cohort_axis_gap': 0.0,
        }

    vectors = chunk_vectors[[chunk_id_to_idx[chunk_id] for chunk_id in gold_ids]]
    sim = vectors @ vectors.T
    labels_arr = np.array(labels)
    same = labels_arr[:, None] == labels_arr[None, :]
    not_self = ~np.eye(len(labels_arr), dtype=bool)
    in_vals = sim[same & not_self]
    cross_vals = sim[~same & not_self]
    in_sim = float(in_vals.mean()) if len(in_vals) else 0.0
    cross_sim = float(cross_vals.mean()) if len(cross_vals) else 0.0
    same_axis_diff_cohort: list[float] = []
    same_cohort_diff_axis: list[float] = []
    diff_both: list[float] = []
    for left in range(len(labels)):
        left_cohort, left_axis = facet_meta[labels[left]]
        for right in range(left + 1, len(labels)):
            if labels[left] == labels[right]:
                continue
            right_cohort, right_axis = facet_meta[labels[right]]
            value = float(sim[left, right])
            if left_axis == right_axis:
                same_axis_diff_cohort.append(value)
            elif left_cohort == right_cohort:
                same_cohort_diff_axis.append(value)
            else:
                diff_both.append(value)
    same_axis_mean = float(np.mean(same_axis_diff_cohort))
    same_cohort_mean = float(np.mean(same_cohort_diff_axis))
    diff_both_mean = float(np.mean(diff_both))
    return {
        'mean_in_facet_similarity': in_sim,
        'mean_cross_facet_similarity': cross_sim,
        'in_minus_cross_similarity': in_sim - cross_sim,
        'mean_same_axis_different_cohort_similarity': same_axis_mean,
        'mean_same_cohort_different_axis_similarity': same_cohort_mean,
        'mean_different_axis_cohort_similarity': diff_both_mean,
        'same_axis_cohort_gap': in_sim - same_axis_mean,
        'same_cohort_axis_gap': in_sim - same_cohort_mean,
    }


def _background_outlier_diagnostics(
    topn_chunk_ids: list[str],
    topn_sims: NDArray[np.float32],
    query_qrels: QueryIdToQrels,
    chunk_id_to_idx: dict[str, int],
    chunk_vectors: NDArray[np.float32],
    expected_background_chunks: int,
) -> BackgroundOutlierDiagnostics:
    background_positions = [
        idx
        for idx, chunk_id in enumerate(topn_chunk_ids)
        if _is_background_outlier(query_qrels, chunk_id)
    ]
    background_ids = [topn_chunk_ids[idx] for idx in background_positions]
    background_clusters = {str(query_qrels[chunk_id].cluster_id) for chunk_id in background_ids}
    gold_positions = [
        idx for idx, chunk_id in enumerate(topn_chunk_ids) if is_query_gold(query_qrels, chunk_id)
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


def _is_background_outlier(query_qrels: QueryIdToQrels, chunk_id: str) -> bool:
    row = query_qrels.get(chunk_id)
    return row is not None and row.cluster_role == 'background_outlier'


def _is_query_near_miss_distractor(query_qrels: QueryIdToQrels, chunk_id: str) -> bool:
    row = query_qrels.get(chunk_id)
    return (
        row is not None
        and not row.is_gold
        and row.cluster_role != 'background_outlier'
    )


def _mean_same_cluster_similarity(
    chunk_ids: list[str],
    query_qrels: QueryIdToQrels,
    chunk_id_to_idx: dict[str, int],
    chunk_vectors: NDArray[np.float32],
) -> float | None:
    ids_by_cluster: dict[str, list[str]] = defaultdict(list)
    for chunk_id in chunk_ids:
        if chunk_id in chunk_id_to_idx:
            ids_by_cluster[str(query_qrels[chunk_id].cluster_id)].append(chunk_id)

    values = list[float]()
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


if __name__ == '__main__':
    from experiments.medical_dataset_gen.global_config import (
        load_config_from_cli,
        paths_for,
        setup_logging,
    )

    cfg = load_config_from_cli()
    paths = paths_for(cfg)
    setup_logging(paths)
    run_filter_queries(cfg, paths)
