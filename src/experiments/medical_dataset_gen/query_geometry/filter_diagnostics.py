from __future__ import annotations

from collections import Counter, defaultdict
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from experiments.medical_dataset_gen.retrieval.retrieval_utils import (
    compute_retrieval_diagnostics,
    is_query_gold,
    select_indices,
)
from experiments.medical_dataset_gen.retrieval.schemas import (
    BackgroundOutlierDiagnostics,
    FacetIdToGoldChunks,
    QueryIdToQrels,
    TopKDiagnosticsByK,
)
from experiments.medical_dataset_gen.utils.global_schemas import (
    ExperimentCfg,
    GeometryFilterCfg,
)
from experiments.medical_dataset_gen.utils.io_utils import json_dumps


class StrictGeometryFailures(TypedDict):
    fail_missing_facet: bool
    fail_weak_primary_axis_dominance: bool
    fail_excess_stress_horizon_facet_coverage: bool


def strict_gate_failures(
    cfg: GeometryFilterCfg,
    *,
    n_facets_present: int,
    n_facets: int,
    primary_axis_fraction: float,
    n_topk_retrieved_facets: int,
) -> StrictGeometryFailures:
    return {
        'fail_missing_facet': n_facets_present != n_facets,
        'fail_weak_primary_axis_dominance': (primary_axis_fraction < cfg.min_primary_axis_fraction),
        'fail_excess_stress_horizon_facet_coverage': (
            n_topk_retrieved_facets / n_facets > cfg.max_retrieved_facet_fraction
        ),
    }


def competitive_pool_mass(cfg: ExperimentCfg) -> int:
    chunk_pools = cfg.generation.chunk_pools
    return chunk_pools.gold_chunks_per_query() + chunk_pools.near_miss_distractors_per_query()


def diagnostic_k_values(cfg: ExperimentCfg, *, stress_horizon_k: int) -> list[int]:
    return sorted(
        {
            int(k)
            for k in [
                *cfg.retrieval.k_values,
                stress_horizon_k,
            ]
        }
    )


def topk_diagnostics_by_k(
    *,
    topn_chunk_ids: list[str],
    query_qrels: QueryIdToQrels,
    query_facets: FacetIdToGoldChunks,
    dominant_facet_id: str,
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
        dominant_count = counts.get(dominant_facet_id, 0)
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
            'dominant_primary_count': dominant_count,
            'dominant_primary_fraction': dominant_count / denominator,
            'n_retrieved_facets': len(retrieved_facets),
            'facet_coverage': len(retrieved_facets) / n_facets if n_facets else 0.0,
            'all_facets_covered': len(retrieved_facets) == n_facets,
            'retrieved_facets': retrieved_facets,
        }
    return rows


def flatten_topk_diagnostics(topk_by_k: TopKDiagnosticsByK) -> dict[str, object]:
    flat: dict[str, object] = {}
    for k, row in topk_by_k.items():
        prefix = f'topk_{k}'
        flat[f'{prefix}_dominant_count'] = row['dominant_count']
        flat[f'{prefix}_dominant_fraction'] = row['dominant_fraction']
        flat[f'{prefix}_primary_axis_count'] = row['primary_axis_count']
        flat[f'{prefix}_primary_axis_fraction'] = row['primary_axis_fraction']
        flat[f'{prefix}_dominant_primary_count'] = row['dominant_primary_count']
        flat[f'{prefix}_dominant_primary_fraction'] = row['dominant_primary_fraction']
        flat[f'{prefix}_n_retrieved_facets'] = row['n_retrieved_facets']
        flat[f'{prefix}_facet_coverage'] = row['facet_coverage']
        flat[f'{prefix}_all_facets_covered'] = row['all_facets_covered']
        flat[f'{prefix}_retrieved_facets_json'] = json_dumps(row['retrieved_facets'])
    return flat


def topk_dominant_count(
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


def topk_retrieved_facets(
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


def rank_where_all_facets_first_covered(
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


def facet_separation(
    *,
    query_facets: FacetIdToGoldChunks,
    facet_meta: dict[str, tuple[str, str]],
    chunk_id_to_idx: dict[str, int],
    chunk_vectors: NDArray[np.float32],
) -> dict[str, float]:
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


def background_outlier_diagnostics(
    *,
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


def component_query_similarity_diagnostics(
    *,
    topn_chunk_ids: list[str],
    topn_sims: NDArray[np.float32],
    query_qrels: QueryIdToQrels,
) -> dict[str, object]:
    """Persist realized query similarity by pool component and near-miss type.

    Structural-change labels are design metadata, not a claim about embedding
    hardness.  These diagnostics make the empirical distance visible to the
    report without involving a retrieval outcome.
    """
    by_type: dict[str, list[float]] = {}
    component: dict[str, list[float]] = {'gold': [], 'near_miss': [], 'background': []}
    for chunk_id, similarity in zip(topn_chunk_ids, topn_sims, strict=True):
        qrel = query_qrels.get(chunk_id)
        if qrel is None:
            continue
        value = float(similarity)
        if qrel.is_gold:
            component['gold'].append(value)
        elif qrel.cluster_role == 'background_outlier':
            component['background'].append(value)
        else:
            component['near_miss'].append(value)
            if qrel.distractor_type is not None:
                by_type.setdefault(str(qrel.distractor_type), []).append(value)
    means = {key: float(np.mean(values)) for key, values in component.items() if values}
    type_means = {key: float(np.mean(values)) for key, values in sorted(by_type.items()) if values}
    gold_mean = means.get('gold')
    return {
        'query_to_component_similarity_json': json_dumps(means),
        'query_to_near_miss_type_similarity_json': json_dumps(type_means),
        'query_to_near_miss_mean': means.get('near_miss'),
        'gold_minus_near_miss_similarity_margin': (
            float(gold_mean - means['near_miss'])
            if gold_mean is not None and 'near_miss' in means
            else None
        ),
    }


def is_query_near_miss_distractor(query_qrels: QueryIdToQrels, chunk_id: str) -> bool:
    row = query_qrels.get(chunk_id)
    return row is not None and not row.is_gold and row.cluster_role != 'background_outlier'


def topk_vs_facloc_diagnostics(
    *,
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


def _is_background_outlier(query_qrels: QueryIdToQrels, chunk_id: str) -> bool:
    row = query_qrels.get(chunk_id)
    return row is not None and row.cluster_role == 'background_outlier'


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
