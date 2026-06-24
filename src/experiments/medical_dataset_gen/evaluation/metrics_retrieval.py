from collections import Counter

import numpy as np

from experiments.medical_dataset_gen.evaluation.retrieval_utils import harmonic_mean
from experiments.medical_dataset_gen.schemas.evaluation_schemas import (
    ChunkDocumentRecord,
    QrelRecord,
)

ALPHA_NDCG_REDUNDANCY = 0.5


def compute_retrieval_metrics(
    selected_chunk_ids: list[str],
    chunk_by_id: dict[str, ChunkDocumentRecord],
    query_qrels: dict[str, QrelRecord],
    facet_to_gold: dict[str, list[str]],
    all_gold_ids: set[str],
    primary_axis: str,
    calibrated_primary_facet_id: str,
) -> dict[str, float | int]:
    relevance = _relevance_metrics(selected_chunk_ids, all_gold_ids)
    facet_coverage = _facet_coverage_metrics(selected_chunk_ids, facet_to_gold)
    diversified_ranking = _diversified_ranking_metrics(
        selected_chunk_ids, query_qrels, facet_to_gold, all_gold_ids
    )
    redundancy = _redundancy_metrics(
        selected_chunk_ids=selected_chunk_ids,
        query_qrels=query_qrels,
        all_gold_ids=all_gold_ids,
        primary_axis=primary_axis,
        calibrated_primary_facet_id=calibrated_primary_facet_id,
        n_selected_gold=int(relevance['n_selected_gold']),
        n_facet_hits=int(facet_coverage['n_unique_gold_facets']),
    )
    selected_rows = [chunk_by_id[cid] for cid in selected_chunk_ids]

    return {
        **relevance,
        **facet_coverage,
        **diversified_ranking,
        **redundancy,
        'n_unique_hadms': len({row.admission_id for row in selected_rows if row.admission_id}),
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
        'gold_f1': float(harmonic_mean(gold_precision, gold_recall)),
        'average_precision_at_k': _average_precision_at_k(
            selected_chunk_ids=selected_chunk_ids,
            all_gold_ids=all_gold_ids,
        ),
        'n_selected': n_selected,
        'n_selected_gold': n_selected_gold,
    }


def _redundancy_metrics(
    selected_chunk_ids: list[str],
    query_qrels: dict[str, QrelRecord],
    all_gold_ids: set[str],
    primary_axis: str,
    calibrated_primary_facet_id: str,
    n_selected_gold: int,
    n_facet_hits: int,
) -> dict[str, float | int]:
    n_selected = len(selected_chunk_ids)
    non_gold_ids = [chunk_id for chunk_id in selected_chunk_ids if chunk_id not in all_gold_ids]
    non_gold_count = len(non_gold_ids)
    background_outlier_count = sum(
        1
        for chunk_id in non_gold_ids
        if (qrel := query_qrels.get(chunk_id)) is not None
        and qrel.cluster_role == 'background_outlier'
    )
    same_condition_wrong_axis_count = sum(
        1
        for chunk_id in non_gold_ids
        if (qrel := query_qrels.get(chunk_id)) is not None
        and qrel.cluster_role == 'same_condition_wrong_axis'
    )
    near_miss_distractor_count = sum(
        1 for chunk_id in non_gold_ids if _is_query_near_miss_distractor(query_qrels, chunk_id)
    )
    calibrated_count = sum(
        1
        for chunk_id in selected_chunk_ids
        if (qrel := query_qrels.get(chunk_id)) is not None
        and qrel.facet_id == calibrated_primary_facet_id
    )
    primary_axis_count = sum(
        1
        for chunk_id in selected_chunk_ids
        if (qrel := query_qrels.get(chunk_id)) is not None
        and qrel.is_gold
        and qrel.axis == primary_axis
    )
    selected_facet_counts = Counter(
        qrel.facet_id
        for chunk_id in selected_chunk_ids
        if chunk_id in all_gold_ids
        and (qrel := query_qrels.get(chunk_id)) is not None
        and qrel.facet_id is not None
    )
    max_facet_concentration = (
        selected_facet_counts.most_common(1)[0][1] / n_selected
        if n_selected and selected_facet_counts
        else 0.0
    )
    redundant_gold_count = max(n_selected_gold - n_facet_hits, 0)
    calibrated_facet_rate = calibrated_count / n_selected if n_selected else 0.0
    primary_axis_rate = primary_axis_count / n_selected if n_selected else 0.0

    return {
        'distractor_rate': non_gold_count / n_selected if n_selected else 0.0,
        'near_miss_distractor_rate': (
            near_miss_distractor_count / n_selected if n_selected else 0.0
        ),
        'background_outlier_rate': background_outlier_count / n_selected if n_selected else 0.0,
        'same_condition_wrong_axis_rate': (
            same_condition_wrong_axis_count / n_selected if n_selected else 0.0
        ),
        'primary_axis_rate': float(primary_axis_rate),
        'calibrated_facet_rate': float(calibrated_facet_rate),
        'max_facet_concentration': float(max_facet_concentration),
        'redundant_gold_rate': redundant_gold_count / n_selected if n_selected else 0.0,
        'n_selected_non_gold': non_gold_count,
        'n_selected_near_miss_distractors': near_miss_distractor_count,
        'n_selected_background_outliers': background_outlier_count,
        'n_selected_same_condition_wrong_axis': same_condition_wrong_axis_count,
        'n_redundant_gold': redundant_gold_count,
    }


def _diversified_ranking_metrics(
    selected_chunk_ids: list[str],
    query_qrels: dict[str, QrelRecord],
    facet_to_gold: dict[str, list[str]],
    all_gold_ids: set[str],
) -> dict[str, float]:
    facet_mrr_at_k = DiversifiedRankingIndexMetrics.facet_mrr(
        selected_chunk_ids=selected_chunk_ids,
        query_qrels=query_qrels,
        facet_ids=list(facet_to_gold),
        all_gold_ids=all_gold_ids,
    )
    return {
        'alpha_ndcg': DiversifiedRankingIndexMetrics.alpha_ndcg(
            selected_chunk_ids=selected_chunk_ids,
            query_qrels=query_qrels,
            facet_to_gold=facet_to_gold,
            all_gold_ids=all_gold_ids,
            alpha=ALPHA_NDCG_REDUNDANCY,
        ),
        'facet_mrr_at_k': facet_mrr_at_k,
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
    facet_f1 = harmonic_mean(facet_hit_density, facet_coverage)

    return {
        'facet_coverage': float(facet_coverage),
        'weighted_facet_coverage': float(mean_facet_recall),
        'facet_hit_density': float(facet_hit_density),
        'facet_f1': float(facet_f1),
        'n_unique_gold_facets': n_facet_hits,
        'n_total_facets': n_facets,
    }


def _average_precision_at_k(
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


class DiversifiedRankingIndexMetrics:
    @classmethod
    def facet_mrr(
        cls,
        selected_chunk_ids: list[str],
        query_qrels: dict[str, QrelRecord],
        facet_ids: list[str],
        all_gold_ids: set[str],
    ) -> float:
        if not facet_ids:
            return 0.0
        first_rank: dict[str, int] = {}
        for rank, chunk_id in enumerate(selected_chunk_ids, start=1):
            if chunk_id not in all_gold_ids:
                continue
            qrel = query_qrels.get(chunk_id)
            facet_id = qrel.facet_id if qrel is not None else None
            if facet_id:
                first_rank.setdefault(str(facet_id), rank)
        reciprocal_ranks = [
            1 / first_rank[facet_id] if facet_id in first_rank else 0.0 for facet_id in facet_ids
        ]
        return float(np.mean(reciprocal_ranks))

    @classmethod
    def alpha_ndcg(
        cls,
        selected_chunk_ids: list[str],
        query_qrels: dict[str, QrelRecord],
        facet_to_gold: dict[str, list[str]],
        all_gold_ids: set[str],
        alpha: float,
    ) -> float:
        """alpha-nDCG with facet_id as the subtopic label.

        Repeated gold chunks from the same facet receive diminishing gain, which
        makes this a ranking-sensitive coverage metric for the synthetic benchmark.
        """
        selected_dcg = cls._alpha_dcg(
            selected_chunk_ids=selected_chunk_ids,
            query_qrels=query_qrels,
            all_gold_ids=all_gold_ids,
            alpha=alpha,
        )
        ideal_labels = cls._ideal_alpha_labels(
            facet_to_gold, k=len(selected_chunk_ids), alpha=alpha
        )
        ideal_dcg = cls._alpha_dcg_from_labels(ideal_labels, alpha=alpha)

        return float(selected_dcg / ideal_dcg) if ideal_dcg > 0 else 0.0

    @classmethod
    def _alpha_dcg_from_labels(cls, labels: list[str | None], alpha: float) -> float:
        counts: Counter[str] = Counter()
        total = 0.0
        for rank, facet_id in enumerate(labels, start=1):
            if facet_id is None:
                continue
            gain = (1 - alpha) ** counts[facet_id]
            counts[facet_id] += 1
            total += gain / np.log2(rank + 1)
        return float(total)

    @classmethod
    def _alpha_dcg(
        cls,
        selected_chunk_ids: list[str],
        query_qrels: dict[str, QrelRecord],
        all_gold_ids: set[str],
        alpha: float,
    ) -> float:
        labels: list[str | None] = []
        for chunk_id in selected_chunk_ids:
            qrel = query_qrels.get(chunk_id)
            labels.append(
                qrel.facet_id
                if chunk_id in all_gold_ids and qrel is not None and qrel.facet_id
                else None
            )
        return cls._alpha_dcg_from_labels(labels, alpha=alpha)

    @classmethod
    def _ideal_alpha_labels(
        cls,
        facet_to_gold: dict[str, list[str]],
        k: int,
        alpha: float,
    ) -> list[str | None]:
        remaining = {facet_id: len(gold_ids) for facet_id, gold_ids in facet_to_gold.items()}
        counts: Counter[str] = Counter()
        labels = list[str | None]()

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


def _is_query_near_miss_distractor(query_qrels: dict[str, QrelRecord], chunk_id: str) -> bool:
    row = query_qrels.get(chunk_id)
    return (
        row is not None
        and not row.is_gold
        and row.cluster_role not in {'background_outlier', 'same_condition_wrong_axis'}
    )
