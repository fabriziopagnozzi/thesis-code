"""
Facet-coverage metrics for multi-aspect RAG evaluation.

All functions operate on a *ranked* list of chunk IDs (retrieval order, rank 1 first)
and a facets dict mapping each facet label to the set of gold chunk IDs for that facet.
"""

import math

import numpy as np


def aspect_recall_at_k(
    ranked_chunk_ids: list[str],
    facets: dict[str, set[str]],
    k: int,
) -> float:
    """
    Fraction of facets covered by at least one chunk in the top-k.
    AR@k = |{f ∈ F : top_k(S) ∩ G_f ≠ ∅}| / |F|
    Equivalent to subtopic recall (S-recall) from the TREC Web Track diversity tasks.
    """
    if not facets:
        return 0.0
    selected = set(ranked_chunk_ids[:k])
    covered = sum(1 for gold in facets.values() if selected & gold)
    return covered / len(facets)


def coverage_auc(
    ranked_chunk_ids: list[str],
    facets: dict[str, set[str]],
    k_values: list[int],
) -> float:
    """Area under the AR@k curve (trapezoidal), normalized to [0, 1].

    Samples AR at each breakpoint in k_values and integrates with the trapezoid rule,
    then divides by (k_max - k_min) so the result is always in [0, 1].

    Args:
        ranked_chunk_ids: chunk IDs in retrieval order (rank 1 first).
        facets: mapping from facet label to gold chunk ID set.
        k_values: breakpoints to sample; must have at least 2 distinct values.
    """
    k_sorted = sorted(set(k_values))
    if len(k_sorted) < 2:
        raise ValueError('k_values must contain at least 2 distinct values')
    ar_vals = [aspect_recall_at_k(ranked_chunk_ids, facets, k) for k in k_sorted]
    auc = float(np.trapezoid(ar_vals, k_sorted)) / (k_sorted[-1] - k_sorted[0])
    return auc


def alpha_ndcg(
    ranked_chunk_ids: list[str],
    facets: dict[str, set[str]],
    k: int,
    alpha: float = 0.5,
) -> float:
    """alpha-nDCG@k (Clarke et al., 2008): rank-aware facet coverage metric.
    Args:
        ranked_chunk_ids: chunk IDs in retrieval order (rank 1 first).
        facets: mapping from facet label to gold chunk ID set.
        k: cutoff rank.
        alpha: redundancy penalty parameter (default 0.5, as in the original paper).
    """
    n_facets = len(facets)
    if n_facets == 0 or k == 0:
        return 0.0

    facet_sets = {label: set(cids) for label, cids in facets.items()}
    facet_labels = list(facet_sets.keys())

    counts: dict[str, int] = {label: 0 for label in facet_labels}
    dcg = 0.0
    for r, chunk_id in enumerate(ranked_chunk_ids[:k], start=1):
        gain = 0.0
        for label in facet_labels:
            if chunk_id in facet_sets[label]:
                gain += (1.0 - alpha) ** counts[label]
                counts[label] += 1
        dcg += (gain / n_facets) / math.log2(1.0 + r)

    idcg = 0.0
    for r in range(1, k + 1):
        cycle = (r - 1) // n_facets
        ideal_gain = (1.0 - alpha) ** cycle / n_facets
        idcg += ideal_gain / math.log2(1.0 + r)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg
