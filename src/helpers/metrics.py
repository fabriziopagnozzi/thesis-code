from collections import Counter

import numpy as np
from numpy.typing import NDArray


def fac_cov_score(
    selected_indices: NDArray[np.intp],
    sim_matrix: NDArray[np.float32],
) -> float:
    """fac(S) = (1/|D|) * sum_i max_{j in S} cos(e_i, e_j)"""
    if len(selected_indices) == 0:
        return 0.0
    coverage = sim_matrix[:, selected_indices].max(axis=1)
    return float(coverage.mean())


def avg_cos(
    selected_indices: NDArray[np.intp],
    sim_to_query: NDArray[np.float32],
) -> float:
    """AvgCos = (1/|S|) * sum_{j in S} cos(e_q, e_j)"""
    if len(selected_indices) == 0:
        return 0.0
    return float(sim_to_query[selected_indices].mean())


def recall_at_k(
    selected_indices: NDArray[np.intp],
    relevance: NDArray[np.bool_],
) -> float:
    """Recall@K = |relevant ∩ selected| / |relevant|.

    Args:
        selected_indices: indices of the K selected chunks.
        relevance: boolean array of length n, True if chunk is relevant.
    """
    total_relevant = int(relevance.sum())
    if total_relevant == 0:
        return 0.0
    retrieved_relevant = int(relevance[selected_indices].sum())
    return retrieved_relevant / total_relevant


def precision_at_k(
    selected_indices: NDArray[np.intp],
    relevance: NDArray[np.bool_],
) -> float:
    """Precision@K = |relevant ∩ selected| / K."""
    if len(selected_indices) == 0:
        return 0.0
    retrieved_relevant = int(relevance[selected_indices].sum())
    return retrieved_relevant / len(selected_indices)


def ndcg_at_k(
    selected_indices: NDArray[np.intp],
    relevance_grades: NDArray[np.float64],
) -> float:
    """nDCG@K = DCG@K / IDCG@K.
    Args:
        selected_indices: indices in rank order (first = rank 1).
        relevance_grades: graded relevance for each chunk (length n).
    """
    k = len(selected_indices)
    if k == 0:
        return 0.0

    gains = relevance_grades[selected_indices]
    discounts = np.log2(np.arange(2, k + 2))
    dcg = float(((2.0**gains - 1.0) / discounts).sum())

    # Ideal: sort all relevance grades descending, take top-k
    ideal_gains = np.sort(relevance_grades)[::-1][:k]
    idcg = float(((2.0**ideal_gains - 1.0) / discounts).sum())

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def token_f1(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 between predicted and gold answer (bag-of-words)
    DF-RAG style, post-LLM metric.
    P = |tokens(pred) ∩ tokens(gold)| / |tokens(pred)|
    R = |tokens(pred) ∩ tokens(gold)| / |tokens(gold)|
    F1 = 2PR / (P + R)
    """
    pred_tokens = Counter(prediction.lower().split())
    gold_tokens = Counter(ground_truth.lower().split())

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = sum((pred_tokens & gold_tokens).values())
    if common == 0:
        return 0.0

    precision = common / sum(pred_tokens.values())
    recall = common / sum(gold_tokens.values())
    return 2.0 * precision * recall / (precision + recall)


def jaccard(
    indices_a: NDArray[np.intp],
    indices_b: NDArray[np.intp],
) -> float:
    set_a = set(indices_a.tolist())
    set_b = set(indices_b.tolist())
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)
