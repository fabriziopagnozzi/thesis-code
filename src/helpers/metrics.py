from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Document / Fact Recovery (HotpotQA)
def doc_rec(
    selected_indices: NDArray[np.intp],
    chunks: Sequence[Any],
    gold_doc_titles: set[str],
) -> float:
    """Fraction of gold documents that have at least one selected chunk.
    DocRec = |selected_gold_docs| / |gold_docs|
    """
    if not gold_doc_titles:
        return 0.0
    selected_docs = {chunks[i].doc_title for i in selected_indices}
    recovered = selected_docs & gold_doc_titles
    return len(recovered) / len(gold_doc_titles)


def fact_rec(
    selected_indices: NDArray[np.intp],
    chunks: Sequence[Any],
    gold_facts: list[tuple[str, int]],
) -> float:
    """Fraction of gold supporting facts recovered.

    In sentence mode: exact (title, sentence_idx) match.
    In document mode: any chunk from the gold doc counts
    (is_gold_fact is set at load time).
    """
    if not gold_facts:
        return 0.0

    selected_chunks = [chunks[i] for i in selected_indices]

    # Sentence-level matching
    if selected_chunks and selected_chunks[0].sentence_idx is not None:
        selected_facts = {(c.doc_title, c.sentence_idx) for c in selected_chunks}
        gold_set = set(gold_facts)
        recovered = selected_facts & gold_set
        return len(recovered) / len(gold_set)

    # Document-level: count gold docs recovered
    selected_gold = sum(1 for c in selected_chunks if c.is_gold_fact)
    n_gold_docs_with_facts = len({t for t, _ in gold_facts})
    return min(1.0, selected_gold / n_gold_docs_with_facts)


# ---------------------------------------------------------------------------
def hit_rate(
    selected_indices: NDArray[np.intp],
    chunks: Sequence[Any],
    answer: str | list[str],
) -> float:
    """1 if any gold alias appears in the concatenated selected context, else 0."""
    context = ' '.join(chunks[i].text for i in selected_indices).lower()
    aliases = answer if isinstance(answer, list) else [answer]
    return 1.0 if any(a.lower() in context for a in aliases) else 0.0


# ---------------------------------------------------------------------------
# Embedding-based Metrics
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


# ---------------------------------------------------------------------------
# Recall@K / Precision@K (requires binary relevance judgments per chunk)
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


# ---------------------------------------------------------------------------
# nDCG@K (requires graded relevance judgments)
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


# ---------------------------------------------------------------------------
# Gold-Alias Metrics (TriviaQA)
def gold_concat(
    selected_indices: NDArray[np.intp],
    chunks: Sequence[Any],
    aliases: list[str],
) -> float:
    """
    Number of distinct gold aliases present in the concatenated context.
    GoldConcat(q) = |{a ∈ A(q) : a ⊆ concat(S_q)}|
    """
    if not aliases or len(selected_indices) == 0:
        return 0.0
    context = ' '.join(chunks[i].text for i in selected_indices).lower()
    return float(sum(1 for a in aliases if a.lower() in context))


def gold_per_chunk(
    selected_indices: NDArray[np.intp],
    chunks: Sequence[Any],
    aliases: list[str],
) -> float:
    """
    Average per-chunk count of distinct gold aliases found.
    Gold/Any(q) = (1/|S_q|) * sum_{j in S_q} |{a ∈ A(q) : a ⊆ chunk_j}|
    """
    if not aliases or len(selected_indices) == 0:
        return 0.0
    aliases_lower = [a.lower() for a in aliases]
    total = 0.0
    for i in selected_indices:
        text = chunks[i].text.lower()
        total += sum(1 for a in aliases_lower if a in text)
    return total / len(selected_indices)


def jaccard(
    indices_a: NDArray[np.intp],
    indices_b: NDArray[np.intp],
) -> float:
    set_a = set(indices_a.tolist())
    set_b = set(indices_b.tolist())
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_metrics(
    selected_indices: NDArray[np.intp],
    chunks: Sequence[Any],
    answer: str | list[str],
    gold_doc_titles: set[str],
    gold_facts: list[tuple[str, int]],
    sim_to_query: NDArray[np.float32],
    sim_matrix: NDArray[np.float32],
    answer_aliases: list[str] | None = None,
    relevance: NDArray[np.bool_] | None = None,
    relevance_grades: NDArray[np.float64] | None = None,
) -> dict[str, float]:
    results: dict[str, float] = {
        'doc_rec': doc_rec(selected_indices, chunks, gold_doc_titles),
        'fact_rec': fact_rec(selected_indices, chunks, gold_facts),
        'hit_rate': hit_rate(selected_indices, chunks, answer),
        'fac_cov_score': fac_cov_score(selected_indices, sim_matrix),
        'avg_cos': avg_cos(selected_indices, sim_to_query),
    }

    if answer_aliases is not None:
        results['gold_concat'] = gold_concat(
            selected_indices,
            chunks,
            answer_aliases,
        )
        results['gold_per_chunk'] = gold_per_chunk(
            selected_indices,
            chunks,
            answer_aliases,
        )

    if relevance is not None:
        results['recall_at_k'] = recall_at_k(selected_indices, relevance)
        results['precision_at_k'] = precision_at_k(selected_indices, relevance)

    if relevance_grades is not None:
        results['ndcg_at_k'] = ndcg_at_k(selected_indices, relevance_grades)

    return results
