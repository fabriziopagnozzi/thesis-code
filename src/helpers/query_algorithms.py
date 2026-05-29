"""Chunk selection algorithms.

All functions return an array of selected indices into the candidate pool.

Conventions:
    - sim_to_query: shape (n,) - cosine similarity of each chunk to the query
    - sim_matrix:   shape (n, n) - pairwise cosine similarity between chunks
    - embeddings:   shape (n, d) - chunk embeddings (for FPS / gMMR)
    - k:            number of chunks to select
    - lam:          relevance-diversity trade-off (1.0 = pure relevance)
"""

import heapq
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

type ScoringFunction = Literal['top_k', 'mmr', 'gmmr', 'fps', 'fac_loc']


def top_k(
    sim_to_query: NDArray[np.float32],
    k: int,
    **_kwargs: object,
) -> NDArray[np.intp]:
    k = min(k, len(sim_to_query))
    return np.argsort(sim_to_query)[::-1][:k]


def mmr(
    sim_to_query: NDArray[np.float32],
    k: int,
    sim_matrix: NDArray[np.float32],
    lam: float = 0.5,
    window: int | None = None,
    **_kwargs: object,
) -> NDArray[np.intp]:
    """
    Maximal Marginal Relevance.
    score(i) = lam * cos(q, i) - (1-lam) * max_{j in W} cos(i, j)
    W = last `window` selected items (None = all selected).
    """
    n = len(sim_to_query)
    k = min(k, n)
    selected: list[int] = []
    candidate_mask = np.ones(n, dtype=bool)
    # Running max similarity of each item to the selected set (window=None only)
    max_sim_to_sel = np.zeros(n, dtype=np.float32)

    for step in range(k):
        if step == 0:
            scores = lam * sim_to_query
        elif window is None:
            scores = lam * sim_to_query - (1 - lam) * max_sim_to_sel
        else:
            w = selected[-window:]
            scores = lam * sim_to_query - (1 - lam) * sim_matrix[:, w].max(axis=1)

        scores = scores.copy()
        scores[~candidate_mask] = -np.inf
        best = int(np.argmax(scores))
        selected.append(best)
        candidate_mask[best] = False

        if window is None:
            max_sim_to_sel = np.maximum(max_sim_to_sel, sim_matrix[:, best])

    return np.array(selected, dtype=np.intp)


def gmmr(
    sim_to_query: NDArray[np.float32],
    k: int,
    embeddings: NDArray[np.float32],
    lam: float = 0.5,
    **_kwargs: object,
) -> NDArray[np.intp]:
    """
    Geometric MMR (from DF-RAG).
    score(c) = lam * cos(q, c) + (1-lam) * ||c - centroid(S)||
    Distance is Euclidean on (possibly normalized) embeddings:
    ||c - centroid|| = sqrt(2 - 2*cos(c, centroid)) for unit vectors.
    """
    n = len(sim_to_query)
    k = min(k, n)
    selected: list[int] = []
    mask = np.ones(n, dtype=bool)

    for step in range(k):
        if step == 0:
            # First selection: pick most relevant
            scores = sim_to_query.copy()
        else:
            centroid = embeddings[selected].mean(axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                cos_to_centroid = embeddings @ centroid / centroid_norm
            else:
                cos_to_centroid = np.zeros(n)
            # Euclidean distance on unit sphere
            diversity = np.sqrt(np.maximum(0, 2 - 2 * cos_to_centroid))
            scores = lam * sim_to_query + (1 - lam) * diversity

        scores[~mask] = -np.inf
        best = int(np.argmax(scores))
        selected.append(best)
        mask[best] = False

    return np.array(selected, dtype=np.intp)


def fps(
    embeddings: NDArray[np.float32],
    k: int,
    sim_to_query: NDArray[np.float32] | None = None,
    **_kwargs: object,
) -> NDArray[np.intp]:
    """
    Farthest Point Sampling (pure dispersion, no relevance).
    """
    n = len(embeddings)
    k = min(k, n)

    seed = int(np.argmax(sim_to_query)) if sim_to_query is not None else 0

    selected = [seed]
    min_dists = np.full(n, np.inf)

    for _ in range(k - 1):
        last = selected[-1]
        diff = embeddings - embeddings[last]
        dists = np.sum(diff * diff, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -1.0  # exclude already selected
        best = int(np.argmax(min_dists))
        selected.append(best)

    return np.array(selected, dtype=np.intp)


# def fac_loc(
#     sim_to_query: NDArray[np.float32],
#     k: int,
#     sim_matrix: NDArray[np.float32],
#     lam: float = 0.5,
#     **_kwargs: object,
# ) -> NDArray[np.intp]:
#     n = len(sim_to_query)
#     k = min(k, n)

#     selected: list[int] = []
#     mask = np.ones(n, dtype=bool)
#     m = np.zeros(n, dtype=np.float64)

#     for _ in range(k):
#         # Marginal coverage gain for each candidate j:
#         # new_cov[j] = (1/n) * sum_i max(0, sim_matrix[i,j] - m[i])
#         gains = np.maximum(0, sim_matrix - m[:, None])  # (n, n)
#         marginal_cov = gains.sum(axis=0) / n  # (n,)

#         scores = lam * sim_to_query + (1 - lam) * marginal_cov
#         scores[~mask] = -np.inf
#         best = int(np.argmax(scores))

#         selected.append(best)
#         mask[best] = False
#         m = np.maximum(m, sim_matrix[:, best])

#     return np.array(selected, dtype=np.intp)


# TODO: try and use apricot
def fac_loc(
    query_sim_scores: NDArray[np.float32],
    k: int,
    dataset_sim_matrix: NDArray[np.float32],
    lam: float = 0.5,
    **_kwargs: object,
) -> NDArray[np.intp]:
    n = len(query_sim_scores)
    if n == 0:
        return np.array([], dtype=np.intp)
    k = min(k, n)

    selected: list[int] = []
    m = np.zeros(n, dtype=np.float64)

    initial_coverage = np.maximum(0, dataset_sim_matrix).sum(axis=0) / n
    initial_gains = lam * query_sim_scores + (1 - lam) * initial_coverage

    max_heap = [(-initial_gains[i], i, 0) for i in range(n)]
    heapq.heapify(max_heap)

    # Lazy Greedy Selection
    for step in range(k):
        while True:
            # item with the highest historical gain
            _, node_idx, last_update = heapq.heappop(max_heap)

            if last_update == step:
                selected.append(node_idx)
                m = np.maximum(m, dataset_sim_matrix[:, node_idx])
                break

            marginal_cov = np.sum(np.maximum(0, dataset_sim_matrix[:, node_idx] - m)) / n
            new_gain = lam * query_sim_scores[node_idx] + (1 - lam) * marginal_cov
            heapq.heappush(max_heap, (-new_gain, node_idx, step))

    return np.array(selected, dtype=np.intp)


STRATEGIES: dict[ScoringFunction, Callable[..., NDArray[np.intp]]] = {
    'top_k': top_k,
    'mmr': mmr,
    'gmmr': gmmr,
    'fps': fps,
    'fac_loc': fac_loc,
}


def select(
    strategy: ScoringFunction,
    sim_to_query: NDArray[np.float32],
    k: int,
    sim_matrix: NDArray[np.float32] | None = None,
    embeddings: NDArray[np.float32] | None = None,
    query_embedding: NDArray[np.float32] | None = None,
    lam: float = 0.5,
    window: int | None = None,
    theta: float | None = None,
) -> NDArray[np.intp]:
    return STRATEGIES[strategy](
        sim_to_query=sim_to_query,
        k=k,
        sim_matrix=sim_matrix,
        embeddings=embeddings,
        query_embedding=query_embedding,
        lam=lam,
        theta=theta,
        window=window,
    )
