"""Chunk selection algorithms.

All functions return an array of selected indices into the candidate pool.

Conventions:
    - sim_to_query: shape (n,) — cosine similarity of each chunk to the query
    - sim_matrix:   shape (n, n) — pairwise cosine similarity between chunks
    - embeddings:   shape (n, d) — chunk embeddings (for FPS / gMMR)
    - k:            number of chunks to select
    - lam:          relevance-diversity trade-off (1.0 = pure relevance)
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from experiments.config import ScoringFunction


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
    remaining = set(range(n))

    for _ in range(k):
        best_score = -np.inf
        best_idx = -1

        for i in remaining:
            relevance = lam * sim_to_query[i]
            if selected:
                w = selected[-window:] if window else selected
                redundancy = (1 - lam) * max(sim_matrix[i, j] for j in w)
            else:
                redundancy = 0.0
            score = relevance - redundancy

            if score > best_score:
                best_score = score
                best_idx = i

        selected.append(best_idx)
        remaining.discard(best_idx)

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

    # Seed with most relevant point
    seed = int(np.argmax(sim_to_query)) if sim_to_query is not None else 0

    selected = [seed]
    # Track min squared-Euclidean distance to selected set
    min_dists = np.full(n, np.inf)

    for _ in range(k - 1):
        last = selected[-1]
        diff = embeddings - embeddings[last]
        dists = np.sum(diff * diff, axis=1)  # squared Euclidean
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -1.0  # exclude already selected
        best = int(np.argmax(min_dists))
        selected.append(best)

    return np.array(selected, dtype=np.intp)


def facility_location(
    sim_to_query: NDArray[np.float32],
    k: int,
    sim_matrix: NDArray[np.float32],
    lam: float = 0.5,
    **_kwargs: object,
) -> NDArray[np.intp]:
    """
    Greedy facility-location (coverage) selection.
    """
    n = len(sim_to_query)
    k = min(k, n)
    selected: list[int] = []
    mask = np.ones(n, dtype=bool)
    # m[i] = max similarity of item i to any selected item
    m = np.zeros(n, dtype=np.float64)

    for _ in range(k):
        # Marginal coverage gain for each candidate j:
        # new_cov[j] = (1/n) * sum_i max(0, sim_matrix[i,j] - m[i])
        gains = np.maximum(0, sim_matrix - m[:, None])  # (n, n)
        marginal_cov = gains.sum(axis=0) / n  # (n,)

        scores = lam * sim_to_query + (1 - lam) * marginal_cov
        scores[~mask] = -np.inf
        best = int(np.argmax(scores))

        selected.append(best)
        mask[best] = False
        m = np.maximum(m, sim_matrix[:, best])

    return np.array(selected, dtype=np.intp)


def sector_coverage(
    sim_to_query: NDArray[np.float32],
    k: int,
    embeddings: NDArray[np.float32],
    query_embedding: NDArray[np.float32],
    lam: float = 0.5,
    theta: float | None = None,
    **_kwargs: object,
) -> NDArray[np.intp]:
    """
    Sector-based angular coverage (θ-dominance inspired).

    Instead of soft cosine similarity (which concentrates in high-d),
    uses hard angular thresholds from Guo, Jagadish et al. (2018):
    candidate i is "covered" by selected j iff ∠iqj < θ.

    Greedy maximizes fraction of covered candidates, balanced with
    relevance. Set cover structure → monotone submodular → (1-1/e).

    Args:
        theta: sector half-width in degrees. None = adaptive (median
        pairwise angle from the candidate pool).
    """
    n = len(sim_to_query)
    k = min(k, n)

    # Direction vectors from query, normalized
    directions = embeddings - query_embedding  # (n, d)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    dir_normed = directions / norms  # (n, d)

    # Cosine of angle at query between every pair
    ang_cos = dir_normed @ dir_normed.T  # (n, n)

    # Angular threshold
    if theta is not None:
        cos_thresh = np.cos(np.radians(theta))
    else:
        # Adaptive: median pairwise angular cosine
        upper_tri = ang_cos[np.triu_indices(n, k=1)]
        cos_thresh = float(np.median(upper_tri))

    # Binary coverage: covers[i, j] = True iff j covers i
    covers = ang_cos >= cos_thresh  # (n, n)

    # Greedy set-cover balanced with relevance
    selected: list[int] = []
    mask = np.ones(n, dtype=bool)
    covered = np.zeros(n, dtype=bool)

    for _ in range(k):
        # How many uncovered candidates each j would newly cover
        uncovered = ~covered
        marginal_cov = (covers[uncovered, :]).sum(axis=0).astype(np.float64) / n

        scores = lam * sim_to_query + (1 - lam) * marginal_cov
        scores[~mask] = -np.inf
        best = int(np.argmax(scores))

        selected.append(best)
        mask[best] = False
        covered |= covers[:, best]

    return np.array(selected, dtype=np.intp)


STRATEGIES: dict[ScoringFunction, Callable[..., NDArray[np.intp]]] = {
    'top_k': top_k,
    'mmr': mmr,
    'gmmr': gmmr,
    'fps': fps,
    'facility_location': facility_location,
    'sector_coverage': sector_coverage,
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
