import numpy as np
from numpy.typing import NDArray


def cosine_stats(sim_matrix: NDArray[np.float32]) -> dict[str, float]:
    n = sim_matrix.shape[0]
    if n < 2:
        return {'mean_cos': 0.0, 'median_cos': 0.0, 'p90_cos': 0.0}
    iu = np.triu_indices(n, k=1)
    vals = sim_matrix[iu]
    return {
        'mean_cos': float(vals.mean()),
        'median_cos': float(np.median(vals)),
        'p90_cos': float(np.quantile(vals, 0.9)),
    }


def query_cosine_stats(sim_to_query: NDArray[np.float32]) -> dict[str, float]:
    if sim_to_query.size == 0:
        return {'mean_cos_to_query': 0.0, 'p10_cos_to_query': 0.0}
    return {
        'mean_cos_to_query': float(sim_to_query.mean()),
        'p10_cos_to_query': float(np.quantile(sim_to_query, 0.1)),
    }


def intrinsic_dim(vectors: NDArray[np.float32]) -> dict[str, float]:
    """Effective rank (entropy of normalized eigvals), participation ratio, top-1 EVR."""
    if vectors.shape[0] < 2:
        return {'effective_rank': 0.0, 'participation_ratio': 0.0, 'top1_evr': 0.0}
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    n_samples, n_feat = centered.shape
    if n_feat <= n_samples:
        cov = (centered.T @ centered) / max(1, n_samples - 1)
    else:
        cov = (centered @ centered.T) / max(1, n_samples - 1)

    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    total = eigvals.sum()
    if total <= 0:
        return {'effective_rank': 0.0, 'participation_ratio': 0.0, 'top1_evr': 0.0}

    p = eigvals / total
    p_clip = p[p > 1e-12]
    H = -float((p_clip * np.log(p_clip)).sum())

    return {
        'effective_rank': float(np.exp(H)),
        'participation_ratio': float(total**2 / float((eigvals**2).sum())),
        'top1_evr': float(eigvals.max() / total),
    }
