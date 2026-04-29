import numpy as np
from numpy.typing import NDArray


def lof_scores(
    vectors: NDArray[np.float32],
    n_neighbors: int,
    contamination: float | str,
) -> NDArray[np.float64]:
    """LOF in cosine space. Returns scores with higher = more outlier (>=1 nominal)."""
    from sklearn.neighbors import LocalOutlierFactor

    n = vectors.shape[0]
    k_eff = max(2, min(n_neighbors, n - 1))
    lof = LocalOutlierFactor(
        n_neighbors=k_eff,
        contamination=contamination,
        metric='cosine',
        novelty=False,
    )
    lof.fit_predict(vectors)
    return -np.asarray(lof.negative_outlier_factor_, dtype=np.float64)


def lof_summary(scores: NDArray[np.float64]) -> dict[str, float]:
    if scores.size == 0:
        return {'lof_p90': 0.0, 'lof_p99': 0.0, 'frac_lof_gt_1_5': 0.0}
    return {
        'lof_p90': float(np.quantile(scores, 0.9)),
        'lof_p99': float(np.quantile(scores, 0.99)),
        'frac_lof_gt_1_5': float((scores > 1.5).mean()),
    }
