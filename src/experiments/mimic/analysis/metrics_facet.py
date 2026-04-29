import numpy as np
from numpy.typing import NDArray


def facet_pairwise_cos(
    sim_matrix: NDArray[np.float32],
    facet_onehot: NDArray[np.bool_],
) -> dict[str, float]:
    """Mean pairwise cos within (intra) and across (cross) each modifier facet, averaged over modifiers."""
    n_modifiers = facet_onehot.shape[1]
    n = sim_matrix.shape[0]
    if n < 2 or n_modifiers == 0:
        return {
            'intra_facet_mean_cos': float('nan'),
            'cross_facet_mean_cos': float('nan'),
            'intra_minus_cross': float('nan'),
        }
    iu = np.triu_indices(n, k=1)
    pair_vals = sim_matrix[iu]
    intra_vals: list[float] = []
    cross_vals: list[float] = []
    for j in range(n_modifiers):
        mask = facet_onehot[:, j]
        same = mask[iu[0]] & mask[iu[1]]
        diff = mask[iu[0]] ^ mask[iu[1]]
        if same.any():
            intra_vals.append(float(pair_vals[same].mean()))
        if diff.any():
            cross_vals.append(float(pair_vals[diff].mean()))
    intra = float(np.mean(intra_vals)) if intra_vals else float('nan')
    cross = float(np.mean(cross_vals)) if cross_vals else float('nan')
    diff = (intra - cross) if intra_vals and cross_vals else float('nan')
    return {
        'intra_facet_mean_cos': intra,
        'cross_facet_mean_cos': cross,
        'intra_minus_cross': diff,
    }


def facet_silhouette(
    vectors: NDArray[np.float32],
    facet_combined: NDArray[np.object_],
) -> float:
    from sklearn.metrics import silhouette_score

    classes, counts = np.unique(facet_combined, return_counts=True)
    if len(classes) < 2 or counts.min() < 2:
        return float('nan')
    return float(silhouette_score(vectors, facet_combined, metric='cosine'))


def _supervised_cv_acc(
    estimator,
    vectors: NDArray[np.float32],
    facet_combined: NDArray[np.object_],
    n_splits: int,
) -> float:
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder

    classes, counts = np.unique(facet_combined, return_counts=True)
    if len(classes) < 2 or counts.min() < 3:
        return float('nan')
    n_splits_eff = int(min(n_splits, counts.min()))
    if n_splits_eff < 2:
        return float('nan')
    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=0)
    y = LabelEncoder().fit_transform(facet_combined)
    try:
        scores = cross_val_score(estimator, vectors, y, cv=skf, scoring='accuracy')
    except Exception:
        return float('nan')
    return float(scores.mean())


def lda_cv_acc(
    vectors: NDArray[np.float32],
    facet_combined: NDArray[np.object_],
    n_splits: int = 5,
) -> float:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    return _supervised_cv_acc(
        LinearDiscriminantAnalysis(solver='lsqr'), vectors, facet_combined, n_splits
    )


def logreg_cv_acc(
    vectors: NDArray[np.float32],
    facet_combined: NDArray[np.object_],
    n_splits: int = 5,
) -> float:
    from sklearn.linear_model import LogisticRegression

    return _supervised_cv_acc(
        LogisticRegression(max_iter=200),
        vectors,
        facet_combined,
        n_splits,
    )
