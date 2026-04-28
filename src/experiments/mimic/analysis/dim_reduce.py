import numpy as np
from numpy.typing import NDArray


def umap_reduce(
    vectors: NDArray[np.float32],
    n_components: int,
    n_neighbors: int,
    metric: str,
) -> NDArray[np.float32]:
    """Single UMAP run to n_components dims.
    Use first 2 columns for plotting, all columns for HDBSCAN.
    min_dist=0.0 keeps clusters compact
    """
    import umap

    n = vectors.shape[0]
    n_eff = max(2, min(n_neighbors, n - 1))
    n_comp = max(2, min(n_components, n - 1))
    reducer = umap.UMAP(
        n_components=n_comp,
        n_neighbors=n_eff,
        min_dist=0.0,
        metric=metric,
        n_jobs=-1,
    )
    return np.asarray(reducer.fit_transform(vectors), dtype=np.float32)


def supervised_umap_2d(
    vectors: NDArray[np.float32],
    y: NDArray[np.object_],
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> NDArray[np.float32]:
    """
    Categorical-target UMAP, projects so that points sharing a y-label are pulled together.
    Use to compare against the unsupervised projection
    """
    import umap
    from sklearn.preprocessing import LabelEncoder

    n = vectors.shape[0]
    n_eff = max(2, min(n_neighbors, n - 1))
    y_enc = LabelEncoder().fit_transform(y)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_eff,
        min_dist=min_dist,
        metric=metric,
        target_metric='categorical',
        random_state=random_state,
    )
    return np.asarray(reducer.fit_transform(vectors, y=y_enc), dtype=np.float32)
