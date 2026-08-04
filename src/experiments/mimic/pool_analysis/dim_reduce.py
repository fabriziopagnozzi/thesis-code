import numpy as np
from numpy.typing import NDArray


def umap_2d(
    vectors: NDArray[np.float32],
    n_neighbors: int,
    min_dist: float,
    metric: str,
) -> NDArray[np.float32]:
    """2D UMAP directly from the original high-dim embeddings."""
    import umap

    n = vectors.shape[0]
    n_eff = max(2, min(n_neighbors, n - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_eff,
        min_dist=min_dist,
        metric=metric,
        n_jobs=-1,
    )
    return np.asarray(reducer.fit_transform(vectors), dtype=np.float32)


def umap_reduce(
    vectors: NDArray[np.float32],
    n_components: int,
    n_neighbors: int,
    metric: str,
) -> NDArray[np.float32]:
    """UMAP to n_components dims (min_dist=0 for tight cluster separation). Used for HDBSCAN."""
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
