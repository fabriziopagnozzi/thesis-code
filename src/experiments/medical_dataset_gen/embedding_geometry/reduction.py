import numpy as np
from numpy.typing import NDArray

from experiments.medical_dataset_gen.global_configs import ExperimentCfg


def reduce_for_plot(
    cfg: ExperimentCfg,
    vectors: NDArray[np.float32],
) -> tuple[NDArray[np.float32], str]:
    if len(vectors) < 3:
        coords = np.zeros((len(vectors), 2), dtype=np.float32)
        coords[:, 0] = np.arange(len(vectors), dtype=np.float32)
        return coords, 'trivial'

    features = embedding_geometry_features(cfg, vectors)

    if cfg.embedding_geometry.reduction == 'umap':
        try:
            coords = umap_reduce(cfg, features, n_components=2)
            return coords, 'pca_umap' if features.shape[1] != vectors.shape[1] else 'umap'
        except Exception as exc:
            print(f'[embedding_geometry] UMAP failed; falling back to PCA: {exc}')

    return pca_2d(vectors, cfg.embedding_geometry.random_state), 'pca'


def embedding_geometry_features(
    cfg: ExperimentCfg,
    vectors: NDArray[np.float32],
) -> NDArray[np.float32]:
    if cfg.embedding_geometry.pca_dims is None:
        return vectors.astype(np.float32)

    return pca_preprocess(
        vectors,
        cfg.embedding_geometry.pca_dims,
        cfg.embedding_geometry.random_state,
    )


def cluster_features(cfg: ExperimentCfg, vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    features = embedding_geometry_features(cfg, vectors)
    if len(features) < 4:
        return features

    n_components = min(cfg.embedding_geometry.hdbscan_umap_dims, max(1, len(features) - 2))
    try:
        return umap_reduce(cfg, features, n_components=n_components)
    except Exception as exc:
        print(f'[embedding_geometry] clustering UMAP failed; using unreduced features: {exc}')
        return features


def umap_reduce(
    cfg: ExperimentCfg,
    features: NDArray[np.float32],
    n_components: int,
) -> NDArray[np.float32]:
    import umap

    n_neighbors = min(cfg.embedding_geometry.umap_neighbors, len(features) - 1)
    reducer = umap.UMAP(
        n_components=n_components,
        metric=cfg.embedding_geometry.umap_metric,
        n_neighbors=max(2, n_neighbors),
        min_dist=cfg.embedding_geometry.umap_min_dist,
        random_state=cfg.embedding_geometry.random_state,
    )
    return reducer.fit_transform(features).astype(np.float32)


def pca_preprocess(
    vectors: NDArray[np.float32],
    pca_dims: int,
    random_state: int,
) -> NDArray[np.float32]:
    if pca_dims <= 0:
        return vectors.astype(np.float32)
    if vectors.shape[1] <= pca_dims or len(vectors) <= 3:
        return vectors.astype(np.float32)
    from sklearn.decomposition import PCA

    n_components = min(pca_dims, vectors.shape[1], len(vectors) - 1)
    return (
        PCA(n_components=n_components, random_state=random_state)
        .fit_transform(vectors)
        .astype(np.float32)
    )


def pca_2d(vectors: NDArray[np.float32], random_state: int) -> NDArray[np.float32]:
    if len(vectors) < 2:
        return np.zeros((len(vectors), 2), dtype=np.float32)
    from sklearn.decomposition import PCA

    n_components = min(2, vectors.shape[1], len(vectors))
    coords = PCA(n_components=n_components, random_state=random_state).fit_transform(vectors)
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords), dtype=np.float32)])
    return coords.astype(np.float32)


def hdbscan_labels(cfg: ExperimentCfg, features: NDArray[np.float32]) -> NDArray[np.int32]:
    min_cluster_size = min(
        cfg.embedding_geometry.hdbscan_min_cluster_size, max(2, len(features) // 2)
    )
    if len(features) < max(4, min_cluster_size):
        return np.full(len(features), -1, dtype=np.int32)
    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=cfg.embedding_geometry.hdbscan_min_samples,
            metric='euclidean',
        )
        return np.asarray(clusterer.fit_predict(features), dtype=np.int32)
    except Exception as exc:
        print(f'[embedding_geometry] HDBSCAN failed; marking all points as noise: {exc}')
        return np.full(len(features), -1, dtype=np.int32)
