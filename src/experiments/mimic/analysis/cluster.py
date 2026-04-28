import numpy as np
from numpy.typing import NDArray


def hdbscan_cluster(
    vectors: NDArray[np.float32],
    min_cluster_size: int,
    min_samples: int | None = None,
) -> NDArray[np.int_]:
    """
    Vectors are assumed unit-normalized when high-dim cosine semantics matter;
    after a UMAP reduction we use plain euclidean.
    """
    import hdbscan

    n = vectors.shape[0]
    mc = max(2, min(min_cluster_size, max(2, n // 5)))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mc,
        min_samples=min_samples,
        metric='euclidean',
        core_dist_n_jobs=-1,
    )
    return np.asarray(clusterer.fit_predict(vectors), dtype=int)


def kmeans_elbow(
    vectors: NDArray[np.float32],
    k_min: int,
    k_max: int,
    random_state: int,
) -> tuple[int, NDArray[np.int_]]:
    """Pick k by silhouette across [k_min, k_max]. Returns (best_k, labels)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = vectors.shape[0]
    k_max_eff = max(k_min, min(k_max, n - 1))
    best_k = k_min
    best_score = -1.0
    best_labels = np.zeros(n, dtype=int)
    for k in range(k_min, k_max_eff + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(vectors)
        labels = km.labels_
        if len(np.unique(labels)) < 2:
            continue
        try:
            score = silhouette_score(vectors, labels)
        except ValueError:
            continue
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
    return best_k, np.asarray(best_labels, dtype=int)


def cluster_summary(labels: NDArray[np.int_], prefix: str = 'hdb') -> dict[str, float]:
    n = len(labels)
    is_outlier = labels == -1
    n_outliers = int(is_outlier.sum())
    valid = labels[~is_outlier]
    base = {
        f'n_clusters_{prefix}': 0,
        f'n_outliers_{prefix}': n_outliers,
        f'frac_outliers_{prefix}': (n_outliers / n) if n else 0.0,
        f'dom_cluster_frac_{prefix}': 0.0,
        f'cluster_size_entropy_{prefix}': 0.0,
    }
    if len(valid) == 0:
        return base
    _, counts = np.unique(valid, return_counts=True)
    p = counts / counts.sum()
    H = -float((p * np.log(p + 1e-12)).sum())
    base.update({
        f'n_clusters_{prefix}': len(counts),
        f'dom_cluster_frac_{prefix}': float(counts.max() / n),
        f'cluster_size_entropy_{prefix}': H,
    })
    return base


def alignment(
    cluster_labels: NDArray[np.int_],
    facet_combined: NDArray[np.object_],
    prefix: str = 'hdb',
) -> dict[str, float]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    mask = cluster_labels != -1
    if (
        mask.sum() < 2
        or len(np.unique(facet_combined[mask])) < 2
        or len(np.unique(cluster_labels[mask])) < 2
    ):
        return {
            f'nmi_cluster_facet_{prefix}': float('nan'),
            f'ari_cluster_facet_{prefix}': float('nan'),
        }
    return {
        f'nmi_cluster_facet_{prefix}': float(
            normalized_mutual_info_score(facet_combined[mask], cluster_labels[mask])
        ),
        f'ari_cluster_facet_{prefix}': float(
            adjusted_rand_score(facet_combined[mask], cluster_labels[mask])
        ),
    }
