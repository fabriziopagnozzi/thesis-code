"""Compute per-query diagnostics for the benchmark's embedding geometry.

This module exists to measure whether the generated corpus has the expected
facet separation, redundancy, and distractor structure. It uses simple counts
and similarity summaries so geometry checks stay fast and directly tied to the
synthetic design.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from numpy.typing import NDArray

from experiments.medical_dataset_gen.schemas.query_geometry_schemas import (
    EmbeddingGeometry2DPoint,
    EmbeddingGeometryQueryStats,
    GeometryArtifact,
    GeometryChunkLike,
    GeometryQrelLike,
)


def build_geometry_points_row(artifact: GeometryArtifact) -> list[EmbeddingGeometry2DPoint]:
    selected_sets = {
        strategy: {int(i) for i in payload.local_indices}
        for strategy, payload in artifact.selections.items()
    }
    geom_points: list[EmbeddingGeometry2DPoint] = []

    # Append all the chunks
    for idx, chunk_id in enumerate(artifact.candidate_chunk_ids):
        row = artifact.qrel_by_chunk_id.get(chunk_id, artifact.chunk_by_id[chunk_id])
        hdbscan_label = (
            int(artifact.cluster_labels[idx]) if artifact.cluster_labels is not None else None
        )
        geom_points.append(
            make_geom_point_row(
                artifact=artifact,
                chunk_id=chunk_id,
                row=row,
                idx=idx,
                point_kind='chunk',
                rank=idx + 1,
                x=float(artifact.coords[idx, 0]),
                y=float(artifact.coords[idx, 1]),
                sim_to_query=float(artifact.sim_to_query[idx]),
                plot_label=artifact.labels[idx],
                label_id=artifact.label_ids[idx],
                cluster_role=artifact.roles[idx],
                is_gold=bool(artifact.is_gold[idx]),
                hdbscan_label=hdbscan_label,
                selected_top_k=idx in selected_sets.get('top_k', set()),
                selected_mmr=idx in selected_sets.get('mmr', set()),
                selected_fac_loc=idx in selected_sets.get('fac_loc', set()),
            )
        )

    # Append the query
    geom_points.append(
        make_geom_point_row(
            artifact=artifact,
            chunk_id=None,
            row={},
            idx=None,
            point_kind='query',
            rank=0,
            x=float(artifact.query_coord[0]),
            y=float(artifact.query_coord[1]),
            sim_to_query=1.0,
            plot_label='query',
            label_id='query',
            cluster_role='query',
            is_gold=False,
            hdbscan_label=None,
            selected_top_k=False,
            selected_mmr=False,
            selected_fac_loc=False,
        )
    )

    return geom_points


def query_stats(artifact: GeometryArtifact) -> EmbeddingGeometryQueryStats:
    selected_summaries = {
        strategy: selection_summary(artifact, payload.local_indices)
        for strategy, payload in artifact.selections.items()
    }
    gold_silhouette = gold_silhouette_cosine(artifact)
    in_sim, cross_sim = in_cross_similarity(artifact)
    gold_mask = np.array(artifact.is_gold, dtype=bool)
    distractor_mask = ~gold_mask

    row: EmbeddingGeometryQueryStats = {
        'query_id': artifact.query_id,
        'selection_group': artifact.selection_group,
        'query_type': artifact.query.query_type,
        'condition_id': artifact.query.condition_id,
        'pool_scope': artifact.pool_scope,
        'pool_size': len(artifact.candidate_chunk_ids),
        'plot_k': artifact.k,
        'reduction_method': artifact.reduction_method,
        'n_hidden_labels': len(set(artifact.label_ids)),
        'n_gold_points': int(gold_mask.sum()),
        'n_distractor_points': int(distractor_mask.sum()),
        'gold_silhouette_cosine': gold_silhouette,
        'mean_in_facet_similarity': in_sim,
        'mean_cross_facet_similarity': cross_sim,
        'in_minus_cross_similarity': in_sim - cross_sim,
        'query_to_gold_mean': masked_mean(artifact.sim_to_query, gold_mask),
        'query_to_distractor_mean': masked_mean(artifact.sim_to_query, distractor_mask),
    }
    cluster_labels = artifact.cluster_labels
    if cluster_labels is not None:
        hidden_label_codes = string_codes(artifact.label_ids)
        ari, nmi = cluster_agreement(hidden_label_codes, cluster_labels)
        non_noise = [int(x) for x in cluster_labels if int(x) != -1]
        row['hdbscan_n_clusters'] = len(set(non_noise))
        row['hdbscan_noise_rate'] = float(np.mean(cluster_labels == -1)) if len(cluster_labels) else 0.0
        row['hdbscan_ari_hidden'] = ari
        row['hdbscan_nmi_hidden'] = nmi
    for strategy, summary in selected_summaries.items():
        row[f'{strategy}_n_facets_selected'] = summary['n_facets_selected']
        row[f'{strategy}_gold_precision'] = summary['gold_precision']
        row[f'{strategy}_distractor_rate'] = summary['distractor_rate']
        row[f'{strategy}_dominant_fraction'] = summary['dominant_fraction']

    return row


def selection_summary(
    artifact: GeometryArtifact,
    local_indices: NDArray[np.intp],
) -> dict[str, float | int]:
    indices = [int(i) for i in local_indices]
    labels = [artifact.label_ids[i] for i in indices]
    gold = [bool(artifact.is_gold[i]) for i in indices]
    facet_labels = [label for label, is_gold in zip(labels, gold, strict=True) if is_gold]
    counts = Counter(facet_labels)
    dominant = counts.most_common(1)[0][1] if counts else 0
    return {
        'n_facets_selected': len(set(facet_labels)),
        'gold_precision': sum(gold) / len(gold) if gold else 0.0,
        'distractor_rate': 1.0 - (sum(gold) / len(gold)) if gold else 0.0,
        'dominant_fraction': dominant / len(indices) if indices else 0.0,
    }


def gold_silhouette_cosine(artifact: GeometryArtifact) -> float | None:
    from sklearn.metrics import silhouette_score

    gold_idx = [i for i, flag in enumerate(artifact.is_gold) if flag]
    labels = [artifact.label_ids[i] for i in gold_idx]
    if len(set(labels)) < 2 or len(gold_idx) <= len(set(labels)):
        return None
    vectors = artifact.candidate_vectors[gold_idx]
    try:
        return float(silhouette_score(vectors, labels, metric='cosine'))
    except Exception:
        return None


def in_cross_similarity(artifact: GeometryArtifact) -> tuple[float, float]:
    gold_idx = [i for i, flag in enumerate(artifact.is_gold) if flag]
    labels = np.array([artifact.label_ids[i] for i in gold_idx])
    if len(gold_idx) < 2 or len(set(labels.tolist())) < 2:
        return 0.0, 0.0
    sim = artifact.sim_matrix[np.ix_(gold_idx, gold_idx)]
    same = labels[:, None] == labels[None, :]
    not_self = ~np.eye(len(gold_idx), dtype=bool)
    in_vals = sim[same & not_self]
    cross_vals = sim[~same & not_self]
    return (
        float(in_vals.mean()) if len(in_vals) else 0.0,
        float(cross_vals.mean()) if len(cross_vals) else 0.0,
    )


def cluster_agreement(
    hidden_labels: NDArray[np.int32],
    cluster_labels: NDArray[np.int32],
) -> tuple[float | None, float | None]:
    if len(hidden_labels) < 2 or len(set(hidden_labels.tolist())) < 2:
        return None, None
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        return (
            float(adjusted_rand_score(hidden_labels, cluster_labels)),
            float(normalized_mutual_info_score(hidden_labels, cluster_labels)),
        )
    except Exception:
        return None, None


def masked_mean(values: NDArray[np.float32], mask: NDArray[np.bool_]) -> float | None:
    if not bool(mask.any()):
        return None
    return float(values[mask].mean())


def string_codes(labels: list[str]) -> NDArray[np.int32]:
    mapping = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    return np.array([mapping[label] for label in labels], dtype=np.int32)


def make_geom_point_row(
    artifact: GeometryArtifact,
    chunk_id: str | None,
    row: GeometryQrelLike | GeometryChunkLike | dict[object, object],
    idx: int | None,
    point_kind: str,
    rank: int,
    x: float,
    y: float,
    sim_to_query: float,
    plot_label: str,
    label_id: str,
    cluster_role: str,
    is_gold: bool,
    hdbscan_label: int | None,
    selected_top_k: bool,
    selected_mmr: bool,
    selected_fac_loc: bool,
) -> EmbeddingGeometry2DPoint:
    _ = idx
    facet_id = getattr(row, 'facet_id', None)
    target_facet_id = getattr(row, 'target_facet_id', None)
    distractor_type = getattr(row, 'distractor_type', None)
    return {
        'query_id': artifact.query_id,
        'selection_group': artifact.selection_group,
        'point_kind': point_kind,
        'chunk_id': chunk_id,
        'rank': rank,
        'x': x,
        'y': y,
        'reduction_method': artifact.reduction_method,
        'sim_to_query': sim_to_query,
        'plot_label': plot_label,
        'label_id': label_id,
        'cluster_role': cluster_role,
        'is_gold': is_gold,
        'facet_id': facet_id,
        'target_facet_id': target_facet_id,
        'distractor_type': distractor_type,
        'hdbscan_label': hdbscan_label,
        'selected_top_k': selected_top_k,
        'selected_mmr': selected_mmr,
        'selected_fac_loc': selected_fac_loc,
    }
