from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from experiments.medical_dataset_gen.evaluation.retrieval_utils import select_indices
from experiments.medical_dataset_gen.query_geometry.geom_plots_configs import (
    BACKGROUND_OUTLIER_COLOR,
    BACKGROUND_OUTLIER_LABEL_ID,
    BACKGROUND_OUTLIER_ROLE,
    SAME_CONDITION_WRONG_AXIS_LABEL_ID,
    SAME_CONDITION_WRONG_AXIS_ROLE,
    POINT_DISTRACTOR_TYPE_COLORS,
    UNSELECTED_BACKGROUND_COLOR,
)
from experiments.medical_dataset_gen.schemas.query_geometry_schemas import (
    GeometryArtifact,
    GeometrySelection,
)
from experiments.medical_dataset_gen.schemas.retrieval_schemas import RetrievalStrategy


def plot_strategy_overlay(artifact: GeometryArtifact, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    strategies = artifact.selections.keys()
    if not strategies:
        return

    fig, axes = plt.subplots(
        1, len(strategies), figsize=(5.3 * len(strategies), 5.2), sharex=True, sharey=True
    )
    palette = label_palette(artifact)
    axes_list = [axes] if len(strategies) == 1 else list(axes)
    for ax, strategy in zip(axes_list, strategies, strict=True):
        payload = artifact.selections[strategy]
        title = _selection_panel_title(
            artifact=artifact,
            strategy=strategy,
            lam=payload.lam,
            local_indices=payload.local_indices,
        )
        _plot_selection_panel(
            ax=ax,
            artifact=artifact,
            palette=palette,
            local_indices=payload.local_indices,
            title=title,
        )
    legend_handles = _selection_legend_handles(axes_list)
    fig.legend(
        legend_handles.values(),
        legend_handles.keys(),
        fontsize=7,
        frameon=False,
        loc='center left',
        bbox_to_anchor=(0.985, 0.5),
    )
    fig.suptitle(
        f'{artifact_title_prefix(artifact)}: selected chunks over same embedding coordinates',
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 0.92, 0.96))
    fig.savefig(out_dir / 'strategy_selection_overlay.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_full_strategy_selection_overlay(
    artifact: GeometryArtifact,
    out_dir: Path,
    *,
    k: int | None = None,
) -> None:
    import matplotlib.pyplot as plt

    lambda_values = [float(lam) for lam in artifact.lambda_values]
    if not lambda_values:
        return
    render_k = int(artifact.k if k is None else k)
    effective_k = min(render_k, len(artifact.sim_to_query))
    selection_variants = _selection_variants_for_k(artifact, effective_k)
    rows = ['mmr', 'fac_loc']
    row_variants = {
        strategy: _meaningful_lambda_selection_variants(
            selection_variants.get(strategy, []),
            lambda_values=lambda_values,
        )
        for strategy in rows
    }
    top_row_cols = 3
    n_cols = max(top_row_cols, *(len(row_variants[strategy]) for strategy in rows))
    fig, axes = plt.subplots(
        3,
        n_cols,
        figsize=(5.1 * n_cols, 5.0 * 3),
        squeeze=False,
        constrained_layout=True,
    )  # type: ignore

    palette = label_palette(artifact)
    topk_variants = selection_variants.get('top_k', [])
    if topk_variants:
        _plot_selection_panel(
            ax=axes[0, 0],
            artifact=artifact,
            palette=palette,
            local_indices=topk_variants[0].local_indices,
            title=_selection_panel_title(
                artifact=artifact,
                strategy='top_k',
                lam=None,
                local_indices=topk_variants[0].local_indices,
            ),
        )

    similarity_points = _plot_query_similarity_map_ax(
        axes[0, 1], artifact, include_title_prefix=False
    )
    legend_col_idx = 2
    if similarity_points is not None:
        fig.colorbar(
            similarity_points,
            ax=axes[0, 1],
            fraction=0.05,
            pad=0.02,
            label='query cosine similarity',
        )

    for col_idx in range(top_row_cols, n_cols):
        axes[0, col_idx].axis('off')

    for row_idx, strategy in enumerate(rows, start=1):
        variants = row_variants[strategy]

        for col_idx, payload in enumerate(variants):
            _plot_selection_panel(
                ax=axes[row_idx, col_idx],
                artifact=artifact,
                palette=palette,
                local_indices=payload.local_indices,
                title=_selection_panel_title(
                    artifact=artifact,
                    strategy=strategy,
                    lam=float(payload.lam) if payload.lam else None,
                    local_indices=payload.local_indices,
                ),
            )
        for col_idx in range(len(variants), n_cols):
            axes[row_idx, col_idx].axis('off')

    legend_handles = _selection_legend_handles([ax for ax in axes.ravel() if ax.get_visible()])
    _draw_selection_legend_panel(axes[0, legend_col_idx], legend_handles)

    fig.suptitle(
        f'{artifact_title_prefix(artifact)}: top-k, query cosine maps, and lambda sweeps '
        f'at k={effective_k}' + (f' (requested {render_k})' if render_k != effective_k else ''),
        fontsize=12,
    )
    fig.savefig(
        out_dir / f'full_strategy_selection_overlay_k{render_k}.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.close(fig)


def plot_query_overview_4panel(
    artifact: GeometryArtifact,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11), constrained_layout=True)  # type: ignore
    ax_map, ax_cos, ax_rank, ax_cluster = axes.ravel()

    _plot_query_map_ax(ax_map, artifact, include_title_prefix=False)
    ax_map.legend(fontsize=6, frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))

    similarity_points = _plot_query_similarity_map_ax(ax_cos, artifact, include_title_prefix=False)
    if similarity_points is not None:
        fig.colorbar(
            similarity_points,
            ax=ax_cos,
            fraction=0.05,
            pad=0.02,
            label='query cosine',
        )
    ax_cos.legend(fontsize=6, frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))

    _plot_query_rank_ax(ax_rank, artifact, include_title_prefix=False)
    ax_rank.legend(fontsize=6, frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))

    _plot_discovered_clusters_ax(ax_cluster, artifact, include_title_prefix=False)
    ax_cluster.legend(fontsize=6, frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f'{artifact_title_prefix(artifact)}: embedding geometry overview', fontsize=14)
    fig.savefig(out_dir / 'query_overview_4panel.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_cluster_quality_overview(stats: pl.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    cols = [
        ('gold_silhouette_cosine', 'Gold-facet silhouette'),
        ('mean_in_facet_similarity', 'Mean in-facet cosine'),
        ('mean_cross_facet_similarity', 'Mean cross-facet cosine'),
        ('hdbscan_ari_hidden', 'HDBSCAN ARI vs hidden labels'),
        ('hdbscan_noise_rate', 'HDBSCAN noise rate'),
    ]
    fig, axes = plt.subplots(1, len(cols), figsize=(4.1 * len(cols), 4.2))
    for ax, (col, title) in zip(axes, cols, strict=True):
        values = stats[col].drop_nulls().to_list() if col in stats.columns else []
        if values:
            ax.boxplot(values, tick_labels=['queries'])
        else:
            ax.text(0.5, 0.5, 'n/a', ha='center')
        ax.set_title(title, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Embedding geometry diagnostics across visualized queries', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / 'cluster_quality_overview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_selection_panel(
    ax: Any,
    artifact: GeometryArtifact,
    *,
    palette: dict[str, Any],
    local_indices: Any,
    title: str,
) -> None:
    coords = artifact.coords
    ax.scatter(coords[:, 0], coords[:, 1], color='#dddddd', s=24, alpha=0.55, edgecolors='none')
    selected = {int(i) for i in local_indices}
    label_groups = _label_groups(artifact)
    for label, indices in label_groups.items():
        first_idx = indices[0]
        if _is_background_outlier_point(artifact, first_idx):
            continue
        idx = [i for i in indices if i in selected]
        if not idx:
            continue
        marker = _label_marker(artifact, first_idx)
        scatter_kwargs: dict[str, Any] = {
            's': 74,
            'color': palette[label],
            'marker': marker,
            'zorder': 4,
        }
        if marker == 'x':
            scatter_kwargs['linewidths'] = 1.35
        else:
            scatter_kwargs['edgecolors'] = 'black'
            scatter_kwargs['linewidths'] = 0.65
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            label=label,
            **scatter_kwargs,
        )
    _plot_background_outliers_for_selection(
        ax,
        artifact,
        selected=selected,
        palette=palette,
        label_groups=label_groups,
    )
    draw_query_marker(ax, artifact)
    ax.set_title(title)
    _apply_embedding_limits(ax, artifact)
    ax.grid(alpha=0.18)


def _selection_variants_for_k(
    artifact: GeometryArtifact,
    k: int,
) -> dict[str, list[GeometrySelection]]:
    sim_to_query = np.asarray(artifact.sim_to_query)
    sim_matrix = np.asarray(artifact.sim_matrix)
    mmr_window = artifact.mmr_window
    lambda_values = [float(lam) for lam in artifact.lambda_values]

    variants: dict[str, list[GeometrySelection]] = {
        'top_k': [
            GeometrySelection(
                local_indices=select_indices(
                    strategy='top_k',
                    sim_to_query=sim_to_query,
                    sim_matrix=sim_matrix,
                    k=k,
                    lam=None,
                    mmr_window=mmr_window,
                ),
                lam=None,
            )
        ]
    }
    for strategy in cast(list[RetrievalStrategy], ['mmr', 'fac_loc']):
        variants[strategy] = [
            GeometrySelection(
                local_indices=select_indices(
                    strategy=strategy,
                    sim_to_query=sim_to_query,
                    sim_matrix=sim_matrix,
                    k=k,
                    lam=float(lam),
                    mmr_window=mmr_window,
                ),
                lam=float(lam),
            )
            for lam in lambda_values
        ]
    return variants


def _meaningful_lambda_selection_variants(
    variants: list[GeometrySelection],
    *,
    lambda_values: list[float],
) -> list[GeometrySelection]:
    ordered_variants: list[GeometrySelection] = []
    for lam in sorted(lambda_values, reverse=True):
        payload = next(
            (
                variant
                for variant in variants
                if variant.lam is not None and abs(float(variant.lam) - lam) < 1e-12
            ),
            None,
        )
        if payload is not None:
            ordered_variants.append(payload)

    meaningful: list[GeometrySelection] = []
    previous_signature: frozenset[int] | None = None
    for payload in ordered_variants:
        signature = frozenset(int(index) for index in payload.local_indices)
        if previous_signature is None or signature != previous_signature:
            meaningful.append(payload)
        previous_signature = signature
    return meaningful


def _selection_panel_title(
    artifact: GeometryArtifact,
    strategy: str,
    lam: float | None,
    local_indices: Any,
    *,
    lambda_label: float | None = None,
) -> str:
    selected = {int(i) for i in local_indices}
    selected_labels = [artifact.label_ids[i] for i in selected]
    selected_gold = sum(1 for i in selected if artifact.is_gold[i])
    selected_facets = len({x for x in selected_labels if x.startswith(artifact.query_id)})
    parts = []
    if strategy == 'top_k' and lambda_label is not None:
        parts.append(f'lambda={lambda_label:.2f}')
    parts.append(strategy_title(strategy, lam))
    parts.append(f'facets={selected_facets}, gold={selected_gold}/{len(selected)}')
    return '\n'.join(parts)


def _selection_legend_handles(axes: list[Any]) -> dict[str, Any]:
    from matplotlib.lines import Line2D

    legend_handles: dict[str, Any] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            legend_handles.setdefault(label, handle)
    legend_handles.setdefault(
        'query',
        Line2D(
            [0],
            [0],
            marker='*',
            color='none',
            markerfacecolor='black',
            markeredgecolor='white',
            markersize=12,
            label='query',
        ),
    )
    return legend_handles


def _draw_selection_legend_panel(ax: Any, legend_handles: dict[str, Any]) -> None:
    ax.axis('off')
    ax.set_title('Legend')
    ax.legend(
        legend_handles.values(),
        legend_handles.keys(),
        fontsize=7,
        frameon=False,
        loc='center left',
        bbox_to_anchor=(0.0, 0.5),
    )


def _plot_query_map_ax(
    ax: Any, artifact: GeometryArtifact, *, include_title_prefix: bool = True
) -> None:
    palette = label_palette(artifact)
    label_groups = _label_groups(artifact)
    for label, idx in label_groups.items():
        first_idx = idx[0]
        if _is_background_outlier_point(artifact, first_idx):
            continue
        marker = _label_marker(artifact, first_idx)
        scatter_kwargs: dict[str, Any] = {
            's': 34,
            'alpha': 0.78,
            'color': palette[label],
            'marker': marker,
            'label': label,
        }
        if marker == 'x':
            scatter_kwargs['linewidths'] = 1.15
        else:
            scatter_kwargs['edgecolors'] = 'none'
        ax.scatter(
            artifact.coords[idx, 0],
            artifact.coords[idx, 1],
            **scatter_kwargs,
        )
    _plot_background_outliers(
        ax,
        artifact,
        palette=palette,
        label_groups=label_groups,
    )
    draw_query_marker(ax, artifact)
    title = (
        f'Candidate-pool map ({artifact.reduction_method}, n={len(artifact.candidate_chunk_ids)})'
    )
    ax.set_title(_axis_title(artifact, title, include_title_prefix=include_title_prefix))
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    _apply_embedding_limits(ax, artifact)
    ax.grid(alpha=0.18)


def _plot_query_similarity_map_ax(
    ax: Any, artifact: GeometryArtifact, *, include_title_prefix: bool = True
) -> Any:
    return _plot_query_similarity_points(
        ax=ax,
        artifact=artifact,
        coords=artifact.coords,
        query_coord=artifact.query_coord,
        sim_to_query=artifact.sim_to_query,
        title=f'Map colored by query cosine ({artifact.reduction_method}, n={len(artifact.coords)})',
        x_label='dim 1',
        y_label='dim 2',
        include_title_prefix=include_title_prefix,
    )


def _plot_query_similarity_points(
    ax: Any,
    artifact: GeometryArtifact,
    *,
    coords: NDArray[np.float32],
    query_coord: NDArray[np.float32],
    sim_to_query: NDArray[np.float32],
    title: str,
    x_label: str,
    y_label: str,
    include_title_prefix: bool,
) -> Any:
    order = np.arange(len(artifact.candidate_chunk_ids))
    plot_coords = coords[order]
    plot_sims = sim_to_query[order]
    palette = label_palette(artifact)
    label_groups = _label_groups(artifact)

    vmin = float(plot_sims.min()) if len(plot_sims) else 0.0
    vmax = float(plot_sims.max()) if len(plot_sims) else 1.0
    points = None
    for label, indices in label_groups.items():
        idx = np.array(indices, dtype=int)
        first_idx = indices[0]
        marker = _label_marker(artifact, first_idx)
        if _is_background_outlier_point(artifact, first_idx):
            points = _scatter_circled_x(
                ax,
                plot_coords[idx, 0],
                plot_coords[idx, 1],
                color_values=plot_sims[idx],
                circle_edgecolors=palette[label],
                cmap='viridis',
                vmin=vmin,
                vmax=vmax,
                s=74,
                alpha=0.96,
                linewidth=0.9,
                zorder=5,
                label=label,
            )
            continue
        scatter_kwargs: dict[str, Any] = {
            'c': plot_sims[idx],
            'cmap': 'viridis',
            'vmin': vmin,
            'vmax': vmax,
            'alpha': 0.92,
            'marker': marker,
            'zorder': 4,
            'label': label,
        }
        if marker == 'o':
            scatter_kwargs.update({'s': 44, 'edgecolors': 'none'})
        elif marker == 'D':
            scatter_kwargs.update({'s': 52, 'edgecolors': 'black', 'linewidths': 0.6})
        else:
            scatter_kwargs.update({'s': 38, 'linewidths': 1.25})
        points = ax.scatter(
            plot_coords[idx, 0],
            plot_coords[idx, 1],
            **scatter_kwargs,
        )
    if points is None:
        points = ax.scatter([], [], c=[], cmap='viridis', vmin=vmin, vmax=vmax)
    _draw_query_marker_at(ax, query_coord)
    ax.set_title(_axis_title(artifact, title, include_title_prefix=include_title_prefix))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _apply_coord_limits(ax, coords, query_coord)
    ax.grid(alpha=0.18)
    return points


def _plot_query_rank_ax(
    ax: Any, artifact: GeometryArtifact, *, include_title_prefix: bool = True
) -> None:
    ranks = np.arange(1, len(artifact.sim_to_query) + 1)
    palette = label_palette(artifact)
    label_groups = _label_groups(artifact)
    for label, idx in label_groups.items():
        first_idx = idx[0]
        if _is_background_outlier_point(artifact, first_idx):
            continue
        marker = _label_marker(artifact, first_idx)
        scatter_kwargs: dict[str, Any] = {
            'color': palette[label],
            's': 28,
            'alpha': 0.78,
            'marker': marker,
            'label': label,
        }
        if marker == 'x':
            scatter_kwargs['linewidths'] = 1.15
        else:
            scatter_kwargs['edgecolors'] = 'none'
        ax.scatter(
            ranks[idx],
            artifact.sim_to_query[idx],
            **scatter_kwargs,
        )
    _plot_background_outliers_rank(
        ax,
        artifact,
        ranks=ranks,
        palette=palette,
        label_groups=label_groups,
    )
    ax.axvline(artifact.k, color='black', lw=1.0, ls='--', alpha=0.7, label=f'k={artifact.k}')
    ax.set_xlabel('rank by query cosine')
    ax.set_ylabel('query cosine similarity')
    ax.set_title(
        _axis_title(artifact, 'Query-similarity ranking', include_title_prefix=include_title_prefix)
    )
    ax.grid(axis='y', alpha=0.25)


def _plot_discovered_clusters_ax(
    ax: Any,
    artifact: GeometryArtifact,
    *,
    include_title_prefix: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    coords = artifact.coords
    cluster_labels = artifact.cluster_labels
    unique = sorted(set(int(x) for x in cluster_labels))
    cmap = plt.get_cmap('tab20')  # type: ignore
    palette = label_palette(artifact)
    label_groups = _label_groups(artifact)
    for idx, cluster_id in enumerate(unique):
        mask = cluster_labels == cluster_id
        color = '#bbbbbb' if cluster_id == -1 else cmap(idx % 20)
        label = 'noise' if cluster_id == -1 else f'cluster {cluster_id}'
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=34,
            alpha=0.78,
            color=color,
            label=label,
            edgecolors='none',
        )
    _plot_background_outliers(
        ax,
        artifact,
        palette=palette,
        label_groups=label_groups,
        fallback_color='#333333',
    )
    draw_query_marker(ax, artifact)
    ax.set_title(
        _axis_title(artifact, 'HDBSCAN clusters', include_title_prefix=include_title_prefix)
    )
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    _apply_embedding_limits(ax, artifact)
    ax.grid(alpha=0.18)


def label_palette(artifact: GeometryArtifact) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    label_groups = _label_groups(artifact)
    gold_labels = [label for label, indices in label_groups.items() if artifact.is_gold[indices[0]]]
    gold_cmap = plt.get_cmap('tab20')  # type: ignore
    background_cluster_ids = sorted(
        {
            _cluster_id_for_index(artifact, indices[0]) or label
            for label, indices in label_groups.items()
            if _is_background_outlier_point(artifact, indices[0])
        }
    )
    background_cmap = plt.get_cmap('Dark2')  # type: ignore
    background_palette = {
        cluster_id: background_cmap(i % max(1, background_cmap.N))
        for i, cluster_id in enumerate(background_cluster_ids)
    }
    palette: dict[str, Any] = {}
    for index, label in enumerate(gold_labels):
        palette[label] = gold_cmap(index % 20)
    for label, indices in label_groups.items():
        if label in palette:
            continue
        first_idx = indices[0]
        if _is_background_outlier_point(artifact, first_idx):
            cluster_id = _cluster_id_for_index(artifact, first_idx) or label
            palette[label] = background_palette.get(cluster_id, BACKGROUND_OUTLIER_COLOR)
            continue
        palette[label] = POINT_DISTRACTOR_TYPE_COLORS.get(
            _distractor_type_for_index(artifact, first_idx),
            POINT_DISTRACTOR_TYPE_COLORS['hard_distractor'],
        )
    return palette


def _label_groups(artifact: GeometryArtifact) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for idx, label in enumerate(artifact.labels):
        grouped.setdefault(label, []).append(idx)
    return dict(sorted(grouped.items(), key=lambda item: item[0].lower()))


def _label_marker(artifact: GeometryArtifact, idx: int) -> str:
    if _is_background_outlier_point(artifact, idx):
        return 'x'
    if _is_same_condition_wrong_axis_point(artifact, idx):
        return 'D'
    return 'x' if not artifact.is_gold[idx] else 'o'


def _is_background_outlier_point(artifact: GeometryArtifact, idx: int) -> bool:
    label_id = artifact.label_ids[idx]
    role = artifact.roles[idx]
    return label_id == BACKGROUND_OUTLIER_LABEL_ID or role == BACKGROUND_OUTLIER_ROLE


def _is_same_condition_wrong_axis_point(artifact: GeometryArtifact, idx: int) -> bool:
    label_id = artifact.label_ids[idx]
    role = artifact.roles[idx]
    return label_id == SAME_CONDITION_WRONG_AXIS_LABEL_ID or role == SAME_CONDITION_WRONG_AXIS_ROLE


def _distractor_type_for_index(artifact: GeometryArtifact, idx: int) -> str:
    chunk_id = artifact.candidate_chunk_ids[idx]
    qrel = artifact.qrel_by_chunk_id.get(chunk_id)
    return qrel.distractor_type if qrel is not None and qrel.distractor_type else 'hard_distractor'


def _cluster_id_for_index(artifact: GeometryArtifact, idx: int) -> str | None:
    chunk_id = artifact.candidate_chunk_ids[idx]
    qrel = artifact.qrel_by_chunk_id.get(chunk_id)
    return qrel.cluster_id if qrel is not None else None


def _background_outlier_indices(artifact: GeometryArtifact) -> list[int]:
    return [
        idx
        for idx in range(len(artifact.candidate_chunk_ids))
        if _is_background_outlier_point(artifact, idx)
    ]


def _plot_background_outliers(
    ax: Any,
    artifact: GeometryArtifact,
    *,
    palette: dict[str, Any] | None,
    label_groups: dict[str, list[int]],
    fallback_color: str | None = None,
) -> None:
    coords = artifact.coords
    for label, indices in label_groups.items():
        first_idx = indices[0]
        if not _is_background_outlier_point(artifact, first_idx):
            continue
        _scatter_circled_x(
            ax,
            coords[indices, 0],
            coords[indices, 1],
            color=(palette or {}).get(label, fallback_color or BACKGROUND_OUTLIER_COLOR),
            s=54,
            alpha=0.95,
            label=label,
            linewidth=0.75,
            zorder=5,
        )


def _plot_background_outliers_for_selection(
    ax: Any,
    artifact: GeometryArtifact,
    *,
    selected: set[int],
    palette: dict[str, Any],
    label_groups: dict[str, list[int]],
) -> None:
    coords = artifact.coords
    for label, indices in label_groups.items():
        first_idx = indices[0]
        if not _is_background_outlier_point(artifact, first_idx):
            continue
        selected_idx = [point_idx for point_idx in indices if point_idx in selected]
        unselected_idx = [point_idx for point_idx in indices if point_idx not in selected]
        if unselected_idx:
            _scatter_circled_x(
                ax,
                coords[unselected_idx, 0],
                coords[unselected_idx, 1],
                color=UNSELECTED_BACKGROUND_COLOR,
                s=42,
                alpha=0.58,
                linewidth=0.55,
                zorder=3,
            )
        if selected_idx:
            _scatter_circled_x(
                ax,
                coords[selected_idx, 0],
                coords[selected_idx, 1],
                color=palette[label],
                s=58,
                alpha=0.95,
                label=label,
                linewidth=0.75,
                zorder=5,
            )


def _plot_background_outliers_rank(
    ax: Any,
    artifact: GeometryArtifact,
    *,
    ranks: NDArray[np.int_],
    palette: dict[str, Any],
    label_groups: dict[str, list[int]],
) -> None:
    for label, indices in label_groups.items():
        first_idx = indices[0]
        if not _is_background_outlier_point(artifact, first_idx):
            continue
        _scatter_circled_x(
            ax,
            ranks[indices],
            artifact.sim_to_query[indices],
            color=palette[label],
            s=50,
            alpha=0.94,
            label=label,
            linewidth=0.75,
            zorder=5,
        )


def _scatter_circled_x(
    ax: Any,
    x: Any,
    y: Any,
    *,
    color: str | None = None,
    color_values: Any = None,
    circle_edgecolors: Any = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    s: Any = 54,
    alpha: float = 0.95,
    label: str | None = None,
    linewidth: float = 0.75,
    zorder: int = 5,
) -> Any:
    if color_values is not None:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap or 'viridis')  # type: ignore
        circle_kwargs = {
            'edgecolors': (
                circle_edgecolors if circle_edgecolors is not None else cmap_obj(norm(color_values))
            )
        }
        x_kwargs = {'c': color_values, 'cmap': cmap, 'vmin': vmin, 'vmax': vmax}
    else:
        circle_kwargs = {'edgecolors': color}
        x_kwargs = {'color': color}
    circle = ax.scatter(
        x,
        y,
        s=s,
        alpha=alpha,
        marker='o',
        facecolors='none',
        linewidths=linewidth,
        label=label,
        zorder=zorder,
        **circle_kwargs,
    )
    x_points = ax.scatter(
        x,
        y,
        s=np.asarray(s) * 0.46,
        alpha=alpha,
        marker='x',
        linewidths=linewidth,
        zorder=zorder + 0.1,
        **x_kwargs,
    )
    return x_points if color_values is not None else circle


def draw_query_marker(ax: Any, artifact: GeometryArtifact) -> None:
    _draw_query_marker_at(ax, artifact.query_coord)


def _draw_query_marker_at(ax: Any, query_coord: NDArray[np.float32]) -> None:
    ax.scatter(
        [query_coord[0]],
        [query_coord[1]],
        marker='*',
        s=220,
        color='black',
        edgecolors='white',
        linewidths=0.8,
        label='query',
        zorder=6,
    )


def artifact_title_prefix(artifact: GeometryArtifact) -> str:
    condition = artifact.query.condition_display or artifact.query.condition_id
    return f'{artifact.query_id} - {condition}'


def _axis_title(artifact: GeometryArtifact, title: str, *, include_title_prefix: bool) -> str:
    if include_title_prefix:
        return f'{artifact_title_prefix(artifact)}\n{title}'
    return title


def _apply_embedding_limits(ax: Any, artifact: GeometryArtifact) -> None:
    _apply_coord_limits(ax, artifact.coords, artifact.query_coord)


def _apply_coord_limits(
    ax: Any,
    coords: NDArray[np.float32],
    query_coord: NDArray[np.float32],
) -> None:
    x_left, x_right, y_bottom, y_top = _coord_limits(coords, query_coord)
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)


def _coord_limits(
    coords: NDArray[np.float32],
    query_coord: NDArray[np.float32],
) -> tuple[float, float, float, float]:
    coords_all = np.vstack([coords, query_coord[None, :]])
    x_min, y_min = coords_all.min(axis=0)
    x_max, y_max = coords_all.max(axis=0)
    x_pad = max(float(x_max - x_min) * 0.05, 0.5)
    y_pad = max(float(y_max - y_min) * 0.05, 0.5)
    return (
        float(coords_all[:, 0].min()) - x_pad,
        float(coords_all[:, 0].max()) + x_pad,
        float(coords_all[:, 1].min()) - y_pad,
        float(coords_all[:, 1].max()) + y_pad,
    )


def strategy_title(strategy: str, lam: float | None) -> str:
    if strategy == 'top_k':
        return 'top-k'
    label = 'MMR' if strategy == 'mmr' else 'Coverage'
    return f'{label} lambda={lam:.2f}' if lam is not None else label
