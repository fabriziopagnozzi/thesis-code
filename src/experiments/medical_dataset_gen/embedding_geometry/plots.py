from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from experiments.medical_dataset_gen.retrieval.utils import Strategy, select_indices

_FIXED_LABEL_COLORS = {
    'soft distractor: same condition, wrong subgroup': '#f3a6a6',
    'hard distractor: wrong condition, same subgroup': '#d62728',
    'hard distractor: wrong condition, same answer axis': '#8c1d18',
    'hard distractor': '#d62728',
    'off-query wrong-condition chunks': '#d62728',
}

_DISTRACTOR_LABELS = set(_FIXED_LABEL_COLORS)


def plot_query_map(artifact: dict[str, Any], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    _plot_query_map_ax(ax, artifact)
    ax.legend(fontsize=7, frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(out_dir / 'candidate_pool_map.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_strategy_overlay(artifact: dict[str, Any], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    strategies = [s for s in ['top_k', 'mmr', 'fac_loc'] if s in artifact['selections']]
    if not strategies:
        return
    fig, axes = plt.subplots(
        1, len(strategies), figsize=(5.3 * len(strategies), 5.2), sharex=True, sharey=True
    )
    palette = label_palette(artifact['labels'])
    axes_list = [axes] if len(strategies) == 1 else list(axes)
    for ax, strategy in zip(axes_list, strategies, strict=True):
        payload = artifact['selections'][strategy]
        title = _selection_panel_title(
            artifact=artifact,
            strategy=strategy,
            lam=payload['lam'],
            local_indices=payload['local_indices'],
        )
        _plot_selection_panel(
            ax=ax,
            artifact=artifact,
            palette=palette,
            local_indices=payload['local_indices'],
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
    artifact: dict[str, Any],
    out_dir: Path,
    *,
    k: int | None = None,
) -> None:
    import matplotlib.pyplot as plt

    lambda_values = [float(lam) for lam in artifact.get('lambda_values', [])]
    if not lambda_values:
        return
    render_k = int(artifact['k'] if k is None else k)
    effective_k = min(render_k, len(artifact['sim_to_query']))
    selection_variants = _selection_variants_for_k(artifact, effective_k)
    n_cols = max(2, len(lambda_values))
    rows = ['mmr', 'fac_loc']
    fig, axes = plt.subplots(
        3,
        n_cols,
        figsize=(5.1 * n_cols, 5.0 * 3),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    palette = label_palette(artifact['labels'])
    topk_variants = selection_variants.get('top_k', [])
    if topk_variants:
        _plot_selection_panel(
            ax=axes[0, 0],
            artifact=artifact,
            palette=palette,
            local_indices=topk_variants[0]['local_indices'],
            title=_selection_panel_title(
                artifact=artifact,
                strategy='top_k',
                lam=None,
                local_indices=topk_variants[0]['local_indices'],
            ),
        )

    query_points = _plot_query_similarity_map_ax(axes[0, 1], artifact, include_title_prefix=False)
    fig.colorbar(
        query_points,
        ax=axes[0, 1],
        fraction=0.046,
        pad=0.04,
        label='query cosine similarity',
    )

    for col_idx in range(2, n_cols):
        axes[0, col_idx].axis('off')

    for col_idx, lam in enumerate(lambda_values):
        for row_idx, strategy in enumerate(rows, start=1):
            variants = selection_variants.get(strategy, [])
            if not variants:
                continue
            payload = next(
                (variant for variant in variants if abs(float(variant['lam']) - lam) < 1e-12),
                None,
            )
            if payload is None:
                continue
            _plot_selection_panel(
                ax=axes[row_idx, col_idx],
                artifact=artifact,
                palette=palette,
                local_indices=payload['local_indices'],
                title=_selection_panel_title(
                    artifact=artifact,
                    strategy=strategy,
                    lam=float(payload['lam']),
                    local_indices=payload['local_indices'],
                ),
            )

    for row_idx in [1, 2]:
        for col_idx in range(len(lambda_values), n_cols):
            axes[row_idx, col_idx].axis('off')

    legend_handles = _selection_legend_handles([ax for ax in axes.ravel() if ax.get_visible()])
    fig.legend(
        legend_handles.values(),
        legend_handles.keys(),
        fontsize=7,
        frameon=False,
        loc='center left',
        bbox_to_anchor=(0.985, 0.5),
    )
    fig.suptitle(
        f'{artifact_title_prefix(artifact)}: top-k, query cosine map, and lambda sweeps '
        f'at k={effective_k}' + (f' (requested {render_k})' if render_k != effective_k else ''),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 0.92, 0.95))
    fig.savefig(
        out_dir / f'full_strategy_selection_overlay_k{render_k}.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.close(fig)


def plot_query_similarity_map(artifact: dict[str, Any], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    points = _plot_query_similarity_map_ax(ax, artifact)
    fig.colorbar(points, ax=ax, fraction=0.046, pad=0.04, label='query cosine similarity')
    fig.tight_layout()
    fig.savefig(out_dir / 'query_cosine_similarity_map.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_query_rank(artifact: dict[str, Any], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.8))
    _plot_query_rank_ax(ax, artifact)
    ax.legend(fontsize=7, frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(out_dir / 'query_similarity_rank.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_discovered_clusters(artifact: dict[str, Any], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    _plot_discovered_clusters_ax(ax, artifact)
    ax.legend(fontsize=7, frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(out_dir / 'hdbscan_cluster_map.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_query_overview_4panel(artifact: dict[str, Any], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    ax_map, ax_cos, ax_rank, ax_cluster = axes.ravel()

    _plot_query_map_ax(ax_map, artifact, include_title_prefix=False)
    ax_map.legend(fontsize=6, frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))

    points = _plot_query_similarity_map_ax(ax_cos, artifact, include_title_prefix=False)
    fig.colorbar(points, ax=ax_cos, fraction=0.046, pad=0.04, label='query cosine')

    _plot_query_rank_ax(ax_rank, artifact, include_title_prefix=False)
    ax_rank.legend(fontsize=6, frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))

    _plot_discovered_clusters_ax(ax_cluster, artifact, include_title_prefix=False)
    ax_cluster.legend(fontsize=6, frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))

    fig.suptitle(f'{artifact_title_prefix(artifact)}: embedding geometry overview', fontsize=14)
    fig.tight_layout()
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
    artifact: dict[str, Any],
    *,
    palette: dict[str, Any],
    local_indices: Any,
    title: str,
) -> None:
    coords = artifact['coords']
    ax.scatter(coords[:, 0], coords[:, 1], color='#dddddd', s=24, alpha=0.55, edgecolors='none')
    selected = {int(i) for i in local_indices}
    for label in sorted(set(artifact['labels'])):
        idx = [i for i, value in enumerate(artifact['labels']) if value == label and i in selected]
        if not idx:
            continue
        marker = _label_marker(label)
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
    draw_query_marker(ax, artifact)
    ax.set_title(title)
    _apply_embedding_limits(ax, artifact)
    ax.grid(alpha=0.18)


def _selection_variants_for_k(
    artifact: dict[str, Any],
    k: int,
) -> dict[str, list[dict[str, Any]]]:
    sim_to_query = np.asarray(artifact['sim_to_query'])
    sim_matrix = np.asarray(artifact['sim_matrix'])
    mmr_window = artifact.get('mmr_window')
    lambda_values = [float(lam) for lam in artifact.get('lambda_values', [])]

    variants: dict[str, list[dict[str, Any]]] = {
        'top_k': [
            {
                'local_indices': select_indices(
                    strategy='top_k',
                    sim_to_query=sim_to_query,
                    sim_matrix=sim_matrix,
                    k=k,
                    lam=None,
                    mmr_window=mmr_window,
                ),
                'lam': None,
            }
        ]
    }
    for strategy in cast(list[Strategy], ['mmr', 'fac_loc']):
        variants[strategy] = [
            {
                'local_indices': select_indices(
                    strategy=strategy,
                    sim_to_query=sim_to_query,
                    sim_matrix=sim_matrix,
                    k=k,
                    lam=float(lam),
                    mmr_window=mmr_window,
                ),
                'lam': float(lam),
            }
            for lam in lambda_values
        ]
    return variants


def _selection_panel_title(
    artifact: dict[str, Any],
    strategy: str,
    lam: float | None,
    local_indices: Any,
    *,
    lambda_label: float | None = None,
) -> str:
    selected = {int(i) for i in local_indices}
    selected_labels = [artifact['label_ids'][i] for i in selected]
    selected_gold = sum(1 for i in selected if artifact['is_gold'][i])
    selected_facets = len({x for x in selected_labels if x.startswith(artifact['query_id'])})
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


def _plot_query_map_ax(
    ax: Any, artifact: dict[str, Any], *, include_title_prefix: bool = True
) -> None:
    palette = label_palette(artifact['labels'])
    for label in sorted(set(artifact['labels'])):
        idx = [i for i, value in enumerate(artifact['labels']) if value == label]
        marker = _label_marker(label)
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
            artifact['coords'][idx, 0],
            artifact['coords'][idx, 1],
            **scatter_kwargs,
        )
    draw_query_marker(ax, artifact)
    title = f'Candidate-pool map ({artifact["reduction_method"]}, n={len(artifact["candidate_chunk_ids"])})'
    ax.set_title(_axis_title(artifact, title, include_title_prefix=include_title_prefix))
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    _apply_embedding_limits(ax, artifact)
    ax.grid(alpha=0.18)


def _plot_query_similarity_map_ax(
    ax: Any,
    artifact: dict[str, Any],
    *,
    include_title_prefix: bool = True,
) -> Any:
    order = np.arange(len(artifact['candidate_chunk_ids']))
    coords = artifact['coords'][order]
    sim_to_query = artifact['sim_to_query'][order]
    is_gold = np.array([artifact['is_gold'][i] for i in order], dtype=bool)

    vmin = float(sim_to_query.min()) if len(sim_to_query) else 0.0
    vmax = float(sim_to_query.max()) if len(sim_to_query) else 1.0
    points = None
    if is_gold.any():
        points = ax.scatter(
            coords[is_gold, 0],
            coords[is_gold, 1],
            c=sim_to_query[is_gold],
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            s=44,
            alpha=0.88,
            marker='o',
            edgecolors='none',
        )
    if (~is_gold).any():
        points = ax.scatter(
            coords[~is_gold, 0],
            coords[~is_gold, 1],
            c=sim_to_query[~is_gold],
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            s=38,
            alpha=0.92,
            marker='x',
            linewidths=1.25,
        )
    if points is None:
        points = ax.scatter([], [], c=[], cmap='viridis', vmin=vmin, vmax=vmax)
    draw_query_marker(ax, artifact)
    ax.set_title(
        _axis_title(
            artifact,
            f'Map colored by query cosine (n={len(order)})',
            include_title_prefix=include_title_prefix,
        )
    )
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    _apply_embedding_limits(ax, artifact)
    ax.grid(alpha=0.18)
    return points


def _plot_query_rank_ax(
    ax: Any, artifact: dict[str, Any], *, include_title_prefix: bool = True
) -> None:
    ranks = np.arange(1, len(artifact['sim_to_query']) + 1)
    palette = label_palette(artifact['labels'])
    for label in sorted(set(artifact['labels'])):
        idx = [i for i, value in enumerate(artifact['labels']) if value == label]
        marker = _label_marker(label)
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
            artifact['sim_to_query'][idx],
            **scatter_kwargs,
        )
    ax.axvline(artifact['k'], color='black', lw=1.0, ls='--', alpha=0.7, label=f'k={artifact["k"]}')
    ax.set_xlabel('rank by query cosine')
    ax.set_ylabel('query cosine similarity')
    ax.set_title(
        _axis_title(artifact, 'Query-similarity ranking', include_title_prefix=include_title_prefix)
    )
    ax.grid(axis='y', alpha=0.25)


def _plot_discovered_clusters_ax(
    ax: Any,
    artifact: dict[str, Any],
    *,
    include_title_prefix: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    coords = artifact['coords']
    cluster_labels = artifact['cluster_labels']
    unique = sorted(set(int(x) for x in cluster_labels))
    cmap = plt.get_cmap('tab20')  # type: ignore
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
    draw_query_marker(ax, artifact)
    ax.set_title(
        _axis_title(artifact, 'HDBSCAN clusters', include_title_prefix=include_title_prefix)
    )
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    _apply_embedding_limits(ax, artifact)
    ax.grid(alpha=0.18)


def label_palette(labels: list[str]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    unique = [label for label in sorted(set(labels)) if label not in _FIXED_LABEL_COLORS]
    cmap = plt.get_cmap('tab20')  # type: ignore
    palette = {label: cmap(i % 20) for i, label in enumerate(unique)}
    palette.update({
        label: color for label, color in _FIXED_LABEL_COLORS.items() if label in labels
    })
    return palette


def _label_marker(label: str) -> str:
    return 'x' if label in _DISTRACTOR_LABELS else 'o'


def draw_query_marker(ax: Any, artifact: dict[str, Any]) -> None:
    ax.scatter(
        [artifact['query_coord'][0]],
        [artifact['query_coord'][1]],
        marker='*',
        s=220,
        color='black',
        edgecolors='white',
        linewidths=0.8,
        label='query',
        zorder=6,
    )


def artifact_title_prefix(artifact: dict[str, Any]) -> str:
    condition = artifact['query'].get('condition_display') or artifact['query'].get('condition_id')
    return f'{artifact["query_id"]} - {condition}'


def _axis_title(artifact: dict[str, Any], title: str, *, include_title_prefix: bool) -> str:
    if include_title_prefix:
        return f'{artifact_title_prefix(artifact)}\n{title}'
    return title


def _apply_embedding_limits(ax: Any, artifact: dict[str, Any]) -> None:
    coords = np.vstack([artifact['coords'], artifact['query_coord'][None, :]])
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    x_pad = max(float(x_max - x_min) * 0.05, 0.5)
    y_pad = max(float(y_max - y_min) * 0.05, 0.5)
    ax.set_xlim(float(x_min) - x_pad, float(x_max) + x_pad)
    ax.set_ylim(float(y_min) - y_pad, float(y_max) + y_pad)


def strategy_title(strategy: str, lam: float | None) -> str:
    if strategy == 'top_k':
        return 'top-k'
    label = 'MMR' if strategy == 'mmr' else 'FacLoc'
    return f'{label} lambda={lam:.2f}' if lam is not None else label
