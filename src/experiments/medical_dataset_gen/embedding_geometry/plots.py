from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


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
    fig, axes = plt.subplots(1, len(strategies), figsize=(5.3 * len(strategies), 5.2), sharex=True, sharey=True)
    if len(strategies) == 1:
        axes = [axes]

    palette = label_palette(artifact['labels'])
    coords = artifact['coords']
    for ax, strategy in zip(axes, strategies, strict=True):
        ax.scatter(coords[:, 0], coords[:, 1], color='#dddddd', s=24, alpha=0.55, edgecolors='none')
        selected = set(int(i) for i in artifact['selections'][strategy]['local_indices'])
        for label in sorted(set(artifact['labels'])):
            idx = [i for i, value in enumerate(artifact['labels']) if value == label and i in selected]
            if not idx:
                continue
            ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                s=74,
                color=palette[label],
                edgecolors='black',
                linewidths=0.65,
                label=label,
                zorder=4,
            )
        draw_query_marker(ax, artifact)
        selected_labels = [artifact['label_ids'][i] for i in selected]
        selected_gold = sum(1 for i in selected if artifact['is_gold'][i])
        ax.set_title(
            f'{strategy_title(strategy, artifact["selections"][strategy]["lam"])}\n'
            f'facets={len({x for x in selected_labels if x.startswith(artifact["query_id"])})}, '
            f'gold={selected_gold}/{len(selected)}'
        )
        _apply_embedding_limits(ax, artifact)
        ax.grid(alpha=0.18)
    axes[-1].legend(fontsize=7, frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f'{artifact["query_id"]}: selected chunks over same embedding coordinates', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / 'strategy_selection_overlay.png', dpi=150, bbox_inches='tight')
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

    _plot_query_map_ax(ax_map, artifact)
    ax_map.legend(fontsize=6, frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))

    points = _plot_query_similarity_map_ax(ax_cos, artifact)
    fig.colorbar(points, ax=ax_cos, fraction=0.046, pad=0.04, label='query cosine')

    _plot_query_rank_ax(ax_rank, artifact)
    ax_rank.legend(fontsize=6, frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))

    _plot_discovered_clusters_ax(ax_cluster, artifact)
    ax_cluster.legend(fontsize=6, frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))

    fig.suptitle(f'{artifact["query_id"]}: embedding geometry overview', fontsize=14)
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


def _plot_query_map_ax(ax: Any, artifact: dict[str, Any]) -> None:
    palette = label_palette(artifact['labels'])
    for label in sorted(set(artifact['labels'])):
        idx = [i for i, value in enumerate(artifact['labels']) if value == label]
        ax.scatter(
            artifact['coords'][idx, 0],
            artifact['coords'][idx, 1],
            s=34,
            alpha=0.78,
            color=palette[label],
            edgecolors='none',
            label=label,
        )
    draw_query_marker(ax, artifact)
    ax.set_title(
        f'Candidate-pool map ({artifact["reduction_method"]}, '
        f'n={len(artifact["candidate_chunk_ids"])})'
    )
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    _apply_embedding_limits(ax, artifact)
    ax.grid(alpha=0.18)


def _plot_query_similarity_map_ax(ax: Any, artifact: dict[str, Any]) -> Any:
    order = np.arange(len(artifact['candidate_chunk_ids']))
    coords = artifact['coords'][order]
    sim_to_query = artifact['sim_to_query'][order]
    is_gold = np.array([artifact['is_gold'][i] for i in order], dtype=bool)

    vmin = float(sim_to_query.min()) if len(sim_to_query) else 0.0
    vmax = float(sim_to_query.max()) if len(sim_to_query) else 1.0
    points = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=sim_to_query,
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
        s=np.where(is_gold, 44, 28),
        alpha=0.88,
        edgecolors='none',
    )
    draw_query_marker(ax, artifact)
    ax.set_title(f'Map colored by query cosine (n={len(order)})')
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    _apply_embedding_limits(ax, artifact)
    ax.grid(alpha=0.18)
    return points


def _plot_query_rank_ax(ax: Any, artifact: dict[str, Any]) -> None:
    ranks = np.arange(1, len(artifact['sim_to_query']) + 1)
    palette = label_palette(artifact['labels'])
    for label in sorted(set(artifact['labels'])):
        idx = [i for i, value in enumerate(artifact['labels']) if value == label]
        ax.scatter(
            ranks[idx],
            artifact['sim_to_query'][idx],
            color=palette[label],
            s=28,
            alpha=0.78,
            label=label,
            edgecolors='none',
        )
    ax.axvline(artifact['k'], color='black', lw=1.0, ls='--', alpha=0.7, label=f'k={artifact["k"]}')
    ax.set_xlabel('rank by query cosine')
    ax.set_ylabel('query cosine similarity')
    ax.set_title('Query-similarity ranking')
    ax.grid(axis='y', alpha=0.25)


def _plot_discovered_clusters_ax(ax: Any, artifact: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    coords = artifact['coords']
    cluster_labels = artifact['cluster_labels']
    unique = sorted(set(int(x) for x in cluster_labels))
    cmap = plt.get_cmap('tab20')
    for idx, cluster_id in enumerate(unique):
        mask = cluster_labels == cluster_id
        color = '#bbbbbb' if cluster_id == -1 else cmap(idx % 20)
        label = 'noise' if cluster_id == -1 else f'cluster {cluster_id}'
        ax.scatter(coords[mask, 0], coords[mask, 1], s=34, alpha=0.78, color=color, label=label, edgecolors='none')
    draw_query_marker(ax, artifact)
    ax.set_title('HDBSCAN clusters')
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    _apply_embedding_limits(ax, artifact)
    ax.grid(alpha=0.18)


def label_palette(labels: list[str]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    unique = sorted(set(labels))
    cmap = plt.get_cmap('tab20')
    return {label: cmap(i % 20) for i, label in enumerate(unique)}


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
