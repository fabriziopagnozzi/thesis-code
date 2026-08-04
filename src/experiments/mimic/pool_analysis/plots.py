from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from experiments.mimic.global_configs import MimicPaths, get_pool_analysis_path
from experiments.mimic.pool_analysis.embedding_geometry import render_embedding_geometry_figures
from experiments.mimic.pool_analysis.schemas_pool_analysis import PoolAnalysisCfg
from experiments.mimic.utils.chunk_pools import ChunkPoolBuilder


def _safe(d: dict[str, Any], k: str, default: Any = float('nan')) -> Any:
    v = d.get(k, default)
    return default if v is None else v


def plot_per_query_card(points_df: pl.DataFrame, stats: dict[str, Any], out_path: Path) -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    x = points_df['umap_x'].to_numpy()
    y = points_df['umap_y'].to_numpy()

    # 1. HDBSCAN clusters
    ax = axes[0, 0]
    clusters = points_df['hdbscan_cluster'].to_numpy()
    is_out = clusters == -1
    if (~is_out).any():
        ax.scatter(x[~is_out], y[~is_out], c=clusters[~is_out], cmap='tab20', s=8, alpha=0.75)
    if is_out.any():
        ax.scatter(x[is_out], y[is_out], c='lightgray', s=8, alpha=0.5)
    ax.set_title(
        f'HDBSCAN (n={int(_safe(stats, "n_clusters_hdb", 0))}, '
        f'outliers={float(_safe(stats, "frac_outliers_hdb", 0)):.2f})'
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # 2. Facet labels
    ax = axes[0, 1]
    facets = points_df['facet_combined'].to_numpy()
    base_palette = {'neither': '#cccccc', 'both': '#d62728'}
    others = [f for f in np.unique(facets) if f not in base_palette]
    cmap = matplotlib.colormaps['tab10']  # type: ignore
    palette = {**base_palette, **{f: cmap(i) for i, f in enumerate(others)}}
    for f in np.unique(facets):
        m = facets == f
        ax.scatter(x[m], y[m], c=[palette[f]], s=8, alpha=0.75, label=str(f))
    ax.legend(loc='best', fontsize=7, markerscale=2)
    ax.set_title(
        f'Facet (LDA={float(_safe(stats, "facet_lda_acc_cv")):.2f}, '
        f'NMI={float(_safe(stats, "nmi_cluster_facet_hdb")):.2f}, '
        f'ARI={float(_safe(stats, "ari_cluster_facet_hdb")):.2f})'
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # 3. cos to query
    ax = axes[1, 0]
    cos = points_df['cos_to_query'].to_numpy()
    sc = ax.scatter(x, y, c=cos, cmap='viridis', s=8, alpha=0.75)
    plt.colorbar(sc, ax=ax, fraction=0.04)
    ax.set_title(f'cos(chunk, query) (mean={float(_safe(stats, "mean_cos_to_query")):.2f})')
    ax.set_xticks([])
    ax.set_yticks([])

    # 4. Section name (top-N)
    ax = axes[1, 1]
    sections = points_df['section_name'].to_numpy()
    top_counts = pl.Series(sections).value_counts(sort=True).head(8)
    top = set(top_counts.to_series(0).to_list())
    cmap2 = matplotlib.colormaps['tab10']  # type: ignore
    sec_palette = {s: cmap2(i) for i, s in enumerate(sorted(top))}
    legend_seen: set[str] = set()
    for s in np.unique(sections):
        m = sections == s
        c = sec_palette.get(s, '#dddddd')
        label = s if (s in top and s not in legend_seen) else None
        if label is not None:
            legend_seen.add(s)
        ax.scatter(x[m], y[m], c=[c], s=8, alpha=0.7, label=label)
    ax.legend(loc='best', fontsize=6, markerscale=2)
    ax.set_title('section_name')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle(
        f'q={stats.get("query_id")} icd={stats.get("icd10_3char")} '
        f'stratum={stats.get("stratum")} pool={stats.get("pool_size")} | '
        f'dom={float(_safe(stats, "dom_cluster_frac_hdb")):.2f} '
        f'eff_rank={float(_safe(stats, "effective_rank")):.1f}',
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def _plot_metric_grid(
    stats_df: pl.DataFrame,
    metrics: list[str],
    title: str,
    out_path: Path,
    ncols: int = 3,
) -> None:
    import matplotlib.pyplot as plt

    nrows = -(-len(metrics) // ncols)  # ceil div
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    ax_flat = np.asarray(axes).flatten()
    strata = (
        sorted(stats_df['stratum'].drop_nulls().unique().to_list())
        if 'stratum' in stats_df.columns
        else []
    )
    for ax, m in zip(ax_flat, metrics, strict=False):
        if m not in stats_df.columns:
            ax.set_visible(False)
            continue
        if strata:
            data = [
                stats_df.filter(pl.col('stratum') == s)[m].fill_nan(None).drop_nulls().to_numpy()
                for s in strata
            ]
            ax.boxplot(data, tick_labels=[f'S{int(s)}' for s in strata])
        else:
            ax.hist(stats_df[m].fill_nan(None).drop_nulls().to_numpy(), bins=30)
        ax.set_title(m)
    for ax in ax_flat[len(metrics) :]:
        ax.set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_aggregate(stats_df: pl.DataFrame, out_dir: Path) -> None:
    _plot_metric_grid(
        stats_df,
        metrics=[
            'mean_cos',
            'effective_rank',
            'top1_evr',
            'n_clusters_hdb',
            'dom_cluster_frac_hdb',
            'cluster_size_entropy_hdb',
            'frac_outliers_hdb',
        ],
        title='Pool geometry & cluster structure - by stratum',
        out_path=out_dir / 'metrics_geom_cluster_by_stratum.png',
    )
    _plot_metric_grid(
        stats_df,
        metrics=[
            'intra_minus_cross',
            'facet_silhouette',
            'facet_lda_acc_cv',
            'facet_logreg_acc_cv',
            'nmi_cluster_facet_hdb',
            'ari_cluster_facet_hdb',
        ],
        title='Pool facet separability - by stratum',
        out_path=out_dir / 'metrics_facet_by_stratum.png',
    )


if __name__ == '__main__':
    cfg = PoolAnalysisCfg.load()

    stats_path = get_pool_analysis_path('per_query_stats.parquet')
    points_path = get_pool_analysis_path('pool_points.parquet')
    if not stats_path.exists() or not points_path.exists():
        raise FileNotFoundError(
            'Missing pool-analysis artifacts for '
            f'{MimicPaths.experiment_dir}.\n'
            f'Expected:\n- {stats_path}\n- {points_path}\n'
            'Generate them first with:\n'
            f'  EXP={MimicPaths.exp_name} uv run task mimic_pool_analysis\n'
            'or:\n'
            f'  EXP={MimicPaths.exp_name} uv run python -m '
            'experiments.mimic.pool_analysis.run_pool_analysis'
        )

    stats_df = pl.read_parquet(stats_path)
    points_df = pl.read_parquet(points_path)

    fig_per = MimicPaths.figures_dir / 'pool_analysis' / 'per_query'
    fig_agg = MimicPaths.figures_dir / 'pool_analysis' / 'aggregate'
    fig_per.mkdir(parents=True, exist_ok=True)
    fig_agg.mkdir(parents=True, exist_ok=True)

    points_by_qid = {df['query_id'][0]: df for df in points_df.partition_by('query_id')}

    fig_count_per_stratum: dict[Any, int] = {}
    for row in stats_df.iter_rows(named=True):
        stratum = row.get('stratum')
        if fig_count_per_stratum.get(stratum, 0) >= cfg.n_figures:
            continue
        qid = row['query_id']
        pts = points_by_qid.get(qid)
        if pts is None:
            continue
        fig_path = fig_per / f'{stratum}_q{int(qid):04d}_{row["icd10_3char"]}.png'
        plot_per_query_card(pts, row, fig_path)
        fig_count_per_stratum[stratum] = fig_count_per_stratum.get(stratum, 0) + 1

    print(f'Per-query figures → {fig_per}')

    plot_aggregate(stats_df, fig_agg)
    print(f'Aggregate figures → {fig_agg}')

    from experiments.mimic.global_configs import global_cfg

    render_embedding_geometry_figures(
        cfg=cfg,
        stats_df=stats_df,
        points_df=points_df,
        pool_builder=ChunkPoolBuilder(model_name=global_cfg.embedding_model),
    )
