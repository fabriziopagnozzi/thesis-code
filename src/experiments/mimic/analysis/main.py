import argparse
from dataclasses import dataclass
from typing import Any

import polars as pl
from tqdm import tqdm

from experiments.mimic.configs import EvaluateCfg, PoolAnalysisCfg, setup_logging
from experiments.mimic.evaluation.candidate_pool import CandidatePoolBuilder
from experiments.mimic.utils.duck_db_init import connect_mimic_duckdb

from .aggregate import aggregate_stats
from .cluster import alignment, cluster_summary, hdbscan_cluster
from .dim_reduce import umap_reduce
from .metrics_facet import facet_pairwise_cos, facet_silhouette, lda_cv_acc, logreg_cv_acc
from .metrics_geom import cosine_stats, intrinsic_dim, knn_density_stats, query_cosine_stats
from .outliers import lof_scores, lof_summary
from .plots import plot_aggregate, plot_per_query_card
from .pool_loader import QueryPool, iter_query_pools


@dataclass
class PerQueryResult:
    stats: dict[str, Any]
    points: pl.DataFrame


def run_pool_analysis() -> None:
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--pool-n', type=int, default=None)
    parser.add_argument('--n-figures', type=int, default=None)
    parser.add_argument(
        '--limit', type=int, default=None, help='Process only the first N queries (debug).'
    )
    args, _ = parser.parse_known_args()

    cfg = PoolAnalysisCfg.load(args.config)

    if args.pool_n is not None:
        cfg.pool_n = args.pool_n
    if args.n_figures is not None:
        cfg.n_figures = args.n_figures

    out_dir = cfg.output_dir
    fig_per = out_dir / 'figures' / 'per_query'
    fig_agg = out_dir / 'figures' / 'aggregate'
    for d in (out_dir, fig_per, fig_agg):
        d.mkdir(parents=True, exist_ok=True)

    eval_cfg = EvaluateCfg.load()
    eval_cfg.embedding_model = cfg.embedding_model

    con = connect_mimic_duckdb()
    builder = CandidatePoolBuilder(con, cfg=eval_cfg)

    rows: list[dict] = []
    point_frames: list[pl.DataFrame] = []
    fig_count = 0

    for qp in tqdm(
        iter_query_pools(builder, cfg, limit=args.limit), desc='Pool analysis', dynamic_ncols=True
    ):
        try:
            r = analyze_query(qp, cfg)
        except Exception as exc:
            print(f'  [skip] query_id={qp.query_id} ({qp.icd10_3char}): {exc}')
            continue

        rows.append(r.stats)
        point_frames.append(r.points)
        if fig_count < cfg.n_figures:
            plot_per_query_card(
                r.points,
                r.stats,
                fig_per / f'q{qp.query_id:04d}_{qp.icd10_3char}.png',
            )
            fig_count += 1

    if not rows:
        print('No queries analyzed')
        return

    stats_df = pl.DataFrame(rows)
    stats_path = out_dir / 'per_query_stats.parquet'
    stats_df.write_parquet(stats_path)
    print(f'Wrote {stats_path}')

    points_df = pl.concat(point_frames, how='diagonal_relaxed')
    pts_path = out_dir / 'pool_points.parquet'
    points_df.write_parquet(pts_path)
    print(f'Wrote {pts_path}')

    agg = aggregate_stats(stats_df)
    agg_path = out_dir / 'aggregate_stats.parquet'
    agg.write_parquet(agg_path)
    print(f'Wrote {agg_path}')

    plot_aggregate(stats_df, fig_agg)
    print(f'Wrote aggregate figures to {fig_agg}')


def analyze_query(qp: QueryPool, cfg: PoolAnalysisCfg) -> PerQueryResult:
    pool = qp.pool
    vectors = pool.vectors
    sim_matrix = pool.sim_matrix()
    sim_to_query = pool.sim_to_query(qp.query_vec)

    geom = {
        **cosine_stats(sim_matrix),
        **query_cosine_stats(sim_to_query),
        **intrinsic_dim(vectors),
        **knn_density_stats(vectors, cfg.knn_k),
    }
    facet = {
        **facet_pairwise_cos(sim_matrix, qp.facet_onehot),
        'facet_silhouette': facet_silhouette(vectors, qp.facet_combined),
        'facet_lda_acc_cv': lda_cv_acc(vectors, qp.facet_combined, cfg.cv_n_splits),
        'facet_logreg_acc_cv': logreg_cv_acc(vectors, qp.facet_combined, cfg.cv_n_splits),
    }

    umap_emb = umap_reduce(
        vectors,
        n_components=cfg.umap_dim_for_cluster,
        n_neighbors=cfg.umap_n_neighbors,
        metric=cfg.umap_metric,
    )

    hdb_labels = hdbscan_cluster(
        umap_emb,
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
    )
    hdb_stats = cluster_summary(hdb_labels, prefix='hdb')
    hdb_align = alignment(hdb_labels, qp.facet_combined, prefix='hdb')

    scores = lof_scores(vectors, cfg.lof_n_neighbors, cfg.lof_contamination)
    lof_stat = lof_summary(scores)

    facet_counts = {
        f'n_{lab}': int(qp.facet_onehot[:, j].sum()) for j, lab in enumerate(qp.modifier_labels)
    }

    stats: dict[str, Any] = {
        'query_id': qp.query_id,
        'icd10_3char': qp.icd10_3char,
        'stratum': qp.stratum,
        'pool_size': pool.n,
        'n_modifiers': len(qp.modifier_labels),
        'modifier_labels': qp.modifier_labels,
        'n_with_facet': int(qp.facet_onehot.any(axis=1).sum()),
        'frac_neither': float((qp.facet_combined == 'neither').mean()),
        **facet_counts,
        **geom,
        **facet,
        **hdb_stats,
        **hdb_align,
        **lof_stat,
    }

    points = pl.DataFrame(
        {
            'query_id': [qp.query_id] * pool.n,
            'chunk_id': pool.chunk_ids,
            'hadm_id': pool.hadm_ids.tolist(),
            'section_name': pool.section_names,
            'facet_combined': qp.facet_combined.tolist(),
            'cos_to_query': sim_to_query.tolist(),
            'umap_x': umap_emb[:, 0].tolist(),
            'umap_y': umap_emb[:, 1].tolist(),
            'hdbscan_cluster': hdb_labels.tolist(),
            'lof': scores.tolist(),
        }
    )

    return PerQueryResult(stats=stats, points=points)


if __name__ == '__main__':
    run_pool_analysis()
