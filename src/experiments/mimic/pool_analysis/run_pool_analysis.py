import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from tqdm import tqdm

from experiments.mimic.global_configs import MimicPaths, global_cfg, setup_logging
from experiments.mimic.pool_analysis.schemas_pool_analysis import PoolAnalysisCfg
from experiments.mimic.utils.chunk_pools import ChunkPoolBuilder
from experiments.mimic.utils.utils import load_filtered_queries

from .aggregate import aggregate_stats
from .cluster import alignment, cluster_summary, hdbscan_cluster
from .dim_reduce import umap_2d, umap_reduce
from .metrics_facet import facet_pairwise_cos, facet_silhouette, lda_cv_acc, logreg_cv_acc
from .metrics_geom import cosine_stats, intrinsic_dim, query_cosine_stats
from .outliers import lof_scores, lof_summary
from .plots import plot_aggregate, plot_per_query_card
from .pool_loader import QueryPool, iter_query_pools

pool_analysis_cfg = PoolAnalysisCfg.load()


@dataclass
class PerQueryResult:
    stats: dict[str, Any]
    points: pl.DataFrame


def run_pool_analysis(cfg: PoolAnalysisCfg | None = None) -> None:
    global pool_analysis_cfg
    if cfg is not None:
        pool_analysis_cfg = cfg

    embedding_model = global_cfg.embedding_model
    pool_builder = ChunkPoolBuilder(model_name=embedding_model)

    fig_per = MimicPaths.figures_dir / 'pool_analysis' / 'per_query'
    fig_agg = MimicPaths.figures_dir / 'pool_analysis' / 'aggregate'
    for d in (fig_per, fig_agg):
        d.mkdir(parents=True, exist_ok=True)

    ckpt_rows = MimicPaths.experiment_dir / 'checkpoint_rows.jsonl'
    ckpt_points = MimicPaths.experiment_dir / 'checkpoint_points.parquet'
    ckpt_meta = MimicPaths.experiment_dir / 'checkpoint_meta.json'

    rows: list[dict] = []
    point_frames: list[pl.DataFrame] = []
    fig_count_per_stratum: dict[int, int] = {}
    processed_ids: set[int] = set()

    if ckpt_rows.exists() and ckpt_points.exists() and ckpt_meta.exists():
        print('[checkpoint] Resuming from checkpoint …')
        rows = pl.read_ndjson(ckpt_rows).to_dicts()
        point_frames = [pl.read_parquet(ckpt_points)]
        processed_ids = {int(r['query_id']) for r in rows}
        meta = json.loads(ckpt_meta.read_text())
        fig_count_per_stratum = {int(k): v for k, v in meta['fig_count_per_stratum'].items()}
        print(f'[checkpoint] {len(processed_ids)} queries already done, resuming …')

    new_since_ckpt = 0
    print(
        f'\n[1/4] Analyzing pools ({cfg.pool_n} chunks/query, output → {MimicPaths.experiment_dir})'
    )

    queries_filtered_df = load_filtered_queries(embedding_model)
    if pool_analysis_cfg.limit is not None:
        queries_filtered_df = queries_filtered_df.head(pool_analysis_cfg.limit)

    print(f'Loaded {len(queries_filtered_df):,} queries')

    for qp in tqdm(
        iter_query_pools(queries_filtered_df, pool_builder, pool_analysis_cfg),
        desc='Pool analysis',
        dynamic_ncols=True,
        total=len(queries_filtered_df),
    ):
        if qp.query_id in processed_ids:
            continue

        try:
            r = analyze_query(qp, pool_analysis_cfg)
        except Exception as exc:
            print(f'  [skip] query_id={qp.query_id} ({qp.icd10_3char}): {exc}')
            continue

        rows.append(r.stats)
        point_frames.append(r.points)
        processed_ids.add(qp.query_id)
        new_since_ckpt += 1

        stratum = qp.stratum
        stratum_count = fig_count_per_stratum.get(qp.stratum, 0)
        if stratum_count < pool_analysis_cfg.n_figures:
            fig_path = fig_per / f'{stratum}_q{qp.query_id:04d}_{qp.icd10_3char}.png'
            plot_per_query_card(r.points, r.stats, fig_path)
            fig_count_per_stratum[stratum] = stratum_count + 1
            total_figs = sum(fig_count_per_stratum.values())
            print(
                f'  [fig stratum={stratum} {stratum_count + 1}/{pool_analysis_cfg.n_figures}] {fig_path.name}  (total={total_figs})'
            )

        if new_since_ckpt >= pool_analysis_cfg.commit_every:
            _write_checkpoint(
                rows, point_frames, fig_count_per_stratum, ckpt_rows, ckpt_points, ckpt_meta
            )
            print(f'  [checkpoint] saved ({len(processed_ids)} queries total)')
            new_since_ckpt = 0

        time.sleep(2)

    if new_since_ckpt:
        _write_checkpoint(
            rows, point_frames, fig_count_per_stratum, ckpt_rows, ckpt_points, ckpt_meta
        )
        print(f'  [checkpoint] final save ({len(processed_ids)} queries total)')

    if not rows:
        print('No queries analyzed')
        return

    print(f'\n[2/4] Writing per-query stats ({len(rows):,} rows)')
    stats_df = pl.DataFrame(rows)
    stats_path = MimicPaths.experiment_dir / 'per_query_stats.parquet'
    stats_df.write_parquet(stats_path)

    points_df = pl.concat(point_frames, how='diagonal_relaxed')
    pts_path = MimicPaths.experiment_dir / 'pool_points.parquet'
    points_df.write_parquet(pts_path)

    print('\n[3/4] Aggregating stats')
    agg = aggregate_stats(stats_df)
    agg_path = MimicPaths.experiment_dir / 'aggregate_stats.parquet'
    agg.write_parquet(agg_path)

    print('\n[4/4] Plotting aggregate figures')
    plot_aggregate(stats_df, fig_agg)

    for ckpt_file in (ckpt_rows, ckpt_points, ckpt_meta):
        ckpt_file.unlink(missing_ok=True)
    print('\n[checkpoint] cleaned up')


def analyze_query(qp: QueryPool, cfg: PoolAnalysisCfg) -> PerQueryResult:
    tqdm.write(f'  q={qp.query_id} {qp.icd10_3char} pool_size={qp.pool.n} | geom+facet …')
    pool = qp.pool
    vectors = pool.vectors
    sim_matrix = pool.sim_matrix()
    sim_to_query = pool.sim_scores(qp.query_vec)

    geom = {
        **cosine_stats(sim_matrix),
        **query_cosine_stats(sim_to_query),
        **intrinsic_dim(vectors),
    }
    facet = {
        **facet_pairwise_cos(sim_matrix, qp.facet_onehot),
        'facet_silhouette': facet_silhouette(vectors, qp.facet_combined),
        'facet_lda_acc_cv': lda_cv_acc(vectors, qp.facet_combined, cfg.cv_n_splits),
        'facet_logreg_acc_cv': logreg_cv_acc(vectors, qp.facet_combined, cfg.cv_n_splits),
    }

    tqdm.write(f'  q={qp.query_id} | umap {cfg.umap_dim_for_cluster}D (cluster) …')
    umap_emb = umap_reduce(
        vectors,
        n_components=cfg.umap_dim_for_cluster,
        n_neighbors=cfg.umap_n_neighbors,
        metric=cfg.umap_metric,
    )
    tqdm.write(f'  q={qp.query_id} | umap 2D (plot) …')
    umap_plot = umap_2d(
        vectors,
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric=cfg.umap_metric,
    )

    tqdm.write(f'  q={qp.query_id} | hdbscan + lof …')
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

    points = pl.DataFrame({
        'query_id': [qp.query_id] * pool.n,
        'chunk_id': pool.chunk_ids,
        'hadm_id': pool.hadm_ids.tolist(),
        'section_name': pool.section_names,
        'facet_combined': qp.facet_combined.tolist(),
        'cos_to_query': sim_to_query.tolist(),
        'umap_x': umap_plot[:, 0].tolist(),
        'umap_y': umap_plot[:, 1].tolist(),
        'hdbscan_cluster': hdb_labels.tolist(),
        'lof': scores.tolist(),
    })

    return PerQueryResult(stats=stats, points=points)


def _write_checkpoint(
    rows: list[dict],
    point_frames: list[pl.DataFrame],
    fig_count_per_stratum: dict[int, int],
    ckpt_rows: Path,
    ckpt_points: Path,
    ckpt_meta: Path,
) -> None:
    pl.DataFrame(rows).write_ndjson(ckpt_rows)
    pl.concat(point_frames, how='diagonal_relaxed').write_parquet(ckpt_points)
    ckpt_meta.write_text(json.dumps({'fig_count_per_stratum': fig_count_per_stratum}))


if __name__ == '__main__':
    setup_logging()
    from experiments.mimic.global_configs import load_config_from_main

    raw = load_config_from_main(key='pool_analysis')
    run_pool_analysis(cfg=PoolAnalysisCfg(**raw))
