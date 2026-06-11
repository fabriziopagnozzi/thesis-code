"""Run embedding-geometry analysis and persist its outputs.

This module exists to connect the geometry diagnostics with the experiment
artifacts that downstream plots and evaluation checks consume. It uses the same
parquet-backed tables as the rest of the pipeline so the analysis remains
reproducible and easy to inspect.
"""

from typing import Any

import polars as pl
from tqdm import tqdm

from experiments.medical_dataset_gen.embedding_geometry.artifacts import (
    build_query_artifact,
    choose_query_ids,
)
from experiments.medical_dataset_gen.embedding_geometry.diagnostics import (
    point_rows,
    query_stats,
)
from experiments.medical_dataset_gen.embedding_geometry.plots import (
    plot_cluster_quality_overview,
    plot_full_strategy_selection_overlay,
    plot_query_overview_4panel,
    plot_strategy_overlay,
)
from experiments.medical_dataset_gen.evaluation.evaluate import _assert_pool_scope_match
from experiments.medical_dataset_gen.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
    dump_effective_config,
    load_config_from_cli,
    paths_for,
    read_parquet,
    setup_logging,
    write_parquet,
)
from experiments.medical_dataset_gen.retrieval.embed import load_embedding_arrays
from experiments.medical_dataset_gen.retrieval.utils import build_index_maps


def run_embedding_geometry(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    required_paths = [
        paths.table_path('chunks'),
        paths.table_path('queries'),
        paths.table_path('qrels'),
        paths.embeddings_npz_path,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        print(f'[embedding_geometry] skipping; missing required artifacts: {missing}')
        return pl.DataFrame()

    chunks = read_parquet(paths, 'chunks')
    queries = read_parquet(paths, 'queries')
    qrels = read_parquet(paths, 'qrels')
    geometry = _maybe_read(paths, 'geometry_stats')
    eval_stats = _maybe_read(paths, 'evaluation_stats')
    eval_results = _maybe_read(paths, 'evaluation_results')
    _assert_pool_scope_match(geometry, cfg.retrieval.pool_scope, table_name='geometry_stats')
    _assert_pool_scope_match(eval_results, cfg.retrieval.pool_scope, table_name='evaluation_results')

    chunk_vectors, query_vectors, chunk_ids, query_ids = load_embedding_arrays(paths)
    maps = build_index_maps(chunks, queries, chunk_ids, query_ids)
    selected_query_ids = choose_query_ids(cfg, queries, geometry, eval_results)
    if not selected_query_ids:
        print('[embedding_geometry] no queries selected')
        return pl.DataFrame()

    out_dir = paths.figures_dir / 'embedding_geometry'
    out_dir.mkdir(parents=True, exist_ok=True)

    all_point_rows: list[dict[str, Any]] = []
    all_stat_rows: list[dict[str, Any]] = []

    for qid in tqdm(selected_query_ids, desc='Embedding geometry', dynamic_ncols=True):
        if qid not in maps['query_id_to_idx']:
            continue
        artifact = build_query_artifact(
            cfg=cfg,
            qid=qid,
            queries=queries,
            qrels=qrels,
            chunk_vectors=chunk_vectors,
            query_vectors=query_vectors,
            chunk_ids=chunk_ids,
            maps=maps,
            eval_stats=eval_stats,
        )
        if artifact is None:
            continue

        query_dir = out_dir / qid
        query_dir.mkdir(parents=True, exist_ok=True)
        plot_query_overview_4panel(artifact, query_dir)
        plot_strategy_overlay(artifact, query_dir)
        for k in list(dict.fromkeys(cfg.retrieval.k_values)):
            plot_full_strategy_selection_overlay(artifact, query_dir, k=k)

        all_point_rows.extend(point_rows(artifact))
        all_stat_rows.append(query_stats(artifact))

    points = pl.DataFrame(all_point_rows) if all_point_rows else pl.DataFrame()
    stats = pl.DataFrame(all_stat_rows) if all_stat_rows else pl.DataFrame()
    if len(points):
        write_parquet(paths, 'embedding_geometry_points', points)
    if len(stats):
        write_parquet(paths, 'embedding_geometry_query_stats', stats)
        plot_cluster_quality_overview(stats, out_dir)

    print(f'[embedding_geometry] saved figures to {out_dir}')
    return stats


def _maybe_read(paths: MedicalDatasetGenPaths, table: str) -> pl.DataFrame:
    if paths.table_path(table).exists():  # type: ignore[arg-type]
        return read_parquet(paths, table)  # type: ignore[arg-type]
    return pl.DataFrame()


if __name__ == '__main__':
    cfg = load_config_from_cli()
    paths = paths_for(cfg)
    setup_logging(paths)
    dump_effective_config(cfg, paths)
    run_embedding_geometry(cfg, paths)
