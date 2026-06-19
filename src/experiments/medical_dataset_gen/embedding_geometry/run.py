"""Run embedding-geometry analysis and persist its outputs.

This module exists to connect the geometry diagnostics with the experiment
artifacts that downstream plots and evaluation checks consume.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl
from tqdm import tqdm

from experiments.medical_dataset_gen.embedding_geometry.artifacts import (
    build_query_artifact,
    choose_query_groups,
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
from experiments.medical_dataset_gen.evaluation.utils import assert_pool_scope_match
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

_WORKER_STATE: dict[str, Any] | None = None


def run_embedding_geometry(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    required_paths = [
        paths.table_path('chunk_documents'),
        paths.table_path('chunk_memberships'),
        paths.table_path('queries'),
        paths.table_path('qrels'),
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    has_memmaps = all(
        path.exists()
        for path in [
            paths.embeddings_meta_path,
            paths.embeddings_chunk_vectors_path,
            paths.embeddings_query_vectors_path,
            paths.embeddings_chunk_ids_path,
            paths.embeddings_query_ids_path,
        ]
    )
    has_legacy_npz = paths.embeddings_npz_path.exists()
    if not has_memmaps and not has_legacy_npz:
        missing.append(str(paths.embeddings_npz_path))
    if missing:
        print(f'[embedding_geometry] skipping; missing required artifacts: {missing}')
        return pl.DataFrame()

    chunk_documents = read_parquet(paths, 'chunk_documents')
    chunk_memberships = read_parquet(paths, 'chunk_memberships')
    queries = read_parquet(paths, 'queries')
    geometry = _maybe_read(paths, 'geometry_stats')
    eval_results = _maybe_read(paths, 'evaluation_results')
    assert_pool_scope_match(geometry, cfg.retrieval.pool_scope, table_name='geometry_stats')
    assert_pool_scope_match(eval_results, cfg.retrieval.pool_scope, table_name='evaluation_results')

    _chunk_vectors, _query_vectors, chunk_ids, query_ids = load_embedding_arrays(paths)
    maps = build_index_maps(chunk_documents, chunk_memberships, queries, chunk_ids, query_ids)
    selected_query_groups = choose_query_groups(cfg, queries, geometry, eval_results)
    selected_query_ids: list[str] = []
    selected_query_group_by_id: dict[str, str] = {}
    for group, group_query_ids in selected_query_groups.items():
        for qid in group_query_ids:
            if qid not in maps['query_id_to_idx']:
                continue
            selected_query_ids.append(qid)
            selected_query_group_by_id[qid] = group
    if not selected_query_ids:
        print('[embedding_geometry] no queries selected')
        return pl.DataFrame()

    out_dir = paths.figures_dir / 'embedding_geometry'
    out_dir.mkdir(parents=True, exist_ok=True)

    all_point_rows: list[dict[str, Any]] = []
    all_stat_rows: list[dict[str, Any]] = []

    worker_count = _embedding_geometry_worker_count(len(selected_query_ids))
    if worker_count == 1:
        _init_embedding_geometry_worker(
            cfg_dump=cfg.model_dump(mode='python'),
            exp_name=paths.exp_name,
            out_dir=str(out_dir),
            query_group_by_id=selected_query_group_by_id,
        )
        results = map(_render_embedding_geometry_query, selected_query_ids)
        progress = tqdm(
            results,
            total=len(selected_query_ids),
            desc='Embedding geometry',
            dynamic_ncols=True,
        )
        for result in progress:
            if result is None:
                continue
            all_point_rows.extend(result['point_rows'])
            all_stat_rows.append(result['query_stats'])
    else:
        print(
            f'[embedding_geometry] rendering {len(selected_query_ids):,} queries with {worker_count} workers'
        )
        worker_context = mp.get_context('spawn')
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=worker_context,
            initializer=_init_embedding_geometry_worker,
            initargs=(
                cfg.model_dump(mode='python'),
                paths.exp_name,
                str(out_dir),
                selected_query_group_by_id,
            ),
        ) as executor:
            for result in tqdm(
                executor.map(_render_embedding_geometry_query, selected_query_ids, chunksize=1),
                total=len(selected_query_ids),
                desc='Embedding geometry',
                dynamic_ncols=True,
            ):
                if result is None:
                    continue
                all_point_rows.extend(result['point_rows'])
                all_stat_rows.append(result['query_stats'])

    points = pl.DataFrame(all_point_rows) if all_point_rows else pl.DataFrame()
    stats = pl.DataFrame(all_stat_rows) if all_stat_rows else pl.DataFrame()
    if len(points):
        write_parquet(paths, 'embedding_geometry_points', points)
    if len(stats):
        write_parquet(paths, 'embedding_geometry_query_stats', stats)
        plot_cluster_quality_overview(stats, out_dir)

    print(f'[embedding_geometry] saved figures to {out_dir}')
    return stats


def _embedding_geometry_worker_count(n_queries: int) -> int:
    requested = os.getenv('EMBEDDING_GEOMETRY_WORKERS')
    if requested is not None:
        try:
            workers = int(requested)
        except ValueError as exc:
            raise ValueError('EMBEDDING_GEOMETRY_WORKERS must be an integer') from exc
    else:
        workers = os.cpu_count() or 1
    return max(1, min(n_queries, workers))


def _init_embedding_geometry_worker(
    cfg_dump: dict[str, Any],
    exp_name: str,
    out_dir: str,
    query_group_by_id: dict[str, str],
) -> None:
    os.environ.setdefault('MPLBACKEND', 'Agg')
    cfg = ExperimentCfg.model_validate(cfg_dump)
    paths = MedicalDatasetGenPaths(exp_name)

    chunk_documents = read_parquet(paths, 'chunk_documents')
    chunk_memberships = read_parquet(paths, 'chunk_memberships')
    queries = read_parquet(paths, 'queries')
    qrels = read_parquet(paths, 'qrels')
    eval_stats = _maybe_read(paths, 'evaluation_stats')
    eval_results = _maybe_read(paths, 'evaluation_results')
    chunk_vectors, query_vectors, chunk_ids, query_ids = load_embedding_arrays(paths)
    maps = build_index_maps(chunk_documents, chunk_memberships, queries, chunk_ids, query_ids)

    global _WORKER_STATE
    _WORKER_STATE = {
        'cfg': cfg,
        'queries': queries,
        'qrels': qrels,
        'chunk_vectors': chunk_vectors,
        'query_vectors': query_vectors,
        'chunk_ids': chunk_ids,
        'maps': maps,
        'eval_stats': eval_stats,
        'eval_results': eval_results,
        'out_dir': Path(out_dir),
        'query_group_by_id': query_group_by_id,
        'k_values': list(dict.fromkeys(cfg.retrieval.k_values)),
    }


def _render_embedding_geometry_query(qid: str) -> dict[str, Any] | None:
    if _WORKER_STATE is None:
        raise RuntimeError('embedding geometry worker was not initialized')
    state = _WORKER_STATE
    if qid not in state['maps']['query_id_to_idx']:
        return None

    artifact = build_query_artifact(
        cfg=state['cfg'],
        qid=qid,
        queries=state['queries'],
        qrels=state['qrels'],
        chunk_vectors=state['chunk_vectors'],
        query_vectors=state['query_vectors'],
        chunk_ids=state['chunk_ids'],
        maps=state['maps'],
        eval_stats=state['eval_stats'],
        eval_results=state['eval_results'],
    )
    if artifact is None:
        return None

    query_group = state['query_group_by_id'].get(qid, 'manual')
    artifact['selection_group'] = query_group
    query_dir = state['out_dir'] / query_group / qid
    query_dir.mkdir(parents=True, exist_ok=True)
    plot_query_overview_4panel(artifact, query_dir)
    plot_strategy_overlay(artifact, query_dir)
    for k in state['k_values']:
        plot_full_strategy_selection_overlay(artifact, query_dir, k=k)

    return {
        'point_rows': point_rows(artifact),
        'query_stats': query_stats(artifact),
    }


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
