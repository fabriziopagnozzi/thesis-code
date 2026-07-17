"""Run embedding-geometry analysis and persist its outputs.

This module exists to connect the geometry diagnostics with the experiment
artifacts that downstream plots and evaluation checks consume.
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import sys
from collections.abc import Container
from concurrent.futures import ProcessPoolExecutor
from typing import cast

import numpy as np
import polars as pl
from tqdm import tqdm

from experiments.medical_dataset_gen.evaluation.eval_worker_handler import (
    load_selected_parquet_columns,
)
from experiments.medical_dataset_gen.evaluation.retrieval_utils import (
    assert_pool_scope_match,
)
from experiments.medical_dataset_gen.query_geometry.artifacts import (
    build_query_artifact,
    choose_query_groups,
    query_directory_names_for_groups,
)
from experiments.medical_dataset_gen.query_geometry.diagnostics import (
    build_geometry_points_row,
    query_stats,
)
from experiments.medical_dataset_gen.query_geometry.geom_plots import (
    plot_cluster_quality_overview,
    plot_full_strategy_selection_overlay,
    plot_query_overview_4panel,
    plot_strategy_overlay,
)
from experiments.medical_dataset_gen.query_geometry.geom_plots_configs import (
    GEOM_PLOT_FILE_NAMES,
    GeomPlotName,
)
from experiments.medical_dataset_gen.query_geometry.geom_worker_handler import (
    get_geom_worker_state,
    init_query_geometry_worker,
    load_selected_parquet_columns_if_exists,
    query_geometry_worker_count,
)
from experiments.medical_dataset_gen.schemas.global_config_schemas import (
    ExperimentCfg,
)
from experiments.medical_dataset_gen.schemas.query_geometry_schemas import (
    EmbeddingGeometry2DPoint,
    EmbeddingGeometryQueryStats,
    EmbeddingGeometryWorkerState,
    RenderedGeometryResult,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    load_config_from_cli,
    paths_for,
)
from experiments.medical_dataset_gen.utils.io_utils import write_parquet
from experiments.medical_dataset_gen.utils.logging_utils import setup_logging

_PARENT_QUERY_COLUMNS = ['query_id']
_PARENT_GEOMETRY_COLUMNS = [
    'query_id',
    'pool_scope',
    'passes_filter',
    'topk_dominant_count',
    'in_minus_cross_similarity',
    'n_distractors_in_pool',
]
_PARENT_EVAL_RESULTS_COLUMNS = [
    'query_id',
    'pool_scope',
    'strategy',
    'k',
    'facet_coverage',
    'gold_precision',
    'distractor_rate',
    'weighted_facet_coverage',
    'alpha_ndcg',
    'lam',
]


def run_query_geom_plots(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    selected_plots: set[GeomPlotName] | None = None,
) -> pl.DataFrame:
    required_paths = [
        paths.table_path('chunk_documents'),
        paths.table_path('chunk_memberships'),
        paths.table_path('queries'),
        paths.table_path('qrels'),
    ]
    missing = [str(path) for path in required_paths if not path.exists()]

    embedding_paths = [
        paths.embeddings_paths('metadata'),
        paths.embeddings_paths('chunk_vectors'),
        paths.embeddings_paths('query_vectors'),
        paths.embeddings_paths('chunk_ids'),
        paths.embeddings_paths('query_ids'),
    ]
    missing.extend(str(path) for path in embedding_paths if not path.exists())
    if missing:
        print(f'[query_geometry] skipping; missing required artifacts: {missing}')
        return pl.DataFrame()

    queries = load_selected_parquet_columns(paths, 'queries', _PARENT_QUERY_COLUMNS)
    geometry = load_selected_parquet_columns_if_exists(
        paths,
        'geometry_stats',
        required_columns=['query_id'],
        optional_columns=[col for col in _PARENT_GEOMETRY_COLUMNS if col != 'query_id'],
    )
    eval_results = load_selected_parquet_columns_if_exists(
        paths,
        'evaluation_results',
        required_columns=['query_id', 'strategy', 'k'],
        optional_columns=[
            col for col in _PARENT_EVAL_RESULTS_COLUMNS if col not in {'query_id', 'strategy', 'k'}
        ],
    )
    assert_pool_scope_match(geometry, cfg.retrieval.pool_scope, table_name='geometry_stats')
    assert_pool_scope_match(eval_results, cfg.retrieval.pool_scope, table_name='evaluation_results')

    query_id_set = {
        str(query_id) for query_id in np.load(paths.embeddings_paths('query_ids'), mmap_mode='r')
    }
    selected_query_groups = choose_query_groups(cfg, queries, geometry, eval_results)

    selected_query_ids: list[str] = []
    selected_query_group_by_id: dict[str, str] = {}
    selected_query_dir_name_by_id = query_directory_names_for_groups(
        cfg,
        queries,
        geometry,
        eval_results,
        selected_query_groups,
    )
    for group, group_query_ids in selected_query_groups.items():
        for qid in group_query_ids:
            if qid not in query_id_set:
                continue
            selected_query_ids.append(qid)
            selected_query_group_by_id[qid] = group
    if not selected_query_ids:
        print('[query_geometry] no queries selected')
        return pl.DataFrame()

    out_dir = paths.figures_dir / 'query_geometry'
    out_dir.mkdir(parents=True, exist_ok=True)

    del queries, geometry, eval_results, query_id_set, selected_query_groups
    gc.collect()

    all_point_rows: list[EmbeddingGeometry2DPoint] = []
    all_stat_rows: list[EmbeddingGeometryQueryStats] = []

    worker_count = query_geometry_worker_count(len(selected_query_ids))
    if worker_count == 1:
        init_query_geometry_worker(
            cfg=cfg,
            exp_name=paths.exp_name,
            out_dir=str(out_dir),
            query_group_by_id=selected_query_group_by_id,
            query_dir_name_by_id=selected_query_dir_name_by_id,
            selected_plot_names=selected_plots,
        )
        results = map(_render_query_geometry_query, selected_query_ids)

        for result in tqdm(
            results,
            total=len(selected_query_ids),
            desc='Embedding geometry',
            dynamic_ncols=True,
        ):
            if result is None:
                continue
            all_point_rows.extend(result['point_rows'])
            all_stat_rows.append(result['query_stats'])
    else:
        print(
            f'[query_geometry] rendering {len(selected_query_ids):,} queries with {worker_count} workers'
        )
        worker_context = mp.get_context('spawn')
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=worker_context,
            initializer=init_query_geometry_worker,
            initargs=(
                cfg,
                paths.exp_name,
                str(out_dir),
                selected_query_group_by_id,
                selected_query_dir_name_by_id,
                selected_plots,
            ),
        ) as executor:
            for result in tqdm(
                executor.map(_render_query_geometry_query, selected_query_ids, chunksize=1),
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
        write_parquet(paths, 'query_geometry_points', points)
    if len(stats):
        write_parquet(paths, 'query_geometry_stats', stats)
        if _should_render_plot(selected_plots, 'cluster_quality_overview'):
            plot_cluster_quality_overview(stats, out_dir)

    selected_plot_names = (
        [plot_name for plot_name in sorted(GEOM_PLOT_FILE_NAMES) if plot_name in selected_plots]
        if selected_plots is not None
        else [plot_name for plot_name in sorted(GEOM_PLOT_FILE_NAMES)]
    )
    selection_note = f' ({", ".join(selected_plot_names)})' if selected_plots is not None else ''
    print(f'[query_geometry] saved figures to {out_dir}{selection_note}')
    return stats


def _render_query_geometry_query(qid: str) -> RenderedGeometryResult | None:
    worker_state: EmbeddingGeometryWorkerState | None = get_geom_worker_state()

    if worker_state is None:
        raise RuntimeError('embedding geometry worker was not initialized')

    state = worker_state
    maps = state['maps']
    if qid not in maps['query_id_to_idx']:
        return None
    query = state['queries_by_id'].get(qid)
    if query is None:
        return None

    artifact = build_query_artifact(
        cfg=state['cfg'],
        qid=qid,
        query=query,
        query_qrels=state['qrels_by_query_chunk'].get(qid, {}),
        chunk_vectors=state['chunk_vectors'],
        query_vectors=state['query_vectors'],
        chunk_ids=state['chunk_ids'],
        maps=state['maps'],
        query_best_lambdas=state['query_best_lambdas'],
        global_best_lambdas=state['global_best_lambdas'],
    )
    if artifact is None:
        return None

    query_group = state['query_group_by_id'].get(qid, 'manual')
    artifact.selection_group = query_group
    query_dir_name = state['query_dir_name_by_id'].get(qid, qid)
    query_dir = state['out_dir'] / query_group / query_dir_name
    query_dir.mkdir(parents=True, exist_ok=True)
    selected_plot_names: Container[str] | None = state['selected_plot_names']
    if _should_render_plot(selected_plot_names, 'query_overview_4panel'):
        plot_query_overview_4panel(artifact, query_dir)
    if _should_render_plot(selected_plot_names, 'strategy_overlay'):
        plot_strategy_overlay(artifact, query_dir)
    if _should_render_plot(selected_plot_names, 'full_strategy_selection_overlay'):
        for k in state['k_values']:
            plot_full_strategy_selection_overlay(
                artifact,
                query_dir,
                k=k,
            )

    return {
        'point_rows': build_geometry_points_row(artifact),
        'query_stats': query_stats(artifact),
    }


def _should_render_plot(
    selected_plot_names: Container[GeomPlotName] | None,
    plot_name: GeomPlotName,
) -> bool:
    return selected_plot_names is None or plot_name in selected_plot_names


def parse_plot_names(raw_value: str | None) -> set[GeomPlotName] | None:
    if raw_value is None:
        return None
    plot_names: set[str] = {part.strip() for part in raw_value.split(',') if part.strip()}
    if not plot_names:
        raise ValueError('--plots was provided but no plot names were specified')

    if plot_names:
        unknown_plots = sorted(plot_names - GEOM_PLOT_FILE_NAMES)
        if unknown_plots:
            available = ', '.join(sorted(GEOM_PLOT_FILE_NAMES))
            unknown = ', '.join(unknown_plots)
            raise ValueError(f'Unknown plot name(s): {unknown}. Available plots: {available}')

    return cast(set[GeomPlotName] | None, plot_names)


def parse_geom_plots_cli_args(
    argv: list[str],
) -> tuple[ExperimentCfg, set[GeomPlotName] | None]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--plots',
        type=str,
        help='Comma-separated plot names to generate selectively.',
    )
    args, remaining_argv = parser.parse_known_args(argv)

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *remaining_argv]
        cfg = load_config_from_cli()
    finally:
        sys.argv = original_argv
    return cfg, parse_plot_names(args.plots)


if __name__ == '__main__':
    cfg, selected_plots = parse_geom_plots_cli_args(sys.argv[1:])
    paths = paths_for(cfg)
    setup_logging(paths)
    run_query_geom_plots(cfg, paths, selected_plots=selected_plots)
