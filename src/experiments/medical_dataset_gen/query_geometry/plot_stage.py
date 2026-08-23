"""Run embedding-geometry analysis and persist its outputs.

This module exists to connect the geometry diagnostics with the experiment
artifacts that downstream plots and evaluation checks consume.
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
from collections.abc import Container, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from tqdm import tqdm

from experiments.medical_dataset_gen.evaluation.eval_worker_handler import (
    load_selected_parquet_columns,
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
    plot_candidate_pool_umap,
    plot_candidate_pool_umap_with_legend,
    plot_cluster_quality_overview,
    plot_full_strategy_selection_overlay,
    plot_pairwise_cosine_heatmap,
    plot_query_cosine_heatmap,
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
from experiments.medical_dataset_gen.query_geometry.schemas import (
    EmbeddingGeometry2DPoint,
    EmbeddingGeometryQueryStats,
    EmbeddingGeometryWorkerState,
    RenderedGeometryResult,
)
from experiments.medical_dataset_gen.retrieval.retrieval_utils import (
    assert_pool_scope_match,
)
from experiments.medical_dataset_gen.utils.cli_parsing import parse_comma_separated_names
from experiments.medical_dataset_gen.utils.global_schemas import (
    ExperimentCfg,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
)
from experiments.medical_dataset_gen.utils.io_utils import write_parquet

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


@dataclass(frozen=True, slots=True)
class GeomPlotRunOptions:
    selected_plots: set[GeomPlotName] | None
    query_ids: tuple[str, ...] | None
    output_dir: Path | None
    umap_neighbors: int | None
    umap_min_dist: float | None


def run_query_geom_plots(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    selected_plots: set[GeomPlotName] | None = None,
    query_ids: Sequence[str] | None = None,
    output_dir: Path | None = None,
    umap_neighbors: int | None = None,
    umap_min_dist: float | None = None,
) -> pl.DataFrame:
    cfg = _with_umap_plot_overrides(
        cfg,
        umap_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist,
    )
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
    explicit_query_ids = _unique_query_ids(query_ids)
    if explicit_query_ids is None:
        selected_query_groups = choose_query_groups(cfg, queries, geometry, eval_results)
    else:
        selected_query_groups = _explicit_query_groups(
            query_ids=explicit_query_ids,
            available_query_ids=set(queries['query_id'].cast(pl.String).to_list()),
            embedded_query_ids=query_id_set,
        )

    selected_query_ids: list[str] = []
    selected_query_group_by_id: dict[str, str] = {}
    selected_query_dir_name_by_id = (
        {query_id: query_id for query_id in explicit_query_ids}
        if explicit_query_ids is not None
        else query_directory_names_for_groups(
            cfg,
            queries,
            geometry,
            eval_results,
            selected_query_groups,
        )
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

    out_dir = (
        paths.figures_dir / 'query_geometry'
        if output_dir is None
        else output_dir.expanduser().resolve() / cfg.global_.output_experiment
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    del queries, geometry, eval_results, query_id_set, selected_query_groups
    gc.collect()

    all_point_rows: list[EmbeddingGeometry2DPoint] = []
    all_stat_rows: list[EmbeddingGeometryQueryStats] = []

    worker_count = query_geometry_worker_count(len(selected_query_ids))
    if worker_count == 1:
        init_query_geometry_worker(
            cfg=cfg,
            paths=paths,
            out_dir=str(out_dir),
            query_group_by_id=selected_query_group_by_id,
            query_dir_name_by_id=selected_query_dir_name_by_id,
            selected_plot_names=selected_plots,
            flat_query_dirs=explicit_query_ids is not None,
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
                paths,
                str(out_dir),
                selected_query_group_by_id,
                selected_query_dir_name_by_id,
                selected_plots,
                explicit_query_ids is not None,
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

    points = _rows_to_dataframe(all_point_rows)
    stats = _rows_to_dataframe(all_stat_rows)
    if len(stats):
        stats = stats.with_columns(
            pl.lit(cfg.query_geometry.umap_metric).alias('umap_metric'),
            pl.lit(cfg.query_geometry.umap_neighbors).alias('umap_neighbors'),
            pl.lit(cfg.query_geometry.umap_min_dist).alias('umap_min_dist'),
            pl.lit(cfg.query_geometry.random_state).alias('umap_random_state'),
        )
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
    query_dir = _query_output_directory(
        out_dir=state['out_dir'],
        query_group=query_group,
        query_dir_name=query_dir_name,
        flat_query_dirs=state['flat_query_dirs'],
    )
    query_dir.mkdir(parents=True, exist_ok=True)
    selected_plot_names: Container[GeomPlotName] | None = state['selected_plot_names']
    if _should_render_plot(selected_plot_names, 'candidate_pool_umap'):
        plot_candidate_pool_umap(artifact, query_dir)
    if _should_render_plot(selected_plot_names, 'candidate_pool_umap_with_legend'):
        plot_candidate_pool_umap_with_legend(artifact, query_dir)
    if _should_render_plot(selected_plot_names, 'pairwise_cosine_heatmap'):
        plot_pairwise_cosine_heatmap(artifact, query_dir)
    if _should_render_plot(selected_plot_names, 'query_cosine_heatmap'):
        plot_query_cosine_heatmap(artifact, query_dir)
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


def _rows_to_dataframe(
    rows: Sequence[EmbeddingGeometry2DPoint | EmbeddingGeometryQueryStats],
) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=None)


def _unique_query_ids(query_ids: Sequence[str] | None) -> tuple[str, ...] | None:
    if query_ids is None:
        return None
    unique_ids = tuple(
        dict.fromkeys(query_id.strip() for query_id in query_ids if query_id.strip())
    )
    if not unique_ids:
        raise ValueError('--query-ids must contain at least one non-empty query ID')
    return unique_ids


def _with_umap_plot_overrides(
    cfg: ExperimentCfg,
    *,
    umap_neighbors: int | None,
    umap_min_dist: float | None,
) -> ExperimentCfg:
    updates: dict[str, int | float] = {}
    if umap_neighbors is not None:
        updates['umap_neighbors'] = umap_neighbors
    if umap_min_dist is not None:
        updates['umap_min_dist'] = umap_min_dist
    if not updates:
        return cfg
    return cfg.model_copy(update={'query_geometry': cfg.query_geometry.model_copy(update=updates)})


def _positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError('value must be greater than zero')
    return value


def _unit_interval_float(raw_value: str) -> float:
    value = float(raw_value)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError('value must be between zero and one')
    return value


def _explicit_query_groups(
    *,
    query_ids: Sequence[str],
    available_query_ids: set[str],
    embedded_query_ids: set[str],
) -> dict[str, list[str]]:
    missing_queries = [query_id for query_id in query_ids if query_id not in available_query_ids]
    missing_embeddings = [query_id for query_id in query_ids if query_id not in embedded_query_ids]
    if missing_queries or missing_embeddings:
        details: list[str] = []
        if missing_queries:
            details.append(f'missing from queries: {", ".join(missing_queries)}')
        if missing_embeddings:
            details.append(f'missing from query embeddings: {", ".join(missing_embeddings)}')
        raise ValueError('invalid explicit geometry query selection (' + '; '.join(details) + ')')
    return {'manual': list(query_ids)}


def _query_output_directory(
    *,
    out_dir: Path,
    query_group: str,
    query_dir_name: str,
    flat_query_dirs: bool,
) -> Path:
    if flat_query_dirs:
        return out_dir / query_dir_name
    return out_dir / query_group / query_dir_name


def parse_plot_names(raw_value: str | None) -> set[GeomPlotName] | None:
    parsed = parse_comma_separated_names(
        raw_value=raw_value,
        valid_names=GEOM_PLOT_FILE_NAMES,
        option_name='--plots',
    )
    if parsed is None:
        return None
    return cast(set[GeomPlotName], set(parsed))


def parse_geom_plots_cli_args(
    argv: list[str],
) -> GeomPlotRunOptions:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--plots',
        type=str,
        help='Comma-separated plot names to generate selectively.',
    )
    parser.add_argument(
        '--query-ids',
        type=str,
        help='Comma-separated query IDs to render instead of configured query selection.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Optional export root; the distribution ID is appended automatically.',
    )
    parser.add_argument(
        '--umap-neighbors',
        type=_positive_int,
        help='Optional UMAP neighborhood size override for this plot run.',
    )
    parser.add_argument(
        '--umap-min-dist',
        type=_unit_interval_float,
        help='Optional UMAP minimum-distance override for this plot run.',
    )
    args, remaining_argv = parser.parse_known_args(argv)
    if remaining_argv:
        parser.error(f'unknown geometry-plot argument(s): {" ".join(remaining_argv)}')
    query_ids = _unique_query_ids(args.query_ids.split(',')) if args.query_ids is not None else None
    return GeomPlotRunOptions(
        selected_plots=parse_plot_names(args.plots),
        query_ids=query_ids,
        output_dir=args.output_dir,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
    )
