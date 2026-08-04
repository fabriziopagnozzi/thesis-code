"""Annotate query-local retrieval geometry before evaluation.

This module exists to measure whether the synthetic benchmark produces the
intended query-local geometry, including facet coverage and dominance effects,
before retrieval metrics are computed. It uses nearest-neighbor scoring,
pool-scope filtering, and per-query diagnostic aggregation to label each
query; evaluation retains the full generated population.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import polars as pl
from numpy.typing import NDArray
from tqdm import tqdm

from experiments.medical_dataset_gen.query_geometry.filter_diagnostics import (
    background_outlier_diagnostics,
    facet_separation,
    flatten_topk_diagnostics,
    is_query_near_miss_distractor,
    rank_where_all_facets_first_covered,
    strict_gate_failures,
    topk_diagnostics_by_k,
    topk_dominant_count,
    topk_vs_facloc_diagnostics,
)
from experiments.medical_dataset_gen.query_geometry.filter_diagnostics import (
    competitive_pool_mass as compute_competitive_pool_mass,
)
from experiments.medical_dataset_gen.query_geometry.filter_diagnostics import (
    diagnostic_k_values as compute_diagnostic_k_values,
)
from experiments.medical_dataset_gen.query_geometry.filter_diagnostics import (
    topk_retrieved_facets as compute_topk_retrieved_facets,
)
from experiments.medical_dataset_gen.query_geometry.schemas import (
    GeometryFilterStatsRow,
)
from experiments.medical_dataset_gen.retrieval.retrieval_utils import (
    build_index_maps,
    build_query_to_facet_gold_map,
    get_candidate_pool_indices,
    get_qrels_by_query_chunk,
    load_embedding_arrays,
    run_topn_cosine_retrieval,
)
from experiments.medical_dataset_gen.retrieval.schemas import QueryRecord
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
)
from experiments.medical_dataset_gen.utils.io_utils import (
    json_dumps,
    read_parquet,
    write_json,
    write_parquet,
)


def run_filter_queries(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    chunk_documents = read_parquet(paths, 'chunk_documents')
    chunk_memberships = read_parquet(paths, 'chunk_memberships')
    queries = read_parquet(paths, 'queries')
    qrels = read_parquet(paths, 'qrels')
    chunk_vectors, query_vectors, chunk_ids, query_ids = load_embedding_arrays(paths)
    chunk_vectors = _materialize_embedding_vectors(chunk_vectors)
    query_vectors = _materialize_embedding_vectors(query_vectors)
    maps = build_index_maps(chunk_documents, chunk_memberships, queries, chunk_ids, query_ids)

    facet_gold = build_query_to_facet_gold_map(qrels)
    qrels_by_query_chunk = get_qrels_by_query_chunk(qrels)
    competitive_pool_mass = compute_competitive_pool_mass(cfg)
    stress_horizon_k = cfg.geometry_filter.stress_horizon(
        competitive_pool_mass=competitive_pool_mass
    )
    diagnostic_k_values = compute_diagnostic_k_values(cfg, stress_horizon_k=stress_horizon_k)
    rows: list[dict[str, object]] = []

    for query_row in tqdm(
        queries.iter_rows(named=True), total=len(queries), desc='Geometry', dynamic_ncols=True
    ):
        query = QueryRecord.model_validate(query_row)
        qid = query.query_id
        qidx = maps['query_id_to_idx'][qid]

        candidate_idx = get_candidate_pool_indices(
            query_id=qid,
            chunks_by_source_query=maps['chunks_by_source_query'],
        )
        topn_global, topn_sims = run_topn_cosine_retrieval(
            candidate_indices=candidate_idx,
            chunk_vectors=chunk_vectors,
            query_vector=query_vectors[qidx],
            n=cfg.retrieval.candidate_pool_n,
        )
        topn_chunk_ids = [chunk_ids[int(i)] for i in topn_global]
        topn_set = set(topn_chunk_ids)

        query_facets = facet_gold.get(qid, {})
        query_qrels = qrels_by_query_chunk.get(qid, {})
        if not query_facets:
            continue
        dominant_facet_id = query.dominant_primary_facet_id
        facet_meta = {
            str(facet['facet_id']): (str(facet['subgroup_id']), str(facet['axis']))
            for facet in json.loads(query.facets_json or '[]')
        }
        facets_present = {
            facet_id: bool(topn_set & set(gold_ids)) for facet_id, gold_ids in query_facets.items()
        }
        n_facets_present = sum(facets_present.values())

        topk_by_k = topk_diagnostics_by_k(
            topn_chunk_ids=topn_chunk_ids,
            query_qrels=query_qrels,
            query_facets=query_facets,
            dominant_facet_id=dominant_facet_id,
            primary_axis=query.primary_axis,
            k_values=diagnostic_k_values,
        )
        primary_topk = topk_by_k[stress_horizon_k]
        topk_dominant = topk_dominant_count(
            topn_chunk_ids=topn_chunk_ids,
            query_qrels=query_qrels,
            k=stress_horizon_k,
        )
        topk_retrieved_facets = compute_topk_retrieved_facets(
            topn_chunk_ids=topn_chunk_ids,
            query_qrels=query_qrels,
            k=stress_horizon_k,
        )
        n_topk_retrieved_facets = len(topk_retrieved_facets)
        dominant_primary_topk_count = int(primary_topk['dominant_primary_count'])
        primary_axis_topk_count = int(primary_topk['primary_axis_count'])
        all_facet_rank = rank_where_all_facets_first_covered(
            topn_chunk_ids=topn_chunk_ids,
            query_qrels=query_qrels,
            query_facets=query_facets,
        )
        n_distractors = sum(
            1
            for chunk_id in topn_chunk_ids
            if not ((qrel := query_qrels.get(chunk_id)) and qrel.is_gold)
        )
        n_near_miss_distractors = sum(
            1 for chunk_id in topn_chunk_ids if is_query_near_miss_distractor(query_qrels, chunk_id)
        )
        separation = facet_separation(
            query_facets=query_facets,
            facet_meta=facet_meta,
            chunk_id_to_idx=maps['chunk_id_to_idx'],
            chunk_vectors=chunk_vectors,
        )
        background_diagnostics = background_outlier_diagnostics(
            topn_chunk_ids=topn_chunk_ids,
            topn_sims=topn_sims,
            query_qrels=query_qrels,
            chunk_id_to_idx=maps['chunk_id_to_idx'],
            chunk_vectors=chunk_vectors,
            expected_background_chunks=cfg.generation.chunk_pools.background_outliers_per_query(),
        )

        diagnostics = topk_vs_facloc_diagnostics(
            topn_global=topn_global,
            topn_sims=topn_sims,
            chunk_vectors=chunk_vectors,
            k=stress_horizon_k,
        )

        failures = strict_gate_failures(
            cfg.geometry_filter,
            n_facets_present=n_facets_present,
            n_facets=len(query_facets),
            primary_axis_fraction=float(primary_topk['primary_axis_fraction']),
            n_topk_retrieved_facets=n_topk_retrieved_facets,
            background_outlier_complete=bool(background_diagnostics['background_outlier_complete']),
        )
        passes = not any(failures.values())

        geometry_row_data: dict[str, object] = (
            {
                'query_id': qid,
                'evidence_profile_id': query.evidence_profile_id,
                'pool_id': query.pool_id,
                'condition_id': query.condition_id,
                'cohort_contrast_id': query.cohort_contrast_id,
                'cohort_contrast_family': query.cohort_contrast_family,
                'cohort_dimension_id': query.cohort_dimension_id,
                'template_id': query.template_id,
                'pool_scope': cfg.retrieval.pool_scope,
                'pool_size': len(topn_global),
                'stress_horizon_basis': cfg.geometry_filter.stress_horizon_basis,
                'stress_horizon_competitive_pool_mass': competitive_pool_mass,
                'stress_horizon_k': stress_horizon_k,
                'n_facets': len(query_facets),
                'n_facets_present': n_facets_present,
                'all_facets_present': n_facets_present == len(query_facets),
                'topk_dominant_count': topk_dominant,
                'dominant_primary_facet_id': dominant_facet_id,
                'dominant_primary_topk_count': dominant_primary_topk_count,
                'dominant_primary_topk_fraction': primary_topk['dominant_primary_fraction'],
                'primary_axis': query.primary_axis,
                'secondary_axis': query.secondary_axis,
                'primary_axis_stress_count': primary_axis_topk_count,
                'primary_axis_stress_fraction': primary_topk['primary_axis_fraction'],
                'n_topk_retrieved_facets': n_topk_retrieved_facets,
                'stress_horizon_retrieved_facet_fraction': n_topk_retrieved_facets
                / len(query_facets),
                'max_retrieved_facet_fraction': cfg.geometry_filter.max_retrieved_facet_fraction,
                'rank_where_all_facets_first_covered': all_facet_rank,
                'all_facets_covered_before_stress_horizon': (
                    all_facet_rank is not None and all_facet_rank <= stress_horizon_k
                ),
                'n_distractors_in_pool': n_distractors,
                'n_near_miss_distractors_in_pool': n_near_miss_distractors,
                **separation,
                'passes_filter': passes,
                **failures,
                'facets_present_json': json_dumps(facets_present),
                'topk_retrieved_facets_json': json_dumps(topk_retrieved_facets),
            }
            | flatten_topk_diagnostics(topk_by_k)
            | background_diagnostics
            | diagnostics
        )
        geometry_row = GeometryFilterStatsRow.model_validate(geometry_row_data)
        rows.append(geometry_row.model_dump(mode='python'))

    df = pl.DataFrame(rows)
    write_parquet(paths, 'geometry_stats', df)
    slice_stats = (
        df.group_by(
            'condition_id',
            'cohort_dimension_id',
            'cohort_contrast_family',
            'cohort_contrast_id',
            'primary_axis',
            'secondary_axis',
            'template_id',
            'n_topk_retrieved_facets',
        )
        .agg(
            pl.len().alias('n_queries'),
            pl.col('passes_filter').sum().alias('n_eligible'),
            pl.col('passes_filter').mean().alias('eligible_rate'),
        )
        .sort('condition_id', 'cohort_contrast_id', 'primary_axis', 'secondary_axis')
    )
    write_parquet(paths, 'geometry_slice_stats', slice_stats)
    n_pass = int(df['passes_filter'].sum()) if len(df) else 0
    fail_columns = [column for column in df.columns if column.startswith('fail_')]
    combinations = df.select(fail_columns).to_dicts() if len(df) else []
    failure_combinations = Counter(
        '+'.join(column.removeprefix('fail_') for column, failed in row.items() if failed) or 'pass'
        for row in combinations
    )
    write_json(
        paths,
        'geometry_summary.json',
        {
            'dataset_schema_version': cfg.dataset_schema_version,
            'raw_query_count': len(df),
            'eligible_query_count': n_pass,
            'eligible_target_minimum': 3001,
            'target_met': n_pass > 3000,
            'failure_combinations': dict(sorted(failure_combinations.items())),
        },
    )
    print(f'[geometry] {n_pass:,}/{len(df):,} queries pass')
    return df


def _materialize_embedding_vectors(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    # Geometry filtering does many tiny query-local gathers. Keeping embeddings
    # as np.memmap makes those gathers turn into scattered disk reads on /DATA.
    return np.array(vectors, dtype=np.float32, copy=True)
