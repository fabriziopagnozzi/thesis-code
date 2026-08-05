"""Shared retrieval helpers for the synthetic benchmark.

This module exists to keep candidate-pool construction, similarity ranking, and
selection logic reusable across geometry filtering and evaluation. It uses
small index maps and existing helper algorithms for top-k, MMR, and
facility-location selection so the retrieval stages stay consistent.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import polars as pl
from numpy.typing import NDArray

from experiments.medical_dataset_gen.dataset_generation.schemas import ChunkPoolScope
from experiments.medical_dataset_gen.evaluation.schemas import LightweightQrelRecord
from experiments.medical_dataset_gen.retrieval.schemas import (
    ChunkDocumentRecord,
    ChunkMembershipRecord,
    QueryIdToFacetMap,
    QueryIdToQrels,
    QueryRecord,
    RetrievalIndexMaps,
    RetrievalStrategy,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
)
from helpers.metrics import avg_cos, fac_cov_score, jaccard
from helpers.query_algorithms import fac_loc_lazy_greedy, mmr, top_k


def build_index_maps(
    chunk_documents: pl.DataFrame,
    chunk_memberships: pl.DataFrame,
    queries: pl.DataFrame,
    chunk_ids: Sequence[str],
    query_ids: Sequence[str],
) -> RetrievalIndexMaps:
    chunk_id_to_idx = {chunk_id: idx for idx, chunk_id in enumerate(chunk_ids)}
    query_id_to_idx = {query_id: idx for idx, query_id in enumerate(query_ids)}

    # Pydantic records keep the downstream worker maps explicit and protect
    # against silently accepting malformed parquet columns.
    chunk_rows = [ChunkDocumentRecord.model_validate(row) for row in chunk_documents.to_dicts()]
    membership_rows = [
        ChunkMembershipRecord.model_validate(row) for row in chunk_memberships.to_dicts()
    ]
    query_rows = [QueryRecord.model_validate(row) for row in queries.to_dicts()]
    chunk_by_id = {row.chunk_id: row for row in chunk_rows}
    query_by_id = {row.query_id: row for row in query_rows}

    # Memberships define the query-local candidate pool. Keep the first index
    # for a repeated query/chunk pair while retaining the latest metadata row.
    chunks_by_source_query: dict[str, list[int]] = defaultdict(list)
    membership_by_query_chunk: dict[tuple[str, str], ChunkMembershipRecord] = {}
    seen_by_query: dict[str, set[int]] = defaultdict(set)
    for row in membership_rows:
        chunk_id = row.chunk_id
        if chunk_id not in chunk_id_to_idx:
            continue
        query_id = row.query_id
        chunk_idx = chunk_id_to_idx[chunk_id]
        if chunk_idx not in seen_by_query[query_id]:
            chunks_by_source_query[query_id].append(chunk_idx)
            seen_by_query[query_id].add(chunk_idx)
        membership_by_query_chunk[(query_id, chunk_id)] = row

    return {
        'chunk_id_to_idx': chunk_id_to_idx,
        'query_id_to_idx': query_id_to_idx,
        'chunk_by_id': chunk_by_id,
        'membership_by_query_chunk': membership_by_query_chunk,
        'query_by_id': query_by_id,
        'chunks_by_source_query': chunks_by_source_query,
    }


def get_candidate_pool_indices(
    query_id: str,
    chunks_by_source_query: dict[str, list[int]],
) -> NDArray[np.intp]:
    return np.array(chunks_by_source_query.get(query_id, []), dtype=np.intp)


def run_topn_cosine_retrieval(
    candidate_indices: NDArray[np.intp],
    chunk_vectors: NDArray[np.float32],
    query_vector: NDArray[np.float32],
    n: int,
) -> tuple[NDArray[np.intp], NDArray[np.float32]]:
    if len(candidate_indices) == 0:
        return candidate_indices, np.array([], dtype=np.float32)

    sims = chunk_vectors[candidate_indices] @ query_vector
    order = np.argsort(sims)[::-1][: min(n, len(sims))]
    return candidate_indices[order], sims[order].astype(np.float32)


def compute_retrieval_diagnostics(
    selected_local_indices: NDArray[np.intp],
    sim_to_query: NDArray[np.float32],
    sim_matrix: NDArray[np.float32],
    topk_local_indices: NDArray[np.intp] | None = None,
) -> dict[str, float]:
    diagnostics = {
        'fac_cov_score': fac_cov_score(selected_local_indices, sim_matrix),
        'avg_cos': avg_cos(selected_local_indices, sim_to_query),
    }
    if topk_local_indices is not None:
        diagnostics['jaccard_vs_topk'] = jaccard(selected_local_indices, topk_local_indices)
    else:
        diagnostics['jaccard_vs_topk'] = 1.0
    return diagnostics


def select_indices(
    strategy: RetrievalStrategy,
    sim_to_query: NDArray[np.float32],
    sim_matrix: NDArray[np.float32],
    k: int,
    lam: float | None = 0.5,
    mmr_window: int | None = None,
) -> NDArray[np.intp]:
    if strategy == 'top_k':
        return top_k(sim_to_query=sim_to_query, k=k)

    if lam is None:
        raise ValueError(f'{strategy} retrieval requires a lambda value')

    if strategy == 'mmr':
        return mmr(
            sim_to_query=sim_to_query,
            sim_matrix=sim_matrix,
            k=k,
            lam=lam,
            window=mmr_window,
        )
    if strategy == 'fac_loc':
        return fac_loc_lazy_greedy(
            sim_to_query=sim_to_query,
            sim_matrix=sim_matrix,
            k=k,
            lam=lam,
        )

    raise ValueError(f'Unsupported strategy: {strategy}')


# Qrel and facet mappings.
def is_query_gold(query_qrels: QueryIdToQrels, chunk_id: str) -> bool:
    row = query_qrels.get(chunk_id)
    return row is not None and row.is_gold


def get_qrels_by_query_chunk(qrels: pl.DataFrame) -> dict[str, dict[str, LightweightQrelRecord]]:
    result: dict[str, dict[str, LightweightQrelRecord]] = defaultdict(dict)
    qrel_columns = set(qrels.columns)
    cluster_id_expr = (
        pl.col('cluster_id').cast(pl.String)
        if 'cluster_id' in qrel_columns
        else pl.lit(None, dtype=pl.String).alias('cluster_id')
    )

    for row in qrels.select(
        'query_id',
        'chunk_id',
        'facet_id',
        cluster_id_expr,
        'cluster_role',
        'axis',
        'is_gold',
    ).iter_rows(named=False):
        query_id, chunk_id, facet_id, cluster_id, cluster_role, axis, is_gold = row
        result[str(query_id)][str(chunk_id)] = LightweightQrelRecord(
            facet_id=None if facet_id is None else str(facet_id),
            cluster_id=None if cluster_id is None else str(cluster_id),
            cluster_role=None if cluster_role is None else str(cluster_role),
            axis=None if axis is None else str(axis),
            is_gold=bool(is_gold),
        )

    return result


def build_query_to_facet_gold_map(qrels: pl.DataFrame) -> QueryIdToFacetMap:
    result: QueryIdToFacetMap = defaultdict(lambda: defaultdict(list))

    for query_id, chunk_id, facet_id in (
        qrels.filter(pl.col('is_gold'))
        .select(
            'query_id',
            'chunk_id',
            'facet_id',
        )
        .iter_rows(named=False)
    ):
        if facet_id is None:
            raise ValueError(f'gold qrel {chunk_id!r} has no facet_id')
        result[str(query_id)][str(facet_id)].append(str(chunk_id))

    return result


# Metric helpers.
def ci_half_width(values: list[float], z: float = 1.96) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return float('nan')
    return z * float(arr.std(ddof=1)) / float(np.sqrt(len(arr)))


def harmonic_mean(left: float, right: float) -> float:
    denom = left + right
    return 0.0 if denom <= 0 else 2 * left * right / denom


# Artifact and provenance checks.
def assert_pool_scope_match(
    df: pl.DataFrame,
    expected_pool_scope: ChunkPoolScope,
    table_name: str,
) -> None:
    if 'pool_scope' not in df.columns or df.is_empty():
        return
    scopes = sorted({str(value) for value in df['pool_scope'].drop_nulls().to_list()})
    if not scopes:
        return
    if scopes != [expected_pool_scope]:
        raise ValueError(
            f'{table_name} was generated with pool_scope={scopes}, '
            f'but the current config expects pool_scope={expected_pool_scope!r}. '
            'Rerun from the geometry stage, or use a config matching the stored artifacts.'
        )


def load_embedding_arrays(
    paths: MedicalDatasetGenPaths,
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[str], list[str]]:
    chunk_vectors = np.load(paths.embeddings_paths('chunk_vectors'), mmap_mode='r')
    query_vectors = np.load(paths.embeddings_paths('query_vectors'), mmap_mode='r')
    chunk_ids = [
        str(value) for value in np.load(paths.embeddings_paths('chunk_ids'), mmap_mode='r')
    ]
    query_ids = [
        str(value) for value in np.load(paths.embeddings_paths('query_ids'), mmap_mode='r')
    ]
    return chunk_vectors, query_vectors, chunk_ids, query_ids


def load_embedding_arrays_mmap_ids(
    paths: MedicalDatasetGenPaths,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.str_], NDArray[np.str_]]:
    chunk_vectors = np.load(paths.embeddings_paths('chunk_vectors'), mmap_mode='r')
    query_vectors = np.load(paths.embeddings_paths('query_vectors'), mmap_mode='r')
    chunk_ids = np.load(paths.embeddings_paths('chunk_ids'), mmap_mode='r')
    query_ids = np.load(paths.embeddings_paths('query_ids'), mmap_mode='r')
    return chunk_vectors, query_vectors, chunk_ids, query_ids
