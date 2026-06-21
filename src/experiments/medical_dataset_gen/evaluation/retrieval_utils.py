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

from experiments.medical_dataset_gen.schemas.generation_schemas import ChunkPoolScope
from experiments.medical_dataset_gen.schemas.retrieval_schemas import (
    ChunkDocumentRecord,
    ChunkMembershipRecord,
    QrelRecord,
    QueryIdToFacetMap,
    QueryIdToQrels,
    QueryRecord,
    RetrievalIndexMaps,
    RetrievalStrategy,
)
from experiments.medical_dataset_gen.utils.global_configs import (
    MedicalDatasetGenPaths,
    MethodsComparisonKernelsCfg,
    unreachable_code,
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

    chunk_rows = [ChunkDocumentRecord.model_validate(row) for row in chunk_documents.to_dicts()]
    membership_rows = [
        ChunkMembershipRecord.model_validate(row) for row in chunk_memberships.to_dicts()
    ]
    query_rows = [QueryRecord.model_validate(row) for row in queries.to_dicts()]
    chunk_by_id = {row.chunk_id: row for row in chunk_rows}
    query_by_id = {row.query_id: row for row in query_rows}

    chunks_by_source_query: dict[str, list[int]] = defaultdict(list)
    chunks_by_condition: dict[str, list[int]] = defaultdict(list)
    for row in chunk_rows:
        chunk_id = row.chunk_id
        if chunk_id not in chunk_id_to_idx:
            continue
        chunk_idx = chunk_id_to_idx[chunk_id]
        condition_id = row.condition_id
        if condition_id:
            chunks_by_condition[str(condition_id)].append(chunk_idx)

    membership_by_query_chunk: dict[tuple[str, str], ChunkMembershipRecord] = {}
    seen_by_query: dict[str, set[int]] = defaultdict(set)
    for row in membership_rows:
        chunk_id = row.chunk_id
        if chunk_id not in chunk_id_to_idx:
            continue
        query_id = row.source_query_id
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
        'chunks_by_condition': chunks_by_condition,
    }


def get_candidate_pool_indices(
    query_id: str,
    pool_scope: ChunkPoolScope,
    n_chunks: int,
    chunks_by_source_query: dict[str, list[int]],
    chunks_by_condition: dict[str, list[int]],
    query_condition_id: str | None,
) -> NDArray[np.intp]:
    if pool_scope == 'full_corpus':
        return np.arange(n_chunks, dtype=np.intp)
    if pool_scope == 'same_condition':
        if not query_condition_id:
            return np.array([], dtype=np.intp)

        return np.array(chunks_by_condition.get(query_condition_id, []), dtype=np.intp)
    if pool_scope == 'query_local':
        return np.array(chunks_by_source_query.get(query_id, []), dtype=np.intp)
    else:
        raise RuntimeError('Unexpected pool_scope')


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
    lam: float | None,
    mmr_window: int | None = None,
) -> NDArray[np.intp]:
    if strategy == 'top_k':
        return top_k(sim_to_query=sim_to_query, k=k)
    if strategy == 'mmr':
        return mmr(
            sim_to_query=sim_to_query,
            sim_matrix=sim_matrix,
            k=k,
            lam=0.5 if lam is None else lam,
            window=mmr_window,
        )
    if strategy == 'fac_loc':
        return fac_loc_lazy_greedy(
            sim_to_query=sim_to_query,
            sim_matrix=sim_matrix,
            k=k,
            lam=0.5 if lam is None else lam,
        )
    raise ValueError(f'Unsupported strategy: {strategy}')


# Qrels utils
def is_query_gold(query_qrels: QueryIdToQrels, chunk_id: str) -> bool:
    row = query_qrels.get(chunk_id)
    return row is not None and row.is_gold


def get_qrels_by_query_chunk(qrels: pl.DataFrame) -> dict[str, dict[str, QrelRecord]]:
    result: dict[str, dict[str, QrelRecord]] = defaultdict(dict)

    for row in qrels.iter_rows(named=True):
        qrel_record = QrelRecord.model_validate(row)
        result[qrel_record.query_id][qrel_record.chunk_id] = qrel_record

    return result


def build_query_to_facet_gold_map(qrels: pl.DataFrame) -> QueryIdToFacetMap:
    result: QueryIdToFacetMap = defaultdict(lambda: defaultdict(list))

    for row in qrels.filter(pl.col('is_gold')).iter_rows(named=True):
        qrel = QrelRecord.model_validate(row)
        if qrel.facet_id is None:
            raise ValueError(f'gold qrel {qrel.chunk_id!r} has no facet_id')
        result[qrel.query_id][qrel.facet_id].append(qrel.chunk_id)

    return result


# Useful mathematical functions
def ci_half_width(values: list[float], z: float = 1.96) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return float('nan')
    return z * float(arr.std(ddof=1)) / float(np.sqrt(len(arr)))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-value)))


def harmonic_mean(left: float, right: float) -> float:
    denom = left + right
    return 0.0 if denom <= 0 else 2 * left * right / denom


def pair_kernel_polars_expr(kernel_cfg: MethodsComparisonKernelsCfg) -> pl.Expr:
    if kernel_cfg.pair_aggregation == 'arithmetic_mean':
        return (pl.col('fac_loc_kernel_score') + pl.col('mmr_kernel_score')) / 2.0
    if kernel_cfg.pair_aggregation == 'minimum':
        return pl.min_horizontal('fac_loc_kernel_score', 'mmr_kernel_score')
    if kernel_cfg.pair_aggregation == 'geometric_mean':
        return (pl.col('fac_loc_kernel_score') * pl.col('mmr_kernel_score')).sqrt()
    else:
        unreachable_code(
            f'Unexpected value {kernel_cfg.pair_aggregation} for kernel_cfg.pair_aggregation'
        )


def sigmoid_polars_expr(expr: pl.Expr) -> pl.Expr:
    return 1.0 / (1.0 + (-expr).exp())


# Miscellaneous utils
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
    if (
        paths.embeddings_meta_path.exists()
        and paths.embeddings_chunk_vectors_path.exists()
        and paths.embeddings_query_vectors_path.exists()
        and paths.embeddings_chunk_ids_path.exists()
        and paths.embeddings_query_ids_path.exists()
    ):
        chunk_vectors = np.load(paths.embeddings_chunk_vectors_path, mmap_mode='r')
        query_vectors = np.load(paths.embeddings_query_vectors_path, mmap_mode='r')
        chunk_ids = np.load(paths.embeddings_chunk_ids_path, mmap_mode='r')
        query_ids = np.load(paths.embeddings_query_ids_path, mmap_mode='r')
        return chunk_vectors, query_vectors, chunk_ids, query_ids

    payload = np.load(paths.embeddings_npz_path)
    chunk_vectors = np.asarray(payload['chunk_vectors'], dtype=np.float32)
    query_vectors = np.asarray(payload['query_vectors'], dtype=np.float32)
    chunk_ids = [str(x) for x in payload['chunk_ids'].tolist()]
    query_ids = [str(x) for x in payload['query_ids'].tolist()]
    return chunk_vectors, query_vectors, chunk_ids, query_ids
