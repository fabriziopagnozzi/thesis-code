"""Shared retrieval helpers for the synthetic benchmark.

This module exists to keep candidate-pool construction, similarity ranking, and
selection logic reusable across geometry filtering and evaluation. It uses
small index maps and existing helper algorithms for top-k, MMR, and
facility-location selection so the retrieval stages stay consistent.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from helpers.metrics import avg_cos, fac_cov_score, jaccard
from helpers.query_algorithms import fac_loc_lazy_greedy, mmr, top_k

type Strategy = Literal['top_k', 'mmr', 'fac_loc']


def build_index_maps(
    chunk_documents: pl.DataFrame,
    chunk_memberships: pl.DataFrame,
    queries: pl.DataFrame,
    chunk_ids: Sequence[str],
    query_ids: Sequence[str],
) -> dict[str, Any]:
    chunk_id_to_idx = {chunk_id: idx for idx, chunk_id in enumerate(chunk_ids)}
    query_id_to_idx = {query_id: idx for idx, query_id in enumerate(query_ids)}

    chunk_rows = chunk_documents.to_dicts()
    membership_rows = chunk_memberships.to_dicts()
    query_rows = queries.to_dicts()
    chunk_by_id = {row['chunk_id']: row for row in chunk_rows}
    query_by_id = {row['query_id']: row for row in query_rows}

    chunks_by_source_query: dict[str, list[int]] = defaultdict(list)
    chunks_by_condition: dict[str, list[int]] = defaultdict(list)
    for row in chunk_rows:
        chunk_id = row['chunk_id']
        if chunk_id not in chunk_id_to_idx:
            continue
        chunk_idx = chunk_id_to_idx[chunk_id]
        condition_id = row.get('condition_id')
        if condition_id:
            chunks_by_condition[str(condition_id)].append(chunk_idx)

    membership_by_query_chunk: dict[tuple[str, str], dict[str, Any]] = {}
    seen_by_query: dict[str, set[int]] = defaultdict(set)
    for row in membership_rows:
        chunk_id = row['chunk_id']
        if chunk_id not in chunk_id_to_idx:
            continue
        query_id = str(row['source_query_id'])
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
    pool_scope: Literal['query_local', 'same_condition', 'full_corpus'],
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


def select_indices(
    strategy: Strategy,
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
