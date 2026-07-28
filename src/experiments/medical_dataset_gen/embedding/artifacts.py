from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from experiments.medical_dataset_gen.utils.global_utils import (
    EmbeddingArtifactName,
    MedicalDatasetGenPaths,
)

EMBEDDING_ARRAY_ARTIFACTS: tuple[EmbeddingArtifactName, ...] = (
    'chunk_vectors',
    'chunk_ids',
    'query_vectors',
    'query_ids',
)


def embedding_artifacts_ready(paths: MedicalDatasetGenPaths) -> bool:
    missing = [
        artifact
        for artifact in EMBEDDING_ARRAY_ARTIFACTS
        if not paths.embeddings_paths(artifact).exists()
    ]
    if missing:
        return False

    try:
        chunk_file = pq.ParquetFile(paths.table_path('chunk_documents'))
        query_file = pq.ParquetFile(paths.table_path('queries'))
        n_chunks = chunk_file.metadata.num_rows
        n_queries = query_file.metadata.num_rows
        chunk_vectors = np.load(paths.embeddings_paths('chunk_vectors'), mmap_mode='r')
        query_vectors = np.load(paths.embeddings_paths('query_vectors'), mmap_mode='r')
        chunk_ids = np.load(paths.embeddings_paths('chunk_ids'), mmap_mode='r')
        query_ids = np.load(paths.embeddings_paths('query_ids'), mmap_mode='r')
    except Exception as exc:
        print(f'[embed] existing embedding artifacts are unreadable; rebuilding ({exc})')
        return False

    if chunk_vectors.ndim != 2 or query_vectors.ndim != 2:
        print('[embed] existing embedding vectors have invalid rank; rebuilding')
        return False
    if chunk_vectors.shape[1] != query_vectors.shape[1]:
        print('[embed] existing chunk/query embedding dimensions differ; rebuilding')
        return False
    if chunk_vectors.shape[0] != n_chunks or chunk_ids.shape != (n_chunks,):
        print(
            '[embed] existing chunk embedding row count does not match '
            f'chunk_documents ({chunk_vectors.shape[0]}, {chunk_ids.shape[0]} != {n_chunks}); '
            'rebuilding'
        )
        return False
    if query_vectors.shape[0] != n_queries or query_ids.shape != (n_queries,):
        print(
            '[embed] existing query embedding row count does not match '
            f'queries ({query_vectors.shape[0]}, {query_ids.shape[0]} != {n_queries}); rebuilding'
        )
        return False

    expected_query_ids = [
        str(value)
        for value in pl.read_parquet(paths.table_path('queries'), columns=['query_id'])[
            'query_id'
        ].to_list()
    ]
    stored_query_ids = [str(value) for value in query_ids]
    if stored_query_ids != expected_query_ids:
        print('[embed] existing query embedding IDs do not match queries.parquet; rebuilding')
        return False

    return True
