from __future__ import annotations

import json

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from experiments.medical_dataset_gen.utils.global_utils import (
    EmbeddingArtifactName,
    MedicalDatasetGenPaths,
    SyntheticMedicalDatasetTableName,
)

EMBEDDING_ARRAY_ARTIFACTS: tuple[EmbeddingArtifactName, ...] = (
    'chunk_vectors',
    'chunk_ids',
    'query_vectors',
    'query_ids',
)
CHUNK_EMBEDDING_ARRAY_ARTIFACTS: tuple[EmbeddingArtifactName, ...] = (
    'chunk_vectors',
    'chunk_ids',
)
QUERY_EMBEDDING_ARRAY_ARTIFACTS: tuple[EmbeddingArtifactName, ...] = (
    'query_vectors',
    'query_ids',
)


def chunk_embedding_artifacts_ready(paths: MedicalDatasetGenPaths) -> bool:
    missing = [
        artifact
        for artifact in CHUNK_EMBEDDING_ARRAY_ARTIFACTS
        if not paths.embeddings_paths(artifact).exists()
    ]
    if missing:
        return False

    try:
        chunk_file = pq.ParquetFile(paths.table_path('chunk_documents'))
        n_chunks = chunk_file.metadata.num_rows
        chunk_vectors = np.load(paths.embeddings_paths('chunk_vectors'), mmap_mode='r')
        chunk_ids = np.load(paths.embeddings_paths('chunk_ids'), mmap_mode='r')
    except Exception as exc:
        print(f'[embed] existing chunk embedding artifacts are unreadable ({exc})')
        return False

    if chunk_vectors.ndim != 2:
        print('[embed] existing chunk embedding vectors have invalid rank')
        return False
    if chunk_vectors.shape[0] != n_chunks or chunk_ids.shape != (n_chunks,):
        print(
            '[embed] existing chunk embedding row count does not match '
            f'chunk_documents ({chunk_vectors.shape[0]}, {chunk_ids.shape[0]} != {n_chunks})'
        )
        return False
    if not _stored_ids_match_table(
        paths=paths,
        artifact='chunk_ids',
        table='chunk_documents',
        id_column='chunk_id',
    ):
        print('[embed] existing chunk embedding IDs do not match chunk_documents.parquet')
        return False
    return True


def query_embedding_artifacts_ready(paths: MedicalDatasetGenPaths) -> bool:
    missing = [
        artifact
        for artifact in QUERY_EMBEDDING_ARRAY_ARTIFACTS
        if not paths.embeddings_paths(artifact).exists()
    ]
    if missing:
        return False

    try:
        query_file = pq.ParquetFile(paths.table_path('queries'))
        n_queries = query_file.metadata.num_rows
        query_vectors = np.load(paths.embeddings_paths('query_vectors'), mmap_mode='r')
        query_ids = np.load(paths.embeddings_paths('query_ids'), mmap_mode='r')
    except Exception as exc:
        print(f'[embed] existing query embedding artifacts are unreadable ({exc})')
        return False

    if query_vectors.ndim != 2:
        print('[embed] existing query embedding vectors have invalid rank')
        return False
    if query_vectors.shape[0] != n_queries or query_ids.shape != (n_queries,):
        print(
            '[embed] existing query embedding row count does not match '
            f'queries ({query_vectors.shape[0]}, {query_ids.shape[0]} != {n_queries})'
        )
        return False
    if not _stored_ids_match_table(
        paths=paths,
        artifact='query_ids',
        table='queries',
        id_column='query_id',
    ):
        print('[embed] existing query embedding IDs do not match queries.parquet')
        return False
    return True


def embedding_artifacts_ready(paths: MedicalDatasetGenPaths) -> bool:
    if not _embedding_metadata_is_complete(paths):
        return False
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

    if not _stored_ids_match_table(
        paths=paths,
        artifact='chunk_ids',
        table='chunk_documents',
        id_column='chunk_id',
    ):
        print(
            '[embed] existing chunk embedding IDs do not match chunk_documents.parquet; rebuilding'
        )
        return False

    if not _stored_ids_match_table(
        paths=paths,
        artifact='query_ids',
        table='queries',
        id_column='query_id',
    ):
        print('[embed] existing query embedding IDs do not match queries.parquet; rebuilding')
        return False

    return True


def _embedding_metadata_is_complete(paths: MedicalDatasetGenPaths) -> bool:
    """Use the post-write metadata as the embedding transaction commit marker."""
    path = paths.embeddings_paths('metadata')
    if not path.is_file():
        print('[embed] embedding metadata is absent; rebuilding incomplete artifacts')
        return False
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f'[embed] embedding metadata is unreadable; rebuilding ({exc})')
        return False
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get('n_chunks'), int)
        or not isinstance(raw.get('n_queries'), int)
    ):
        print('[embed] embedding metadata lacks row counts; rebuilding incomplete artifacts')
        return False
    return True


def _stored_ids_match_table(
    *,
    paths: MedicalDatasetGenPaths,
    artifact: EmbeddingArtifactName,
    table: SyntheticMedicalDatasetTableName,
    id_column: str,
) -> bool:
    expected_ids = [
        str(value)
        for value in pl.read_parquet(paths.table_path(table), columns=[id_column])[
            id_column
        ].to_list()
    ]
    stored_ids = [str(value) for value in np.load(paths.embeddings_paths(artifact), mmap_mode='r')]
    return stored_ids == expected_ids
