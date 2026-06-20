from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

from experiments.medical_dataset_gen.utils.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
)


def run_embed(
    cfg: ExperimentCfg, paths: MedicalDatasetGenPaths
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    chunk_vectors, query_vectors, meta = _embed_sentence_transformers_streaming(cfg, paths)

    with open(paths.embeddings_meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(
        '[write] embeddings arrays -> '
        f'{paths.embeddings_chunk_vectors_path}, {paths.embeddings_query_vectors_path}'
    )
    return chunk_vectors, query_vectors


def _embed_sentence_transformers_streaming(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
) -> tuple[NDArray[np.float32], NDArray[np.float32], dict]:
    from helpers.embedder import Embedder

    chunk_file = pq.ParquetFile(paths.table_path('chunk_documents'))
    query_file = pq.ParquetFile(paths.table_path('queries'))
    n_chunks = chunk_file.metadata.num_rows
    n_queries = query_file.metadata.num_rows
    bucket_size = max(cfg.embeddings.batch_size * 32, 32768)

    embedder = Embedder(
        model_name=cfg.embeddings.model_name,
        batch_size=cfg.embeddings.batch_size,
        query_prompt=cfg.embeddings.query_prompt,
        device=cfg.embeddings.device,
        devices=cfg.embeddings.devices,
    )
    try:
        dim = embedder.dim
        chunk_vectors = np.lib.format.open_memmap(
            paths.embeddings_chunk_vectors_path,
            mode='w+',
            dtype=np.float32,
            shape=(n_chunks, dim),
        )
        query_vectors = np.lib.format.open_memmap(
            paths.embeddings_query_vectors_path,
            mode='w+',
            dtype=np.float32,
            shape=(n_queries, dim),
        )
        chunk_ids = np.lib.format.open_memmap(
            paths.embeddings_chunk_ids_path,
            mode='w+',
            dtype='U32',
            shape=(n_chunks,),
        )
        query_ids = np.lib.format.open_memmap(
            paths.embeddings_query_ids_path,
            mode='w+',
            dtype='U32',
            shape=(n_queries,),
        )

        chunk_written = _fill_embedding_memmaps(
            file=chunk_file,
            id_column='chunk_id',
            text_column='text',
            vectors=chunk_vectors,
            ids=chunk_ids,
            embed_fn=lambda texts: embedder.embed_docs(texts, normalize=cfg.embeddings.normalize),
            batch_size=bucket_size,
            desc='Embedding chunks',
        )
        query_written = _fill_embedding_memmaps(
            file=query_file,
            id_column='query_id',
            text_column='query_text',
            vectors=query_vectors,
            ids=query_ids,
            embed_fn=lambda texts: embedder.embed_queries(
                texts,
                normalize=cfg.embeddings.normalize,
            ),
            batch_size=bucket_size,
            desc='Embedding queries',
        )
        if chunk_written != n_chunks or query_written != n_queries:
            raise RuntimeError(
                f'embedding row counts mismatch: chunks={chunk_written}/{n_chunks}, '
                f'queries={query_written}/{n_queries}'
            )
        meta = {
            'backend': 'sentence_transformers',
            'model_name': cfg.embeddings.model_name,
            'dimension': int(dim),
            'normalized': cfg.embeddings.normalize,
            'device': cfg.embeddings.device,
            'devices': list(cfg.embeddings.devices),
            'format': 'npy_memmap',
            'chunk_vectors_file': str(paths.embeddings_chunk_vectors_path),
            'query_vectors_file': str(paths.embeddings_query_vectors_path),
            'chunk_ids_file': str(paths.embeddings_chunk_ids_path),
            'query_ids_file': str(paths.embeddings_query_ids_path),
        }
        return chunk_vectors, query_vectors, meta
    finally:
        embedder.release()


def _fill_embedding_memmaps(
    *,
    file: pq.ParquetFile,
    id_column: str,
    text_column: str,
    vectors: np.ndarray,
    ids: np.ndarray,
    embed_fn,
    batch_size: int,
    desc: str,
) -> int:
    from tqdm import tqdm

    offset = 0
    for batch in tqdm(
        file.iter_batches(columns=[id_column, text_column], batch_size=batch_size),
        desc=desc,
        dynamic_ncols=True,
    ):
        id_values = batch.column(0).to_pylist()
        text_values = batch.column(1).to_pylist()
        batch_vectors = embed_fn(text_values)
        n_rows = len(id_values)
        vectors[offset : offset + n_rows] = batch_vectors
        ids[offset : offset + n_rows] = np.asarray(id_values, dtype=ids.dtype)
        offset += n_rows
    return offset


if __name__ == '__main__':
    from experiments.medical_dataset_gen.utils.global_configs import (
        load_config_from_cli,
        paths_for,
        setup_logging,
    )

    cfg = load_config_from_cli()
    paths = paths_for(cfg)
    setup_logging(paths)
    run_embed(cfg, paths)
