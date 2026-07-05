from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from numpy.typing import NDArray

from experiments.medical_dataset_gen.dataset_generation.deterministic_caches import (
    CHUNK_EMBEDDING_CACHE_VERSION,
    ChunkEmbeddingCacheStats,
    append_chunk_embedding_cache_rows,
    chunk_embedding_cache_key,
    chunk_embedding_signature,
    chunk_embedding_signature_payload,
    embedding_cache_bucket,
    load_matching_chunk_embedding_cache,
    row_payload_sha256,
    text_sha256,
)
from experiments.medical_dataset_gen.schemas.global_config_schemas import (
    ExperimentCfg,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    load_config_from_cli,
    paths_for,
)


def run_embed(
    cfg: ExperimentCfg, paths: MedicalDatasetGenPaths
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    chunk_vectors, query_vectors, meta = _embed_sentence_transformers_streaming(cfg, paths)

    with open(paths.embeddings_paths('metadata'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(
        '[write] embeddings arrays -> '
        f'{paths.embeddings_paths("chunk_vectors"), paths.embeddings_paths("query_vectors")}'
    )
    return chunk_vectors, query_vectors


def _embed_sentence_transformers_streaming(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths):
    from helpers.embedder import Embedder

    chunk_file = pq.ParquetFile(paths.table_path('chunk_documents'))
    query_file = pq.ParquetFile(paths.table_path('queries'))
    n_chunks = chunk_file.metadata.num_rows
    n_queries = query_file.metadata.num_rows
    bucket_size = max(cfg.embeddings.batch_size * 32, 125_000)

    def embed_docs_fn(texts: list[str]) -> NDArray[np.float32]:
        return embedder.embed_docs(texts, normalize=cfg.embeddings.normalize)

    def embed_queries_fn(texts: list[str]) -> NDArray[np.float32]:
        return embedder.embed_queries(texts, normalize=cfg.embeddings.normalize)

    embedder = Embedder(
        model_name=cfg.embeddings.model_name,
        batch_size=cfg.embeddings.batch_size,
        query_prompt=cfg.embeddings.query_prompt,
        document_prompt=cfg.embeddings.document_prompt,
        device=cfg.embeddings.device,
        devices=cfg.embeddings.devices,
    )
    try:
        dim = embedder.dim
        chunk_vectors = np.lib.format.open_memmap(
            paths.embeddings_paths('chunk_vectors'),
            mode='w+',
            dtype=np.float32,
            shape=(n_chunks, dim),
        )
        query_vectors = np.lib.format.open_memmap(
            paths.embeddings_paths('query_vectors'),
            mode='w+',
            dtype=np.float32,
            shape=(n_queries, dim),
        )
        chunk_ids = np.lib.format.open_memmap(
            paths.embeddings_paths('chunk_ids'),
            mode='w+',
            dtype='U32',
            shape=(n_chunks,),
        )
        query_ids = np.lib.format.open_memmap(
            paths.embeddings_paths('query_ids'),
            mode='w+',
            dtype='U32',
            shape=(n_queries,),
        )

        chunk_cache_stats: ChunkEmbeddingCacheStats | None = None
        if _uses_global_chunk_embedding_cache(cfg):
            chunk_written, chunk_cache_stats = _fill_deterministic_chunk_embedding_memmaps(
                cfg=cfg,
                paths=paths,
                vectors=chunk_vectors,
                ids=chunk_ids,
                embed_fn=embed_docs_fn,
                dim=dim,
                # Misses also become parquet cache rows, so keep this bounded by model batch size.
                batch_size=max(cfg.embeddings.batch_size * 2048, 1),
            )
        else:
            chunk_written = _fill_embedding_memmaps(
                file=chunk_file,
                id_column='chunk_id',
                text_column='text',
                vectors=chunk_vectors,
                ids=chunk_ids,
                embed_fn=embed_docs_fn,
                batch_size=bucket_size,
                desc='Embedding chunks',
            )
        query_written = _fill_embedding_memmaps(
            file=query_file,
            id_column='query_id',
            text_column='query_text',
            vectors=query_vectors,
            ids=query_ids,
            embed_fn=embed_queries_fn,
            batch_size=bucket_size,
            desc='Embedding queries',
        )
        if chunk_written != n_chunks or query_written != n_queries:
            raise RuntimeError(
                f'embedding row counts mismatch: chunks={chunk_written}/{n_chunks}, '
                f'queries={query_written}/{n_queries}'
            )
        meta = {
            'dataset_schema_version': cfg.dataset_schema_version,
            'backend': 'medcpt'
            if cfg.embeddings.model_name == 'ncbi/MedCPT'
            else 'sentence_transformers',
            'model_name': cfg.embeddings.model_name,
            'query_prompt': cfg.embeddings.query_prompt,
            'document_prompt': cfg.embeddings.document_prompt,
            'dimension': int(dim),
            'normalized': cfg.embeddings.normalize,
            'device': cfg.embeddings.device,
            'devices': list(cfg.embeddings.devices),
            'format': 'npy_memmap',
            'n_chunks': int(n_chunks),
            'n_queries': int(n_queries),
            'chunk_vectors_file': str(paths.embeddings_paths('chunk_vectors')),
            'query_vectors_file': str(paths.embeddings_paths('query_vectors')),
            'chunk_ids_file': str(paths.embeddings_paths('chunk_ids')),
            'query_ids_file': str(paths.embeddings_paths('query_ids')),
        }
        if chunk_cache_stats is not None:
            meta['chunk_embedding_cache'] = chunk_cache_stats
        return chunk_vectors, query_vectors, meta
    finally:
        embedder.release()


def _uses_global_chunk_embedding_cache(cfg: ExperimentCfg) -> bool:
    return (
        not cfg.generation.llm_config.use_llm_chunk_generation
        and not cfg.generation.llm_config.use_llm_chunk_rewriting
    )


def _fill_deterministic_chunk_embedding_memmaps(
    *,
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    vectors: np.ndarray,
    ids: np.ndarray,
    embed_fn: Callable[[list[str]], NDArray[np.float32]],
    dim: int,
    batch_size: int,
) -> tuple[int, ChunkEmbeddingCacheStats]:
    from tqdm import tqdm

    embedding_signature = chunk_embedding_signature(cfg)
    cache_dir = paths.chunk_embeddings_cache_dir(embedding_signature)
    chunk_documents = _chunk_documents_with_embedding_cache_keys(paths, embedding_signature)
    cache = load_matching_chunk_embedding_cache(
        paths,
        embedding_signature,
        _embedding_cache_keys_by_bucket(chunk_documents),
    )
    if cache.is_empty():
        joined = chunk_documents.with_columns(
            pl.lit(None).alias('cached_text_sha256'),
            pl.lit(None).alias('dimension'),
            pl.lit(None).alias('embedding'),
        )
    else:
        joined = chunk_documents.join(
            cache.select(
                'chunk_embedding_cache_key',
                pl.col('text_sha256').alias('cached_text_sha256'),
                'dimension',
                'embedding',
            ),
            on='chunk_embedding_cache_key',
            how='left',
            validate='m:1',
        )

    miss_rows: list[dict[str, object]] = []
    hits = 0
    for row in joined.iter_rows(named=True):
        row_index = int(row['_row_idx'])
        chunk_id_value = str(row['chunk_id'])
        ids[row_index] = chunk_id_value
        embedding = row.get('embedding')
        if embedding is None:
            miss_rows.append(row)
            continue
        cached_dimension = row.get('dimension')
        if not isinstance(cached_dimension, int):
            raise RuntimeError(f'cached embedding dimension is invalid for {chunk_id_value}')
        if cached_dimension != dim:
            raise RuntimeError(
                f'cached embedding dimension mismatch for {chunk_id_value}: '
                f'{cached_dimension} != {dim}'
            )
        if str(row['cached_text_sha256']) != str(row['text_sha256']):
            raise RuntimeError(f'cached text hash mismatch for {chunk_id_value}')
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.shape != (dim,):
            raise RuntimeError(
                f'cached embedding shape mismatch for {chunk_id_value}: {vector.shape} != {(dim,)}'
            )
        vectors[row_index] = vector
        hits += 1

    print(
        '[embed] deterministic chunk embedding cache: '
        f'{hits:,} hit(s), {len(miss_rows):,} miss(es) -> {cache_dir}\n'
        'Starting embedding...\n'
    )

    effective_batch_size = max(1, batch_size)
    for start in tqdm(
        range(0, len(miss_rows), effective_batch_size),
        desc='Embedding uncached deterministic chunks',
        dynamic_ncols=True,
    ):
        batch_rows = miss_rows[start : start + effective_batch_size]
        batch_vectors = np.asarray(
            embed_fn([str(row['text']) for row in batch_rows]),
            dtype=np.float32,
        )
        if batch_vectors.shape != (len(batch_rows), dim):
            raise RuntimeError(
                f'embedding batch shape mismatch: {batch_vectors.shape} != {(len(batch_rows), dim)}'
            )
        batch_cache_rows: list[dict[str, object]] = []
        for row, vector in zip(batch_rows, batch_vectors, strict=True):
            raw_row_index = row['_row_idx']
            if not isinstance(raw_row_index, int):
                raise RuntimeError('missing chunk row index while writing cached embeddings')
            row_index = raw_row_index
            chunk_id_value = str(row['chunk_id'])
            vectors[row_index] = vector
            ids[row_index] = chunk_id_value
            cache_row = {
                'chunk_embedding_cache_key': str(row['chunk_embedding_cache_key']),
                'chunk_id': chunk_id_value,
                'text_sha256': str(row['text_sha256']),
                'embedding_signature': embedding_signature,
                'embedding_signature_payload_json': json.dumps(
                    chunk_embedding_signature_payload(cfg),
                    sort_keys=True,
                ),
                'dimension': dim,
                'embedding': [float(value) for value in vector.tolist()],
                'chunk_embedding_cache_version': CHUNK_EMBEDDING_CACHE_VERSION,
            }
            cache_row['embedding_payload_sha256'] = row_payload_sha256(
                cache_row,
                excluded_columns={'embedding_payload_sha256'},
            )
            batch_cache_rows.append(cache_row)
        if batch_cache_rows:
            append_chunk_embedding_cache_rows(
                paths,
                embedding_signature,
                pl.from_dicts(batch_cache_rows, infer_schema_length=None).with_columns(
                    pl.col('embedding').cast(pl.List(pl.Float32))
                ),
            )

    stats: ChunkEmbeddingCacheStats = {
        'cache_path': str(cache_dir),
        'embedding_signature': embedding_signature,
        'hits': hits,
        'misses': len(miss_rows),
    }
    return chunk_documents.height, stats


def _chunk_documents_with_embedding_cache_keys(
    paths: MedicalDatasetGenPaths,
    embedding_signature: str,
) -> pl.DataFrame:
    docs = pl.read_parquet(paths.table_path('chunk_documents'))
    required_columns = {'chunk_id', 'text'}
    missing = sorted(required_columns - set(docs.columns))
    if missing:
        raise RuntimeError(f'chunk_documents missing required embedding columns: {missing}')
    if 'text_sha256' not in docs.columns:
        docs = docs.with_columns(
            pl.Series('text_sha256', [text_sha256(str(value)) for value in docs['text'].to_list()])
        )
    keys = [
        chunk_embedding_cache_key(
            embedding_signature,
            str(chunk_id_value),
            str(text_hash),
        )
        for chunk_id_value, text_hash in zip(
            docs['chunk_id'].to_list(),
            docs['text_sha256'].to_list(),
            strict=True,
        )
    ]
    return (
        docs
        .select('chunk_id', 'text', 'text_sha256')
        .with_row_index('_row_idx')
        .with_columns(pl.Series('chunk_embedding_cache_key', keys))
    )


def _embedding_cache_keys_by_bucket(chunk_documents: pl.DataFrame) -> dict[str, list[str]]:
    keys_by_bucket: dict[str, set[str]] = {}
    for raw_key in chunk_documents['chunk_embedding_cache_key'].to_list():
        cache_key = str(raw_key)
        bucket = embedding_cache_bucket(cache_key)
        keys_by_bucket.setdefault(bucket, set()).add(cache_key)
    return {bucket: sorted(cache_keys) for bucket, cache_keys in sorted(keys_by_bucket.items())}


def _fill_embedding_memmaps(
    *,
    file: pq.ParquetFile,
    id_column: str,
    text_column: str,
    vectors: np.ndarray,
    ids: np.ndarray,
    embed_fn: Callable[..., object],
    batch_size: int,
    desc: str,
) -> int:
    from tqdm import tqdm

    offset = 0
    for batch in tqdm(
        file.iter_batches(columns=[id_column, text_column], batch_size=batch_size),  # type: ignore
        desc=desc,
        dynamic_ncols=True,
    ):
        id_values = batch.column(0).to_pylist()
        text_values = batch.column(1).to_pylist()
        batch_vectors = embed_fn(text_values)
        n_rows = len(id_values)  # type: ignore
        vectors[offset : offset + n_rows] = batch_vectors
        ids[offset : offset + n_rows] = np.asarray(id_values, dtype=ids.dtype)
        offset += n_rows
    return offset


if __name__ == '__main__':
    from experiments.medical_dataset_gen.utils.global_utils import (
        setup_logging,
    )

    cfg = load_config_from_cli()
    paths = paths_for(cfg)
    setup_logging(paths)
    run_embed(cfg, paths)
