import json
from collections.abc import Sequence

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from numpy.typing import NDArray

from experiments.medical_dataset_gen.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
    read_parquet,
    write_parquet,
)


def run_embed(
    cfg: ExperimentCfg, paths: MedicalDatasetGenPaths
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if cfg.embeddings.backend == 'tfidf':
        chunks = read_parquet(paths, 'chunks')
        queries = read_parquet(paths, 'queries')
        chunk_texts = chunks['text'].to_list()
        query_texts = queries['query_text'].to_list()
        chunk_vectors, query_vectors, meta = _embed_tfidf(cfg, chunk_texts, query_texts)
    else:
        chunk_vectors, query_vectors, meta = _embed_sentence_transformers_streaming(cfg, paths)

    with open(paths.embeddings_meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    if cfg.embeddings.backend == 'tfidf':
        np.savez_compressed(
            paths.embeddings_npz_path,
            chunk_vectors=chunk_vectors,
            query_vectors=query_vectors,
            chunk_ids=np.array(chunks['chunk_id'].to_list()),
            query_ids=np.array(queries['query_id'].to_list()),
        )
        index_rows = [
            {
                'kind': 'chunk',
                'object_id': chunk_id,
                'row_idx': i,
                'backend': cfg.embeddings.backend,
                'model_name': cfg.embeddings.model_name,
                'vector_file': str(paths.embeddings_npz_path),
            }
            for i, chunk_id in enumerate(chunks['chunk_id'].to_list())
        ]
        index_rows.extend(
            {
                'kind': 'query',
                'object_id': query_id,
                'row_idx': i,
                'backend': cfg.embeddings.backend,
                'model_name': cfg.embeddings.model_name,
                'vector_file': str(paths.embeddings_npz_path),
            }
            for i, query_id in enumerate(queries['query_id'].to_list())
        )
        write_parquet(paths, 'embeddings', pl.DataFrame(index_rows))
        print(f'[write] embeddings arrays -> {paths.embeddings_npz_path}')
    else:
        print(
            '[write] embeddings arrays -> '
            f'{paths.embeddings_chunk_vectors_path}, {paths.embeddings_query_vectors_path}'
        )
    return chunk_vectors, query_vectors


def load_embedding_arrays(
    paths: MedicalDatasetGenPaths,
) -> tuple[NDArray[np.float32], NDArray[np.float32], Sequence[str], Sequence[str]]:
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


def _embed_tfidf(
    cfg: ExperimentCfg,
    chunk_texts: list[str],
    query_texts: list[str],
) -> tuple[NDArray[np.float32], NDArray[np.float32], dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        ngram_range=(cfg.embeddings.tfidf_ngram_min, cfg.embeddings.tfidf_ngram_max),
        min_df=1,
        norm='l2' if cfg.embeddings.normalize else None,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform([*chunk_texts, *query_texts])
    dense = np.asarray(matrix.toarray(), dtype=np.float32)
    chunk_vectors = dense[: len(chunk_texts)]
    query_vectors = dense[len(chunk_texts) :]
    meta = {
        'backend': 'tfidf',
        'model_name': 'sklearn.TfidfVectorizer',
        'dimension': int(dense.shape[1]),
        'ngram_range': [cfg.embeddings.tfidf_ngram_min, cfg.embeddings.tfidf_ngram_max],
        'normalized': cfg.embeddings.normalize,
    }
    return chunk_vectors, query_vectors, meta


def _embed_sentence_transformers(
    cfg: ExperimentCfg,
    chunk_texts: list[str],
    query_texts: list[str],
) -> tuple[NDArray[np.float32], NDArray[np.float32], dict]:
    from helpers.embedder import Embedder

    embedder = Embedder(
        model_name=cfg.embeddings.model_name,
        batch_size=cfg.embeddings.batch_size,
        query_prompt=cfg.embeddings.query_prompt,
        device=cfg.embeddings.device,
    )
    try:
        chunk_vectors = embedder.embed_docs(chunk_texts, normalize=cfg.embeddings.normalize)
        query_vectors = embedder.embed_queries(query_texts, normalize=cfg.embeddings.normalize)
        meta = {
            'backend': 'sentence_transformers',
            'model_name': cfg.embeddings.model_name,
            'dimension': int(chunk_vectors.shape[1]),
            'normalized': cfg.embeddings.normalize,
        }
        return chunk_vectors, query_vectors, meta
    finally:
        embedder.release()


def _embed_sentence_transformers_streaming(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
) -> tuple[NDArray[np.float32], NDArray[np.float32], dict]:
    from helpers.embedder import Embedder

    chunk_file = pq.ParquetFile(paths.table_path('chunks'))
    query_file = pq.ParquetFile(paths.table_path('queries'))
    n_chunks = chunk_file.metadata.num_rows
    n_queries = query_file.metadata.num_rows
    batch_size = max(cfg.embeddings.batch_size * 16, 256)

    embedder = Embedder(
        model_name=cfg.embeddings.model_name,
        batch_size=cfg.embeddings.batch_size,
        query_prompt=cfg.embeddings.query_prompt,
        device=cfg.embeddings.device,
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
            batch_size=batch_size,
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
            batch_size=batch_size,
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
    from experiments.medical_dataset_gen.global_configs import (
        dump_effective_config,
        load_config_from_cli,
        paths_for,
        setup_logging,
    )

    cfg = load_config_from_cli()
    paths = paths_for(cfg)
    setup_logging(paths)
    dump_effective_config(cfg, paths)
    run_embed(cfg, paths)
