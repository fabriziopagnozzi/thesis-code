import json

import numpy as np
import polars as pl
from numpy.typing import NDArray

from experiments.medical_dataset_gen.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
    read_parquet,
    write_parquet,
)


def run_embed(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    chunks = read_parquet(paths, 'chunks')
    queries = read_parquet(paths, 'queries')

    chunk_texts = chunks['text'].to_list()
    query_texts = queries['query_text'].to_list()

    if cfg.embeddings.backend == 'tfidf':
        chunk_vectors, query_vectors, meta = _embed_tfidf(cfg, chunk_texts, query_texts)
    else:
        chunk_vectors, query_vectors, meta = _embed_sentence_transformers(cfg, chunk_texts, query_texts)

    np.savez_compressed(
        paths.embeddings_npz_path,
        chunk_vectors=chunk_vectors,
        query_vectors=query_vectors,
        chunk_ids=np.array(chunks['chunk_id'].to_list()),
        query_ids=np.array(queries['query_id'].to_list()),
    )
    with open(paths.embeddings_meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

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
    return chunk_vectors, query_vectors


def load_embedding_arrays(
    paths: MedicalDatasetGenPaths,
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[str], list[str]]:
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
