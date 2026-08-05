from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TypedDict

import polars as pl

from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths
from helpers.embedder import MODEL_PROFILES

# Bump when the persisted chunk embedding cache schema or key semantics change.
CHUNK_EMBEDDING_CACHE_VERSION = 1
QUERY_EMBEDDING_SIGNATURE_VERSION = 1
# Two hash characters keep each cache shard near 1/256 of a signature cache.
CACHE_BUCKET_HEX_CHARS = 2


class EmbeddingSignaturePayload(TypedDict):
    cache_version: int
    model_name: str
    profile_mode: str
    document_prompt: str | None
    document_prompt_name: str | None
    normalize: bool


class QueryEmbeddingSignaturePayload(TypedDict):
    signature_version: int
    model_name: str
    profile_mode: str
    query_prompt: str | None
    query_prompt_name: str | None
    normalize: bool


def chunk_embedding_signature_payload(cfg: ExperimentCfg) -> EmbeddingSignaturePayload:
    profile = MODEL_PROFILES[cfg.embeddings.model_name]
    return {
        'cache_version': CHUNK_EMBEDDING_CACHE_VERSION,
        'model_name': cfg.embeddings.model_name,
        'profile_mode': profile.mode,
        'document_prompt': (
            cfg.embeddings.document_prompt
            if cfg.embeddings.document_prompt is not None
            else profile.document_prompt
        ),
        'document_prompt_name': profile.document_prompt_name
        if cfg.embeddings.document_prompt is None
        else None,
        'normalize': cfg.embeddings.normalize,
    }


def chunk_embedding_signature(cfg: ExperimentCfg) -> str:
    return _hash_json(chunk_embedding_signature_payload(cfg))


def query_embedding_signature_payload(cfg: ExperimentCfg) -> QueryEmbeddingSignaturePayload:
    profile = MODEL_PROFILES[cfg.embeddings.model_name]
    return {
        'signature_version': QUERY_EMBEDDING_SIGNATURE_VERSION,
        'model_name': cfg.embeddings.model_name,
        'profile_mode': profile.mode,
        'query_prompt': (
            cfg.embeddings.query_prompt
            if cfg.embeddings.query_prompt is not None
            else profile.query_prompt
        ),
        'query_prompt_name': profile.query_prompt_name
        if cfg.embeddings.query_prompt is None
        else None,
        'normalize': cfg.embeddings.normalize,
    }


def query_embedding_signature(cfg: ExperimentCfg) -> str:
    return _hash_json(query_embedding_signature_payload(cfg))


def chunk_embedding_cache_key(
    embedding_signature: str,
    text_sha256: str,
) -> str:
    # Embeddings are a pure function of model/prompt settings plus input text.
    # Local chunk IDs can change across chunk generation runs, so they must not
    # be part of the reusable embedding identity.
    return _hash_json(
        {
            'cache_version': CHUNK_EMBEDDING_CACHE_VERSION,
            'embedding_signature': embedding_signature,
            'text_sha256': text_sha256,
        }
    )


def load_matching_chunk_embedding_cache_by_text(
    paths: MedicalDatasetGenPaths,
    embedding_signature: str,
    text_sha256_values: list[str],
) -> pl.DataFrame:
    text_hashes = sorted(set(text_sha256_values))
    if not text_hashes:
        return pl.DataFrame()

    cache_paths = sorted(paths.chunk_embeddings_cache_dir(embedding_signature).glob('*.parquet'))
    if not cache_paths:
        return pl.DataFrame()

    matched = (
        pl.scan_parquet(cache_paths)
        .filter(pl.col('text_sha256').is_in(text_hashes))
        .collect(engine='streaming')
    )
    if matched.is_empty():
        return matched

    _validate_embedding_cache(matched)
    return _validated_unique_embedding_cache_by_text(matched)


def append_chunk_embedding_cache_rows(
    paths: MedicalDatasetGenPaths,
    embedding_signature: str,
    rows: pl.DataFrame,
) -> None:
    if rows.is_empty():
        return
    bucketed_rows = rows.with_columns(
        pl.Series(
            '_cache_bucket',
            [_cache_bucket(str(value)) for value in rows['chunk_embedding_cache_key'].to_list()],
        )
    )
    for bucket in sorted(str(value) for value in bucketed_rows['_cache_bucket'].unique().to_list()):
        bucket_rows = bucketed_rows.filter(pl.col('_cache_bucket') == bucket).drop('_cache_bucket')
        cache_path = paths.chunk_embeddings_bucket_path(embedding_signature, bucket)
        lock_path = paths.chunk_embeddings_lock_path(embedding_signature, bucket)
        with _file_lock(lock_path):
            existing = _read_parquet_if_exists(cache_path)
            if not existing.is_empty():
                _validate_embedding_cache(existing)
            merged = _validated_unique_cache(
                pl.concat([existing, bucket_rows], how='diagonal_relaxed')
                if not existing.is_empty()
                else bucket_rows,
                key_column='chunk_embedding_cache_key',
                fingerprint_column='embedding_payload_sha256',
                label='global chunk embedding cache',
            )
            _write_parquet_atomic(cache_path, merged)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def row_payload_sha256(row: dict[str, object], excluded_columns: set[str] | None = None) -> str:
    excluded = excluded_columns or set()
    payload = {key: row[key] for key in sorted(row) if key not in excluded}
    return _hash_json(payload)


def _validate_embedding_cache(cache: pl.DataFrame) -> None:
    required_columns = {
        'chunk_embedding_cache_key',
        'chunk_id',
        'text_sha256',
        'embedding_signature',
        'dimension',
        'embedding',
        'embedding_payload_sha256',
    }
    _require_columns(cache, required_columns, 'global chunk embedding cache')
    _validated_unique_cache(
        cache,
        key_column='chunk_embedding_cache_key',
        fingerprint_column='embedding_payload_sha256',
        label='global chunk embedding cache',
    )


def _validated_unique_embedding_cache_by_text(cache: pl.DataFrame) -> pl.DataFrame:
    conflicts = (
        cache.group_by('text_sha256')
        .agg(pl.col('dimension').n_unique().alias('n_dimensions'))
        .filter(pl.col('n_dimensions') > 1)
    )
    if conflicts.height:
        examples = conflicts['text_sha256'].head(5).to_list()
        raise RuntimeError(
            f'global chunk embedding cache has dimension conflicts for text hashes: {examples}'
        )
    return cache.unique(subset=['text_sha256'], keep='first', maintain_order=True)


def _validated_unique_cache(
    cache: pl.DataFrame,
    *,
    key_column: str,
    fingerprint_column: str,
    label: str,
) -> pl.DataFrame:
    conflicts = (
        cache.group_by(key_column)
        .agg(pl.col(fingerprint_column).n_unique().alias('n_fingerprints'))
        .filter(pl.col('n_fingerprints') > 1)
    )
    if conflicts.height:
        examples = conflicts[key_column].head(5).to_list()
        raise RuntimeError(f'{label} has conflicting duplicate rows: {examples}')
    return cache.unique(subset=[key_column], keep='first', maintain_order=True)


def _require_columns(cache: pl.DataFrame, required_columns: set[str], label: str) -> None:
    missing = sorted(required_columns - set(cache.columns))
    if missing:
        raise RuntimeError(f'{label} is missing required columns: {missing}')


def _read_parquet_if_exists(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


def _write_parquet_atomic(path: Path, df: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    df.write_parquet(tmp_path)
    os.replace(tmp_path, path)


@contextmanager
def _file_lock(lock_path: Path, timeout_seconds: float = 600.0) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    fd = -1

    while fd < 0:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
        except FileExistsError:
            if _lock_is_stale(lock_path):
                with suppress(FileNotFoundError):
                    lock_path.unlink()
                continue

            if time.monotonic() >= deadline:
                raise TimeoutError(f'timed out waiting for cache lock: {lock_path}') from None

            time.sleep(0.2)

    try:
        yield
    finally:
        os.close(fd)
        with suppress(FileNotFoundError):
            lock_path.unlink()


def _lock_is_stale(lock_path: Path) -> bool:
    try:
        raw_pid = lock_path.read_text().strip()
    except FileNotFoundError:
        return False
    except OSError:
        return False

    if not raw_pid.isdigit():
        return True

    pid = int(raw_pid)

    if pid <= 0:
        return True

    # Do not treat our own process lock as stale.
    if pid == os.getpid():
        return False

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False

    return False


def _hash_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _cache_bucket(value: str) -> str:
    if len(value) < CACHE_BUCKET_HEX_CHARS:
        raise ValueError(f'cache key is too short to bucket: {value!r}')
    return value[:CACHE_BUCKET_HEX_CHARS]
