import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from random import Random
from typing import Any

import polars as pl
from tqdm import tqdm

from experiments.medical_dataset_gen.generation.ontology import load_ontology
from experiments.medical_dataset_gen.generation.text_templates import (
    ChunkValidation,
    maybe_generate_chunk_text,
    render_chunk_text,
    validate_chunk_text,
)
from experiments.medical_dataset_gen.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
    read_parquet,
    write_parquet,
)

_CACHE_VERSION = 5


def run_make_chunks(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    ontology = load_ontology(cfg)
    facts = read_parquet(paths, 'clinical_facts')
    fact_rows = [dict(row) for row in facts.iter_rows(named=True)]
    rng = Random(cfg.global_.seed + 1000)
    if cfg.generation.use_llm_chunk_generation:
        print(f'[chunks] LLM generation enabled for all {len(facts):,} facts')
        if cfg.generation.llm_workers > 1:
            print(
                f'[chunks] using {cfg.generation.llm_workers} parallel workers; '
                f'configure the Ollama server with OLLAMA_NUM_PARALLEL>={cfg.generation.llm_workers} '
                'for actual concurrent decoding'
            )
    else:
        print('[chunks] LLM generation disabled; using deterministic clinical fallback')

    cache_path = paths.experiment_dir / 'chunk_generation_cache.jsonl'
    cache = _load_generation_cache(cache_path)
    if cache:
        print(f'[chunks] loaded {len(cache):,} cached chunk generations from {cache_path}')

    if cfg.generation.use_llm_chunk_generation and cfg.generation.llm_workers > 1:
        rows, rejects, failed_queries = _render_chunks_parallel_llm(
            cfg=cfg,
            paths=paths,
            facts=fact_rows,
            ontology=ontology,
            cache=cache,
            cache_path=cache_path,
        )
    else:
        rows, rejects, failed_queries = _render_chunks_sequential(
            cfg=cfg,
            paths=paths,
            facts=fact_rows,
            ontology=ontology,
            cache=cache,
            cache_path=cache_path,
            rng=rng,
        )

    chunks = pl.DataFrame(rows)
    write_parquet(paths, 'chunks', chunks)

    reject_df = pl.DataFrame(rejects) if rejects else pl.DataFrame(
        schema={
            'fact_id': pl.String,
            'query_id': pl.String,
            'reason': pl.String,
            'llm_text': pl.String,
        }
    )
    write_parquet(paths, 'generation_rejects', reject_df)
    soft_warning_count = sum(int(row.get('validation_soft_warning_count', 0)) for row in rows)
    chunks_with_soft_warnings = sum(1 for row in rows if int(row.get('validation_soft_warning_count', 0)) > 0)
    if chunks_with_soft_warnings:
        print(
            f'[chunks] kept {chunks_with_soft_warnings:,} chunk(s) with soft warnings '
            f'({soft_warning_count:,} warnings total)'
        )
    if failed_queries:
        print(
            f'[chunks] dropped {len(failed_queries):,} query/queries after LLM validation failure; '
            f'kept {len(chunks):,}/{len(facts):,} chunk rows'
        )
    return chunks


def _write_rejects(paths: MedicalDatasetGenPaths, rejects: list[dict[str, Any]]) -> None:
    reject_df = pl.DataFrame(rejects) if rejects else pl.DataFrame(
        schema={
            'fact_id': pl.String,
            'query_id': pl.String,
            'reason': pl.String,
            'llm_text': pl.String,
        }
    )
    write_parquet(paths, 'generation_rejects', reject_df)


def _render_chunks_sequential(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    facts: list[dict[str, Any]],
    ontology: dict[str, Any],
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    rng: Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    rows: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    failed_queries: set[str] = set()

    for i, fact in tqdm(
        enumerate(facts),
        total=len(facts),
        desc='Rendering chunks',
        dynamic_ncols=True,
    ):
        if fact['query_id'] in failed_queries:
            continue
        cached = _cached_chunk_state(cfg, fact, ontology, cache)
        if cached is not None:
            rows.append(_row_from_state(i, fact, cached))
            continue

        if cfg.generation.use_llm_chunk_generation:
            final_text, attempt_errors = _generate_llm_chunk(cfg=cfg, fact=fact, ontology=ontology)
            if attempt_errors:
                rejects.append(_reject_row(fact, '; '.join(attempt_errors), final_text))
                failed_queries.add(str(fact['query_id']))
                print(
                    f'[chunks] dropping query {fact["query_id"]} after failed chunk {fact["fact_id"]}: '
                    + '; '.join(attempt_errors)
                )
                continue
            validation = validate_chunk_text(final_text, fact, ontology)
            state = _new_chunk_state(
                final_text,
                text_generation_source='llm',
                llm_attempted=True,
                llm_rejected=False,
                validation=validation,
            )
        else:
            final_text = render_chunk_text(fact, ontology, rng)
            validation = validate_chunk_text(final_text, fact, ontology)
            state = _new_chunk_state(
                final_text,
                text_generation_source='fallback',
                llm_attempted=False,
                llm_rejected=False,
                validation=validation,
            )

        try:
            row, cache_entry = _finalize_chunk_row(
                cfg=cfg,
                fact=fact,
                ontology=ontology,
                index=i,
                state=state,
                should_cache=True,
            )
        except RuntimeError as exc:
            rejects.append(_reject_row(fact, str(exc), state['final_text']))
            failed_queries.add(str(fact['query_id']))
            print(
                f'[chunks] dropping query {fact["query_id"]} after final validation failure in {fact["fact_id"]}: {exc}'
            )
            continue

        if cache_entry is not None:
            _append_generation_cache(cache_path, cache_entry)
        rows.append(row)

    kept_rows = [row for row in rows if str(row['query_id']) not in failed_queries]
    return kept_rows, rejects, failed_queries


def _render_chunks_parallel_llm(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    facts: list[dict[str, Any]],
    ontology: dict[str, Any],
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    rows: list[dict[str, Any] | None] = [None] * len(facts)
    rejects: list[dict[str, Any]] = []
    failed_queries: set[str] = set()
    pending: dict[Future[tuple[str, list[str]]], tuple[int, dict[str, Any]]] = {}
    next_idx = 0
    max_in_flight = max(cfg.generation.llm_workers * 2, cfg.generation.llm_workers)
    executor = ThreadPoolExecutor(max_workers=cfg.generation.llm_workers, thread_name_prefix='mdg-llm')

    try:
        with tqdm(total=len(facts), desc='Rendering chunks', dynamic_ncols=True) as pbar:
            while next_idx < len(facts) or pending:
                while next_idx < len(facts) and len(pending) < max_in_flight:
                    fact = facts[next_idx]
                    if fact['query_id'] in failed_queries:
                        pbar.update(1)
                        next_idx += 1
                        continue
                    cached = _cached_chunk_state(cfg, fact, ontology, cache)
                    if cached is not None:
                        rows[next_idx] = _row_from_state(next_idx, fact, cached)
                        pbar.update(1)
                        next_idx += 1
                        continue

                    future = executor.submit(_generate_llm_chunk, cfg, fact, ontology)
                    pending[future] = (next_idx, fact)
                    next_idx += 1

                if not pending:
                    continue

                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    idx, fact = pending.pop(future)
                    final_text, attempt_errors = future.result()
                    if fact['query_id'] in failed_queries:
                        pbar.update(1)
                        continue
                    if attempt_errors:
                        rejects.append(_reject_row(fact, '; '.join(attempt_errors), final_text))
                        failed_queries.add(str(fact['query_id']))
                        print(
                            f'[chunks] dropping query {fact["query_id"]} after failed chunk {fact["fact_id"]}: '
                            + '; '.join(attempt_errors)
                        )
                        pbar.update(1)
                        continue

                    state = _new_chunk_state(
                        final_text,
                        text_generation_source='llm',
                        llm_attempted=True,
                        llm_rejected=False,
                        validation=validate_chunk_text(final_text, fact, ontology),
                    )
                    try:
                        row, cache_entry = _finalize_chunk_row(
                            cfg=cfg,
                            fact=fact,
                            ontology=ontology,
                            index=idx,
                            state=state,
                            should_cache=True,
                        )
                    except RuntimeError as exc:
                        rejects.append(_reject_row(fact, str(exc), final_text))
                        failed_queries.add(str(fact['query_id']))
                        print(
                            f'[chunks] dropping query {fact["query_id"]} after final validation failure in '
                            f'{fact["fact_id"]}: {exc}'
                        )
                        pbar.update(1)
                        continue

                    if cache_entry is not None:
                        _append_generation_cache(cache_path, cache_entry)
                    rows[idx] = row
                    pbar.update(1)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    materialized_rows = [
        row for row in rows
        if row is not None and str(row['query_id']) not in failed_queries
    ]
    return materialized_rows, rejects, failed_queries


def _load_generation_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    cache: dict[str, dict[str, Any]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('cache_version') != _CACHE_VERSION:
                continue
            fact_id = row.get('fact_id')
            text = row.get('text')
            if fact_id and text:
                cache[str(fact_id)] = row
    return cache


def _append_generation_cache(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(row, sort_keys=True) + '\n')
        f.flush()


def _generate_llm_chunk(
    cfg: ExperimentCfg,
    fact: dict[str, Any],
    ontology: dict[str, Any],
) -> tuple[str, list[str]]:
    last_text = ''
    last_errors = ['empty LLM generation']
    feedback: str | None = None

    for attempt in range(1, max(1, cfg.generation.llm_chunk_max_attempts) + 1):
        candidate_text = maybe_generate_chunk_text(
            fallback_text='',
            fact=fact,
            ontology=ontology,
            llm_name=cfg.generation.llm_name,
            use_llm=True,
            temperature=cfg.generation.llm_temperature,
            num_ctx=cfg.generation.llm_num_ctx,
            chunk_min_words=cfg.generation.chunk_min_words,
            chunk_max_words=cfg.generation.chunk_max_words,
            revision_feedback=feedback,
        )
        last_text = candidate_text
        word_count = len(candidate_text.split())
        validation = validate_chunk_text(candidate_text, fact, ontology)
        word_errors = _word_count_errors(
            word_count,
            min_words=cfg.generation.chunk_min_words,
            max_words=cfg.generation.chunk_max_words,
            tolerance=cfg.generation.chunk_word_tolerance,
        )
        errors = validation.hard_errors + word_errors
        if not errors:
            return candidate_text, []

        last_errors = [f'attempt={attempt}', *errors]
        feedback = '\n'.join(f'- {error}' for error in errors)
        print(f'[chunks] retry {attempt} for {fact["fact_id"]}: ' + '; '.join(errors))

    return last_text, last_errors


def _word_count_ok(word_count: int, min_words: int, max_words: int, tolerance: int) -> bool:
    return (min_words - tolerance) <= word_count <= (max_words + tolerance)


def _word_count_errors(word_count: int, min_words: int, max_words: int, tolerance: int) -> list[str]:
    if _word_count_ok(word_count, min_words, max_words, tolerance):
        return []
    if word_count < min_words:
        return [f'word_count={word_count} below minimum {min_words} (tolerance {tolerance})']
    return [f'word_count={word_count} above maximum {max_words} (tolerance {tolerance})']


def _cached_chunk_state(
    cfg: ExperimentCfg,
    fact: dict[str, Any],
    ontology: dict[str, Any],
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    cached = cache.get(fact['fact_id'])
    if not cached:
        return None

    cached_text = str(cached['text'])
    validation = validate_chunk_text(cached_text, fact, ontology)
    errors = [*validation.hard_errors]
    errors.extend(
        _word_count_errors(
            len(cached_text.split()),
            min_words=cfg.generation.chunk_min_words,
            max_words=cfg.generation.chunk_max_words,
            tolerance=cfg.generation.chunk_word_tolerance,
        )
    )
    cache_source = str(cached.get('text_generation_source', 'cache'))
    cache_matches_mode = (
        cache_source == 'llm'
        if cfg.generation.use_llm_chunk_generation
        else cache_source == 'fallback'
    )
    if errors or not cache_matches_mode:
        return None

    return _new_chunk_state(
        cached_text,
        text_generation_source=cache_source,
        llm_attempted=bool(cached.get('llm_attempted', False)),
        llm_rejected=bool(cached.get('llm_rejected', False)),
        cache_hit=True,
        validation=validation,
    )


def _new_chunk_state(
    final_text: str,
    text_generation_source: str,
    llm_attempted: bool,
    llm_rejected: bool,
    validation: ChunkValidation,
    cache_hit: bool = False,
) -> dict[str, Any]:
    return {
        'final_text': final_text,
        'text_generation_source': text_generation_source,
        'llm_attempted': llm_attempted,
        'llm_rejected': llm_rejected,
        'cache_hit': cache_hit,
        'validation_soft_warnings': list(validation.soft_warnings),
    }


def _finalize_chunk_row(
    cfg: ExperimentCfg,
    fact: dict[str, Any],
    ontology: dict[str, Any],
    index: int,
    state: dict[str, Any],
    should_cache: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    final_text = str(state['final_text'])
    validation = validate_chunk_text(final_text, fact, ontology)
    if validation.hard_errors:
        raise RuntimeError('; '.join(validation.hard_errors))
    state['validation_soft_warnings'] = list(validation.soft_warnings)

    word_count = len(final_text.split())
    word_errors = _word_count_errors(
        word_count,
        min_words=cfg.generation.chunk_min_words,
        max_words=cfg.generation.chunk_max_words,
        tolerance=cfg.generation.chunk_word_tolerance,
    )
    if word_errors:
        raise RuntimeError('; '.join(word_errors))

    cache_entry = None
    if should_cache:
        cache_entry = {
            'cache_version': _CACHE_VERSION,
            'fact_id': fact['fact_id'],
            'text': final_text,
            'text_generation_source': state['text_generation_source'],
            'llm_attempted': state['llm_attempted'],
            'llm_rejected': state['llm_rejected'],
        }

    row = _chunk_row(
        fact=fact,
        chunk_id=f'chunk_{index + 1:07d}',
        final_text=final_text,
        word_count=word_count,
        text_generation_source=state['text_generation_source'],
        llm_attempted=state['llm_attempted'],
        llm_rejected=state['llm_rejected'],
        cache_hit=state['cache_hit'],
        validation_soft_warnings=list(state['validation_soft_warnings']),
    )
    return row, cache_entry


def _row_from_state(index: int, fact: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return _chunk_row(
        fact=fact,
        chunk_id=f'chunk_{index + 1:07d}',
        final_text=str(state['final_text']),
        word_count=len(str(state['final_text']).split()),
        text_generation_source=str(state['text_generation_source']),
        llm_attempted=bool(state['llm_attempted']),
        llm_rejected=bool(state['llm_rejected']),
        cache_hit=bool(state['cache_hit']),
        validation_soft_warnings=list(state.get('validation_soft_warnings', [])),
    )


def _reject_row(fact: dict[str, Any], reason: str, text: str) -> dict[str, Any]:
    return {
        'fact_id': fact['fact_id'],
        'query_id': fact['query_id'],
        'reason': reason,
        'llm_text': text,
    }


def _chunk_row(
    fact: dict[str, Any],
    chunk_id: str,
    final_text: str,
    word_count: int,
    text_generation_source: str,
    llm_attempted: bool,
    llm_rejected: bool,
    cache_hit: bool,
    validation_soft_warnings: list[str],
) -> dict[str, Any]:
    return fact | {
        'chunk_id': chunk_id,
        'text': final_text,
        'approx_words': word_count,
        'text_generation_source': text_generation_source,
        'llm_attempted': llm_attempted,
        'llm_rejected': llm_rejected,
        'generation_cache_hit': cache_hit,
        'validation_soft_warning_count': len(validation_soft_warnings),
        'validation_soft_warnings_json': json.dumps(validation_soft_warnings, sort_keys=True),
    }


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
    run_make_chunks(cfg, paths)
