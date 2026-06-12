import hashlib
import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Literal

import polars as pl
from tqdm import tqdm

from experiments.medical_dataset_gen.generation.ontology import load_ontology
from experiments.medical_dataset_gen.generation.schemas import (
    ChunkGenerationCacheEntry,
    ChunkRow,
    ChunkState,
    ClinicalFact,
    MedicalOntology,
)
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

_CACHE_VERSION = 9


@dataclass
class GenerationCache:
    by_fact_id: dict[str, ChunkGenerationCacheEntry]
    by_reuse_key: dict[str, ChunkGenerationCacheEntry]
    loaded_rows: int = 0


def run_make_chunks(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    ontology = load_ontology(cfg)
    facts = read_parquet(paths, 'clinical_facts')
    fact_rows = [ClinicalFact.model_validate(row) for row in facts.iter_rows(named=True)]
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

    local_cache_path = paths.experiment_dir / 'chunk_generation_cache.jsonl'
    shared_cache_path = paths.root / '_cache' / 'chunk_generation_cache.jsonl'
    cache = _load_generation_cache([shared_cache_path, local_cache_path])
    if cache.loaded_rows:
        print(
            f'[chunks] loaded {cache.loaded_rows:,} cached chunk generation row(s) '
            f'({len(cache.by_fact_id):,} exact fact keys, {len(cache.by_reuse_key):,} reusable keys)'
        )

    if cfg.generation.use_llm_chunk_generation:
        rows, rejects, failed_queries = _render_chunks_grouped_llm(
            cfg=cfg,
            facts=fact_rows,
            ontology=ontology,
            cache=cache,
            local_cache_path=local_cache_path,
            shared_cache_path=shared_cache_path,
        )
    else:
        rows, rejects, failed_queries = _render_chunks_sequential(
            cfg=cfg,
            paths=paths,
            facts=fact_rows,
            ontology=ontology,
            cache=cache,
            cache_path=local_cache_path,
            rng=rng,
        )

    chunks = pl.DataFrame([row.model_dump(mode='python') for row in rows])
    write_parquet(paths, 'chunks', chunks)

    reject_df = (
        pl.DataFrame(rejects)
        if rejects
        else pl.DataFrame(
            schema={
                'fact_id': pl.String,
                'query_id': pl.String,
                'reason': pl.String,
                'llm_text': pl.String,
            }
        )
    )
    write_parquet(paths, 'generation_rejects', reject_df)
    soft_warning_count = sum(row.validation_soft_warning_count for row in rows)
    chunks_with_soft_warnings = sum(
        1 for row in rows if row.validation_soft_warning_count > 0
    )
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


def _write_rejects(paths: MedicalDatasetGenPaths, rejects: list[dict[str, object]]) -> None:
    reject_df = (
        pl.DataFrame(rejects)
        if rejects
        else pl.DataFrame(
            schema={
                'fact_id': pl.String,
                'query_id': pl.String,
                'reason': pl.String,
                'llm_text': pl.String,
            }
        )
    )
    write_parquet(paths, 'generation_rejects', reject_df)


def _render_chunks_sequential(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    facts: list[ClinicalFact],
    ontology: MedicalOntology,
    cache: GenerationCache,
    cache_path: Path,
    rng: Random,
) -> tuple[list[ChunkRow], list[dict[str, object]], set[str]]:
    rows: list[ChunkRow] = []
    rejects: list[dict[str, object]] = []
    failed_queries: set[str] = set()

    for i, fact in tqdm(
        enumerate(facts),
        total=len(facts),
        desc='Rendering chunks',
        dynamic_ncols=True,
    ):
        if fact.query_id in failed_queries:
            continue
        cached = _cached_chunk_state(cfg, fact, ontology, cache)
        if cached is not None:
            rows.append(_row_from_state(i, fact, cached[0]))
            continue

        if cfg.generation.use_llm_chunk_generation:
            final_text, attempt_errors = _generate_llm_chunk(cfg=cfg, fact=fact, ontology=ontology)
            if attempt_errors:
                rejects.append(_reject_row(fact, '; '.join(attempt_errors), final_text))
                failed_queries.add(fact.query_id)
                print(
                    f'[chunks] dropping query {fact.query_id} after failed chunk {fact.fact_id}: '
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
            rejects.append(_reject_row(fact, str(exc), state.final_text))
            failed_queries.add(fact.query_id)
            print(
                f'[chunks] dropping query {fact.query_id} after final validation failure in {fact.fact_id}: {exc}'
            )
            continue

        if cache_entry is not None:
            _append_generation_cache(cache_path, cache_entry)
            _remember_cache_entry(cache, cache_entry)
        rows.append(row)

    kept_rows = [row for row in rows if str(row['query_id']) not in failed_queries]
    return kept_rows, rejects, failed_queries


def _render_chunks_grouped_llm(
    cfg: ExperimentCfg,
    facts: list[ClinicalFact],
    ontology: MedicalOntology,
    cache: GenerationCache,
    local_cache_path: Path,
    shared_cache_path: Path,
) -> tuple[list[ChunkRow], list[dict[str, object]], set[str]]:
    rows: list[ChunkRow | None] = [None] * len(facts)
    rejects: list[dict[str, object]] = []
    failed_queries: set[str] = set()
    missing_groups: dict[str, list[tuple[int, ClinicalFact]]] = {}
    exact_cache_hits = 0
    reusable_cache_hits = 0

    for idx, fact in enumerate(facts):
        cached = _cached_chunk_state(cfg, fact, ontology, cache)
        if cached is not None:
            state, hit_kind = cached
            rows[idx] = _row_from_state(idx, fact, state)
            if hit_kind == 'fact_id':
                exact_cache_hits += 1
            else:
                reusable_cache_hits += 1
            continue
        cache_key = _chunk_generation_cache_key(cfg, fact)
        missing_groups.setdefault(cache_key, []).append((idx, fact))

    cache_hits = exact_cache_hits + reusable_cache_hits
    facts_to_generate = len(facts) - cache_hits
    duplicate_jobs_saved = max(0, facts_to_generate - len(missing_groups))
    print(
        f'[chunks] grouped LLM generation: facts={len(facts):,}, cache_hits={cache_hits:,} '
        f'(exact={exact_cache_hits:,}, reusable={reusable_cache_hits:,}), '
        f'unique_llm_jobs={len(missing_groups):,}, duplicate_jobs_saved={duplicate_jobs_saved:,}'
    )

    group_items = list(missing_groups.items())
    with tqdm(total=len(facts), desc='Rendering chunks', dynamic_ncols=True) as pbar:
        if cache_hits:
            pbar.update(cache_hits)
        if cfg.generation.llm_workers > 1:
            _generate_missing_groups_parallel(
                cfg=cfg,
                ontology=ontology,
                cache=cache,
                local_cache_path=local_cache_path,
                shared_cache_path=shared_cache_path,
                group_items=group_items,
                rows=rows,
                rejects=rejects,
                failed_queries=failed_queries,
                pbar=pbar,
            )
        else:
            _generate_missing_groups_sequential(
                cfg=cfg,
                ontology=ontology,
                cache=cache,
                local_cache_path=local_cache_path,
                shared_cache_path=shared_cache_path,
                group_items=group_items,
                rows=rows,
                rejects=rejects,
                failed_queries=failed_queries,
                pbar=pbar,
            )

    materialized_rows: list[ChunkRow] = [
        row for row in rows if row is not None and str(row['query_id']) not in failed_queries
    ]
    return materialized_rows, rejects, failed_queries


def _generate_missing_groups_sequential(
    cfg: ExperimentCfg,
    ontology: MedicalOntology,
    cache: GenerationCache,
    local_cache_path: Path,
    shared_cache_path: Path,
    group_items: list[tuple[str, list[tuple[int, ClinicalFact]]]],
    rows: list[ChunkRow | None],
    rejects: list[dict[str, object]],
    failed_queries: set[str],
    pbar: tqdm,
) -> None:
    for cache_key, group in group_items:
        active = _active_group(group, failed_queries)
        if not active:
            pbar.update(len(group))
            continue
        final_text, attempt_errors = _generate_llm_chunk(
            cfg=cfg, fact=active[0][1], ontology=ontology
        )
        _materialize_generated_group(
            cfg=cfg,
            ontology=ontology,
            cache=cache,
            local_cache_path=local_cache_path,
            shared_cache_path=shared_cache_path,
            cache_key=cache_key,
            group=group,
            rows=rows,
            rejects=rejects,
            failed_queries=failed_queries,
            final_text=final_text,
            attempt_errors=attempt_errors,
            pbar=pbar,
        )


def _generate_missing_groups_parallel(
    cfg: ExperimentCfg,
    ontology: MedicalOntology,
    cache: GenerationCache,
    local_cache_path: Path,
    shared_cache_path: Path,
    group_items: list[tuple[str, list[tuple[int, ClinicalFact]]]],
    rows: list[ChunkRow | None],
    rejects: list[dict[str, object]],
    failed_queries: set[str],
    pbar: tqdm,
) -> None:
    pending: dict[
        Future[tuple[str, list[str]]],
        tuple[str, list[tuple[int, ClinicalFact]]],
    ] = {}
    next_idx = 0
    max_in_flight = max(cfg.generation.llm_workers * 2, cfg.generation.llm_workers)
    executor = ThreadPoolExecutor(
        max_workers=cfg.generation.llm_workers, thread_name_prefix='mdg-llm'
    )

    try:
        while next_idx < len(group_items) or pending:
            while next_idx < len(group_items) and len(pending) < max_in_flight:
                cache_key, group = group_items[next_idx]
                next_idx += 1
                active = _active_group(group, failed_queries)
                if not active:
                    pbar.update(len(group))
                    continue
                future = executor.submit(_generate_llm_chunk, cfg, active[0][1], ontology)
                pending[future] = (cache_key, group)

            if not pending:
                continue

            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                cache_key, group = pending.pop(future)
                final_text, attempt_errors = future.result()
                _materialize_generated_group(
                    cfg=cfg,
                    ontology=ontology,
                    cache=cache,
                    local_cache_path=local_cache_path,
                    shared_cache_path=shared_cache_path,
                    cache_key=cache_key,
                    group=group,
                    rows=rows,
                    rejects=rejects,
                    failed_queries=failed_queries,
                    final_text=final_text,
                    attempt_errors=attempt_errors,
                    pbar=pbar,
                )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _active_group(
    group: list[tuple[int, ClinicalFact]],
    failed_queries: set[str],
) -> list[tuple[int, ClinicalFact]]:
    return [(idx, fact) for idx, fact in group if fact.query_id not in failed_queries]


def _materialize_generated_group(
    cfg: ExperimentCfg,
    ontology: MedicalOntology,
    cache: GenerationCache,
    local_cache_path: Path,
    shared_cache_path: Path,
    cache_key: str,
    group: list[tuple[int, ClinicalFact]],
    rows: list[ChunkRow | None],
    rejects: list[dict[str, object]],
    failed_queries: set[str],
    final_text: str,
    attempt_errors: list[str],
    pbar: tqdm,
) -> None:
    active = _active_group(group, failed_queries)
    if not active:
        pbar.update(len(group))
        return

    if attempt_errors:
        reason = '; '.join(attempt_errors)
        for _, fact in active:
            rejects.append(_reject_row(fact, reason, final_text))
            failed_queries.add(fact.query_id)
            print(
                f'[chunks] dropping query {fact.query_id} after failed chunk {fact.fact_id}: '
                + reason
            )
        pbar.update(len(group))
        return

    local_cache_written = cache_key in cache.by_reuse_key
    shared_cache_written = cache_key in cache.by_reuse_key
    for idx, fact in active:
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
            failed_queries.add(fact.query_id)
            print(
                f'[chunks] dropping query {fact.query_id} after final validation failure in '
                f'{fact.fact_id}: {exc}'
            )
            continue

        if cache_entry is not None:
            _remember_cache_entry(cache, cache_entry)
            if not local_cache_written:
                _append_generation_cache(local_cache_path, cache_entry)
                local_cache_written = True
            if not shared_cache_written:
                _append_generation_cache(shared_cache_path, cache_entry)
                shared_cache_written = True
        rows[idx] = row
    pbar.update(len(group))


def _load_generation_cache(paths: list[Path]) -> GenerationCache:
    cache = GenerationCache(by_fact_id={}, by_reuse_key={})

    for path in paths:
        if not path.exists():
            continue
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
                text = row.get('text')
                if not text:
                    continue
                cache.loaded_rows += 1
                _remember_cache_entry(cache, row)
    return cache


def _append_generation_cache(
    path: Path, row: ChunkGenerationCacheEntry | dict[str, object]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        row.model_dump(mode='json') if isinstance(row, ChunkGenerationCacheEntry) else row
    )
    with open(path, 'a') as f:
        f.write(json.dumps(payload, sort_keys=True) + '\n')
        f.flush()


def _remember_cache_entry(
    cache: GenerationCache, row: ChunkGenerationCacheEntry | dict[str, object]
) -> None:
    entry = (
        row
        if isinstance(row, ChunkGenerationCacheEntry)
        else ChunkGenerationCacheEntry.model_validate(row)
    )
    fact_id = entry.fact_id
    if fact_id:
        cache.by_fact_id[str(fact_id)] = entry
    reuse_key = entry.chunk_generation_cache_key
    if reuse_key:
        cache.by_reuse_key[str(reuse_key)] = entry


def _generate_llm_chunk(
    cfg: ExperimentCfg,
    fact: ClinicalFact,
    ontology: MedicalOntology,
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


def _word_count_errors(
    word_count: int, min_words: int, max_words: int, tolerance: int
) -> list[str]:
    if _word_count_ok(word_count, min_words, max_words, tolerance):
        return []
    if word_count < min_words:
        return [f'word_count={word_count} below minimum {min_words} (tolerance {tolerance})']
    return [f'word_count={word_count} above maximum {max_words} (tolerance {tolerance})']


def _chunk_generation_cache_key(cfg: ExperimentCfg, fact: ClinicalFact) -> str:
    payload: dict[str, object] = {
        'cache_version': _CACHE_VERSION,
        'source': 'llm' if cfg.generation.use_llm_chunk_generation else 'fallback',
        'fact_chunk_reuse_key': fact.chunk_reuse_key,
        'condition_id': fact.condition_id,
        'condition_display': fact.condition_display,
        'subgroup_id': fact.subgroup_id,
        'subgroup_label': fact.subgroup_label,
        'subgroup_axis': fact.subgroup_axis,
        'axis': fact.axis,
        'value_bin': fact.value_bin,
        'patient_age': fact.patient_age,
        'patient_sex': fact.patient_sex,
        'clinical_subgroup_phrase': fact.clinical_subgroup_phrase,
        'note_style': fact.note_style,
        'chunk_min_words': cfg.generation.chunk_min_words,
        'chunk_max_words': cfg.generation.chunk_max_words,
        'chunk_word_tolerance': cfg.generation.chunk_word_tolerance,
    }
    if cfg.generation.use_llm_chunk_generation:
        payload.update(
            {
                'llm_name': cfg.generation.llm_name,
                'llm_temperature': cfg.generation.llm_temperature,
                'llm_num_ctx': cfg.generation.llm_num_ctx,
            }
        )
    if fact.axis == 'treatment_duration':
        payload.update(
            {
                'duration_days': fact.duration_days,
                'treatment': fact.treatment,
            }
        )
    else:
        payload['rehab_outcome'] = fact.rehab_outcome

    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _cached_chunk_state(
    cfg: ExperimentCfg,
    fact: ClinicalFact,
    ontology: MedicalOntology,
    cache: GenerationCache,
) -> tuple[ChunkState, str] | None:
    current_cache_key = _chunk_generation_cache_key(cfg, fact)
    hit_kind: Literal['fact_id', 'reuse_key'] = 'fact_id'
    cached = cache.by_fact_id.get(fact.fact_id)
    if cached and cached.chunk_generation_cache_key not in {None, current_cache_key}:
        cached = None
    if not cached:
        hit_kind = 'reuse_key'
        cached = cache.by_reuse_key.get(current_cache_key)
    if not cached:
        return None

    cached_text = cached.text
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
    cache_source = cached.text_generation_source
    cache_matches_mode = (
        cache_source == 'llm'
        if cfg.generation.use_llm_chunk_generation
        else cache_source == 'fallback'
    )
    if errors or not cache_matches_mode:
        return None

    return (
        _new_chunk_state(
            cached_text,
            text_generation_source='cache',
            llm_attempted=cached.llm_attempted,
            llm_rejected=cached.llm_rejected,
            cache_hit=True,
            cache_hit_kind=hit_kind,
            validation=validation,
        ),
        hit_kind,
    )


def _new_chunk_state(
    final_text: str,
    text_generation_source: Literal['llm', 'fallback', 'cache'],
    llm_attempted: bool,
    llm_rejected: bool,
    validation: ChunkValidation,
    cache_hit: bool = False,
    cache_hit_kind: Literal['miss', 'fact_id', 'reuse_key'] = 'miss',
) -> ChunkState:
    return ChunkState(
        final_text=final_text,
        text_generation_source=text_generation_source,
        llm_attempted=llm_attempted,
        llm_rejected=llm_rejected,
        cache_hit=cache_hit,
        cache_hit_kind=cache_hit_kind,
        validation_soft_warnings=list(validation.soft_warnings),
    )


def _finalize_chunk_row(
    cfg: ExperimentCfg,
    fact: ClinicalFact,
    ontology: MedicalOntology,
    index: int,
    state: ChunkState,
    should_cache: bool,
) -> tuple[ChunkRow, ChunkGenerationCacheEntry | None]:
    final_text = state.final_text
    validation = validate_chunk_text(final_text, fact, ontology)
    if validation.hard_errors:
        raise RuntimeError('; '.join(validation.hard_errors))
    state.validation_soft_warnings = list(validation.soft_warnings)

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
        cache_text_source: Literal['llm', 'fallback'] = (
            'llm' if state.text_generation_source == 'cache' else state.text_generation_source
        )
        cache_entry = ChunkGenerationCacheEntry(
            cache_version=_CACHE_VERSION,
            fact_id=fact.fact_id,
            fact_chunk_reuse_key=fact.chunk_reuse_key,
            chunk_generation_cache_key=_chunk_generation_cache_key(cfg, fact),
            text=final_text,
            text_generation_source=cache_text_source,
            llm_attempted=state.llm_attempted,
            llm_rejected=state.llm_rejected,
        )

    row = _chunk_row(
        fact=fact,
        chunk_id=f'chunk_{index + 1:07d}',
        final_text=final_text,
        word_count=word_count,
        text_generation_source=state.text_generation_source,
        llm_attempted=state.llm_attempted,
        llm_rejected=state.llm_rejected,
        cache_hit=state.cache_hit,
        cache_hit_kind=state.cache_hit_kind,
        validation_soft_warnings=list(state.validation_soft_warnings),
    )
    return row, cache_entry


def _row_from_state(index: int, fact: ClinicalFact, state: ChunkState) -> ChunkRow:
    return _chunk_row(
        fact=fact,
        chunk_id=f'chunk_{index + 1:07d}',
        final_text=state.final_text,
        word_count=len(state.final_text.split()),
        text_generation_source=state.text_generation_source,
        llm_attempted=state.llm_attempted,
        llm_rejected=state.llm_rejected,
        cache_hit=state.cache_hit,
        cache_hit_kind=state.cache_hit_kind,
        validation_soft_warnings=list(state.validation_soft_warnings),
    )


def _reject_row(fact: ClinicalFact, reason: str, text: str) -> dict[str, object]:
    return {
        'fact_id': fact.fact_id,
        'query_id': fact.query_id,
        'reason': reason,
        'llm_text': text,
    }


def _chunk_row(
    fact: ClinicalFact,
    chunk_id: str,
    final_text: str,
    word_count: int,
    text_generation_source: Literal['llm', 'fallback', 'cache'],
    llm_attempted: bool,
    llm_rejected: bool,
    cache_hit: bool,
    cache_hit_kind: Literal['miss', 'fact_id', 'reuse_key'],
    validation_soft_warnings: list[str],
) -> ChunkRow:
    return ChunkRow(
        **fact.model_dump(mode='python'),
        chunk_id=chunk_id,
        text=final_text,
        approx_words=word_count,
        text_generation_source=text_generation_source,
        llm_attempted=llm_attempted,
        llm_rejected=llm_rejected,
        generation_cache_hit=cache_hit,
        generation_cache_hit_kind=cache_hit_kind,
        validation_soft_warning_count=len(validation_soft_warnings),
        validation_soft_warnings_json=json.dumps(validation_soft_warnings, sort_keys=True),
    )


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
