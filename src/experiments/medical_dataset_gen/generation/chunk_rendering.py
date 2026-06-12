from collections.abc import Callable
from typing import Literal

import polars as pl

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
    maybe_rewrite_chunk_text,
    validate_chunk_text,
)
from experiments.medical_dataset_gen.global_configs import ExperimentCfg


def rejects_frame(rejects: list[dict[str, object]]) -> pl.DataFrame:
    return (
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


def generate_llm_chunk(
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
        word_errors = word_count_errors(
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


def rewrite_llm_chunk(
    cfg: ExperimentCfg,
    fact: ClinicalFact,
    ontology: MedicalOntology,
    draft_text: str,
) -> tuple[str, list[str]]:
    last_errors = ['empty LLM rewrite']
    feedback: str | None = None

    for attempt in range(1, max(1, cfg.generation.llm_chunk_max_attempts) + 1):
        candidate_text = maybe_rewrite_chunk_text(
            draft_text=draft_text,
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
        word_count = len(candidate_text.split())
        validation = validate_chunk_text(candidate_text, fact, ontology)
        word_errors = word_count_errors(
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
        print(f'[chunks] retry rewrite {attempt} for {fact.fact_id}: ' + '; '.join(errors))

    return draft_text, last_errors


def word_count_ok(word_count: int, min_words: int, max_words: int, tolerance: int) -> bool:
    return (min_words - tolerance) <= word_count <= (max_words + tolerance)


def word_count_errors(word_count: int, min_words: int, max_words: int, tolerance: int) -> list[str]:
    if word_count_ok(word_count, min_words, max_words, tolerance):
        return []
    if word_count < min_words:
        return [f'word_count={word_count} below minimum {min_words} (tolerance {tolerance})']
    return [f'word_count={word_count} above maximum {max_words} (tolerance {tolerance})']


def new_chunk_state(
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


def finalize_chunk_row(
    cfg: ExperimentCfg,
    fact: ClinicalFact,
    ontology: MedicalOntology,
    index: int,
    state: ChunkState,
    should_cache: bool,
    cache_key: str | None = None,
    cache_version: int = 0,
    cache_key_fn: Callable[[ExperimentCfg, ClinicalFact], str] | None = None,
) -> tuple[ChunkRow, ChunkGenerationCacheEntry | None]:
    final_text = state.final_text
    validation = validate_chunk_text(final_text, fact, ontology)
    if validation.hard_errors:
        raise RuntimeError('; '.join(validation.hard_errors))
    state.validation_soft_warnings = list(validation.soft_warnings)

    word_count = len(final_text.split())
    word_errors = word_count_errors(
        word_count,
        min_words=cfg.generation.chunk_min_words,
        max_words=cfg.generation.chunk_max_words,
        tolerance=cfg.generation.chunk_word_tolerance,
    )
    if word_errors:
        raise RuntimeError('; '.join(word_errors))

    cache_entry = None
    if should_cache:
        if cache_key is None:
            if cache_key_fn is None:
                raise ValueError('cache_key_fn is required when cache_key is not provided')
            cache_key = cache_key_fn(cfg, fact)
        cache_text_source: Literal['llm', 'fallback'] = (
            'llm' if state.text_generation_source == 'cache' else state.text_generation_source
        )
        cache_entry = ChunkGenerationCacheEntry(
            cache_version=cache_version,
            fact_id=fact.fact_id,
            fact_chunk_reuse_key=fact.chunk_reuse_key,
            chunk_generation_cache_key=cache_key,
            text=final_text,
            text_generation_source=cache_text_source,
            llm_attempted=state.llm_attempted,
            llm_rejected=state.llm_rejected,
        )

    row = ChunkRow.from_fact(
        fact,
        chunk_id=chunk_id(index),
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


def row_from_state(index: int, fact: ClinicalFact, state: ChunkState) -> ChunkRow:
    return ChunkRow.from_state(fact, chunk_id=chunk_id(index), state=state)


def reject_row(fact: ClinicalFact, reason: str, text: str) -> dict[str, object]:
    return {
        'fact_id': fact.fact_id,
        'query_id': fact.query_id,
        'reason': reason,
        'llm_text': text,
    }


def chunk_id(index: int) -> str:
    return f'chunk_{index + 1:07d}'
