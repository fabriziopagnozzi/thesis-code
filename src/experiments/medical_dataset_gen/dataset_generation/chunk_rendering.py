from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from random import Random
from typing import Literal

import polars as pl

from experiments.medical_dataset_gen.dataset_generation.chunk_templates import (
    TEMPLATE_DATA,
    ChunkValidation,
    RenderedChunkTemplate,
    patient_descriptor,
    render_chunk_text_template_result,
    squash_whitespaces,
    validate_chunk_text,
)
from experiments.medical_dataset_gen.dataset_generation.prompts_default import (
    MedicalDatasetGenDefaultPrompts,
)
from experiments.medical_dataset_gen.schemas.generation_schemas import (
    ChunkGenerationCacheEntry,
    ChunkRow,
    ChunkState,
    ChunkSurfaceGroup,
    ChunkTextStyle,
    ClinicalFact,
    MedicalOntology,
)
from experiments.medical_dataset_gen.schemas.global_config_schemas import ExperimentCfg
from helpers.ollama_client import generate

_SECTION_HEADER_RE = re.compile(
    r'^\s*(?:brief hospital course|hospital course|discharge summary|'
    r'discharge diagnosis|clinical summary)\s*:\s*',
    re.IGNORECASE,
)


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

    for attempt in range(1, max(1, cfg.generation.llm_config.max_attempts) + 1):
        candidate_text = generate_chunk_text_with_llm(
            fact=fact,
            ontology=ontology,
            llm_name=cfg.generation.llm_config.model_name,
            temperature=cfg.generation.llm_config.temperature,
            num_ctx=cfg.generation.llm_config.num_ctx,
            chunk_min_words=cfg.generation.chunk_pools.chunk_min_words,
            chunk_max_words=cfg.generation.chunk_pools.chunk_max_words,
            text_style=cfg.generation.chunk_text_style,
            revision_feedback=feedback,
        )
        last_text = candidate_text
        word_count = len(candidate_text.split())
        validation = validate_chunk_text(
            candidate_text,
            fact,
            ontology,
            text_style=cfg.generation.chunk_text_style,
        )
        word_errors = word_count_errors(
            word_count,
            min_words=cfg.generation.chunk_pools.chunk_min_words,
            max_words=cfg.generation.chunk_pools.chunk_max_words,
            tolerance=cfg.generation.chunk_pools.chunk_word_tolerance,
        )
        errors = validation.hard_errors + word_errors
        if not errors:
            return candidate_text, []

        last_errors = [f'attempt={attempt}', *errors]
        feedback = '\n'.join(f'- {error}' for error in errors)
        print(f'[chunks] retry {attempt} for {fact.fact_id}: ' + '; '.join(errors))

    return last_text, last_errors


def rewrite_llm_chunk(
    cfg: ExperimentCfg,
    fact: ClinicalFact,
    ontology: MedicalOntology,
    draft_text: str,
) -> tuple[str, list[str]]:
    last_errors = ['empty LLM rewrite']
    feedback: str | None = None

    for attempt in range(1, max(1, cfg.generation.llm_config.max_attempts) + 1):
        candidate_text = rewrite_chunk_text_with_llm(
            draft_text=draft_text,
            fact=fact,
            ontology=ontology,
            llm_name=cfg.generation.llm_config.model_name,
            temperature=cfg.generation.llm_config.temperature,
            num_ctx=cfg.generation.llm_config.num_ctx,
            chunk_min_words=cfg.generation.chunk_pools.chunk_min_words,
            chunk_max_words=cfg.generation.chunk_pools.chunk_max_words,
            text_style=cfg.generation.chunk_text_style,
            revision_feedback=feedback,
        )
        word_count = len(candidate_text.split())
        validation = validate_chunk_text(
            candidate_text,
            fact,
            ontology,
            text_style=cfg.generation.chunk_text_style,
        )
        word_errors = word_count_errors(
            word_count,
            min_words=cfg.generation.chunk_pools.chunk_min_words,
            max_words=cfg.generation.chunk_pools.chunk_max_words,
            tolerance=cfg.generation.chunk_pools.chunk_word_tolerance,
        )
        errors = validation.hard_errors + word_errors
        if not errors:
            return candidate_text, []

        last_errors = [f'attempt={attempt}', *errors]
        feedback = '\n'.join(f'- {error}' for error in errors)
        print(f'[chunks] retry rewrite {attempt} for {fact.fact_id}: ' + '; '.join(errors))

    return draft_text, last_errors


def generate_chunk_text_with_llm(
    *,
    fact: ClinicalFact,
    ontology: MedicalOntology,
    llm_name: str,
    temperature: float,
    num_ctx: int,
    chunk_min_words: int,
    chunk_max_words: int,
    text_style: ChunkTextStyle,
    revision_feedback: str | None = None,
) -> str:
    prompt = MedicalDatasetGenDefaultPrompts.chunk_generation_prompt(
        fact=fact,
        ontology=ontology,
        patient_descriptor=patient_descriptor(fact),
        forbidden_terms=TEMPLATE_DATA.hidden_benchmark_terms,
        required_facts=_required_facts_for_text_style(
            fact,
            ontology,
            text_style=text_style,
        ),
        min_words=chunk_min_words,
        max_words=chunk_max_words,
        revision_feedback=revision_feedback,
    )
    generated = generate(
        prompt,
        model=llm_name,
        system=MedicalDatasetGenDefaultPrompts.chunk_generation_system,
        temperature=temperature,
        num_ctx=num_ctx,
    )
    return _cleanup_generated_text(generated)


def rewrite_chunk_text_with_llm(
    *,
    draft_text: str,
    fact: ClinicalFact,
    ontology: MedicalOntology,
    llm_name: str,
    temperature: float,
    num_ctx: int,
    chunk_min_words: int,
    chunk_max_words: int,
    text_style: ChunkTextStyle,
    revision_feedback: str | None = None,
) -> str:
    prompt = MedicalDatasetGenDefaultPrompts.chunk_rewrite_prompt(
        fact=fact,
        draft_text=draft_text,
        patient_descriptor=patient_descriptor(fact),
        required_facts=_required_facts_for_text_style(
            fact,
            ontology,
            text_style=text_style,
        ),
        forbidden_facts=list(fact.must_not_mention),
        min_words=chunk_min_words,
        max_words=chunk_max_words,
        revision_feedback=revision_feedback,
    )
    generated = generate(
        prompt,
        model=llm_name,
        system=MedicalDatasetGenDefaultPrompts.chunk_rewrite_system,
        temperature=temperature,
        num_ctx=num_ctx,
    )
    return _cleanup_generated_text(generated)


def _cleanup_generated_text(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    fence = re.search(r'```(?:text)?\s*(.*?)```', text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    text = text.strip().strip('"').strip("'").strip()
    lines = [line.strip('- ').strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        text = ' '.join(lines)
    return squash_whitespaces(_SECTION_HEADER_RE.sub('', text))


def _required_facts_for_text_style(
    fact: ClinicalFact,
    ontology: MedicalOntology | None,
    *,
    text_style: ChunkTextStyle,
) -> list[str]:
    if text_style == 'ontology_explicit':
        return list(fact.must_mention)
    omitted = {fact.axis_bin_term.casefold()}
    if ontology is not None:
        omitted.add(ontology.clinical_axes[fact.axis].label.casefold())
    return [item for item in fact.must_mention if item.casefold() not in omitted]


def word_count_ok(word_count: int, min_words: int, max_words: int, tolerance: int) -> bool:
    return (min_words - tolerance) <= word_count <= (max_words + tolerance)


def word_count_errors(word_count: int, min_words: int, max_words: int, tolerance: int) -> list[str]:
    if word_count_ok(word_count, min_words, max_words, tolerance):
        return []
    if word_count < min_words:
        return [f'word_count={word_count} below minimum {min_words} (tolerance {tolerance})']
    return [f'word_count={word_count} above maximum {max_words} (tolerance {tolerance})']


def render_canonical_chunk(
    fact: ClinicalFact,
    ontology: MedicalOntology,
    text_style: ChunkTextStyle = 'semantic_hardened',
) -> RenderedChunkTemplate:
    """Render deterministic chunk prose from the reusable semantic chunk key."""
    rng = Random(_stable_seed(str(fact.chunk_reuse_key or fact.fact_id)))
    return render_chunk_text_template_result(fact, ontology, rng, text_style=text_style)


def render_canonical_chunk_text(
    fact: ClinicalFact,
    ontology: MedicalOntology,
    text_style: ChunkTextStyle = 'semantic_hardened',
    surface_group: ChunkSurfaceGroup | None = None,
) -> str:
    """Render deterministic chunk prose from the reusable semantic chunk key."""
    rng = Random(_stable_seed(str(fact.chunk_reuse_key or fact.fact_id)))
    return render_chunk_text_template_result(
        fact,
        ontology,
        rng,
        text_style=text_style,
        surface_group=surface_group,
    ).text


def new_chunk_state(
    final_text: str,
    text_generation_source: Literal['llm', 'fallback', 'cache'],
    llm_attempted: bool,
    llm_rejected: bool,
    validation: ChunkValidation,
    rendered_template: RenderedChunkTemplate | None = None,
    cache_hit: bool = False,
    cache_hit_kind: Literal['miss', 'fact_id', 'reuse_key'] = 'miss',
) -> ChunkState:
    provenance = rendered_template.provenance if rendered_template is not None else None
    return ChunkState(
        final_text=final_text,
        text_generation_source=text_generation_source,
        llm_attempted=llm_attempted,
        llm_rejected=llm_rejected,
        cache_hit=cache_hit,
        cache_hit_kind=cache_hit_kind,
        validation_soft_warnings=list(validation.soft_warnings),
        outer_template_family=provenance.outer_template_family if provenance is not None else None,
        outer_template_id=provenance.outer_template_id if provenance is not None else None,
        axis_template_family=provenance.axis_template_family if provenance is not None else None,
        axis_template_id=provenance.axis_template_id if provenance is not None else None,
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
    validation = validate_chunk_text(
        final_text,
        fact,
        ontology,
        text_style=cfg.generation.chunk_text_style,
    )
    if validation.hard_errors:
        raise RuntimeError('; '.join(validation.hard_errors))
    state.validation_soft_warnings = list(validation.soft_warnings)

    word_count = len(final_text.split())
    word_errors = word_count_errors(
        word_count,
        min_words=cfg.generation.chunk_pools.chunk_min_words,
        max_words=cfg.generation.chunk_pools.chunk_max_words,
        tolerance=cfg.generation.chunk_pools.chunk_word_tolerance,
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

    row = ChunkRow.from_state(fact, chunk_id=chunk_id(index), state=state)
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


def _stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)
