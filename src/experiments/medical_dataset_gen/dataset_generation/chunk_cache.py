"""Chunk-generation cache helpers.

This module owns cache persistence, cache-key derivation, and cache-hit recovery
for generated and rewritten chunk text.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from experiments.medical_dataset_gen.dataset_generation.chunk_rendering import (
    new_chunk_state,
    word_count_errors,
)
from experiments.medical_dataset_gen.dataset_generation.chunk_templates import validate_chunk_text
from experiments.medical_dataset_gen.dataset_generation.prompts_default import (
    MedicalDatasetGenDefaultPrompts,
)
from experiments.medical_dataset_gen.schemas.generation_schemas import (
    ChunkGenerationCacheEntry,
    ChunkState,
    ClinicalFact,
    MedicalOntology,
)
from experiments.medical_dataset_gen.utils.global_configs import ExperimentCfg

GENERATION_CACHE_VERSION = 10
REWRITE_CACHE_VERSION = 2


@dataclass
class GenerationCache:
    by_fact_id: dict[str, ChunkGenerationCacheEntry]
    by_reuse_key: dict[str, ChunkGenerationCacheEntry]
    loaded_rows: int = 0


def load_generation_cache(paths: list[Path], cache_version: int) -> GenerationCache:
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
                if row.get('cache_version') != cache_version:
                    continue
                text = row.get('text')
                if not text:
                    continue
                cache.loaded_rows += 1
                remember_cache_entry(cache, row)
    return cache


def append_generation_cache(path: Path, row: ChunkGenerationCacheEntry | dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = row.model_dump(mode='json') if isinstance(row, ChunkGenerationCacheEntry) else row
    with open(path, 'a') as f:
        f.write(json.dumps(payload, sort_keys=True) + '\n')
        f.flush()


def remember_cache_entry(
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


def chunk_generation_cache_key(cfg: ExperimentCfg, fact: ClinicalFact) -> str:
    payload: dict[str, object] = {
        'cache_version': GENERATION_CACHE_VERSION,
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
    payload['axis_payload_json'] = fact.axis_payload_json

    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def chunk_rewrite_cache_key(
    cfg: ExperimentCfg,
    fact: ClinicalFact,
    draft_text: str,
) -> str:
    payload: dict[str, object] = {
        'cache_version': REWRITE_CACHE_VERSION,
        'prompt_id': MedicalDatasetGenDefaultPrompts.chunk_rewrite_prompt_id,
        'llm_name': cfg.generation.llm_name,
        'llm_temperature': cfg.generation.llm_temperature,
        'llm_num_ctx': cfg.generation.llm_num_ctx,
        'draft_text_sha256': hashlib.sha256(draft_text.encode()).hexdigest(),
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
        'must_mention': fact.must_mention,
        'must_not_mention': fact.must_not_mention,
    }
    payload['axis_payload_json'] = fact.axis_payload_json

    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def cached_chunk_state(
    cfg: ExperimentCfg,
    fact: ClinicalFact,
    ontology: MedicalOntology,
    cache: GenerationCache,
) -> tuple[ChunkState, str] | None:
    current_cache_key = chunk_generation_cache_key(cfg, fact)
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
        word_count_errors(
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
        new_chunk_state(
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


def cached_rewrite_chunk_state(
    cfg: ExperimentCfg,
    fact: ClinicalFact,
    ontology: MedicalOntology,
    cache: GenerationCache,
    draft_text: str,
) -> tuple[ChunkState, str] | None:
    current_cache_key = chunk_rewrite_cache_key(cfg, fact, draft_text)
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
        word_count_errors(
            len(cached_text.split()),
            min_words=cfg.generation.chunk_min_words,
            max_words=cfg.generation.chunk_max_words,
            tolerance=cfg.generation.chunk_word_tolerance,
        )
    )
    cache_matches_mode = cached.text_generation_source == 'llm'
    if errors or not cache_matches_mode:
        return None

    return (
        new_chunk_state(
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
