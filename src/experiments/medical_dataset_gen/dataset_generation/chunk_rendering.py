from __future__ import annotations

from random import Random

from experiments.medical_dataset_gen.dataset_generation.chunk_templates import (
    ChunkValidation,
    RenderedChunkTemplate,
    render_chunk_text_template_result,
    validate_chunk_text,
)
from experiments.medical_dataset_gen.dataset_generation.schemas import (
    ChunkRow,
    ChunkState,
    ChunkSurfaceGroup,
    ChunkTextStyle,
    ClinicalFact,
    MedicalOntology,
)
from experiments.medical_dataset_gen.utils.deterministic_ids import stable_seed
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg


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
    rng = Random(stable_seed(str(fact.chunk_reuse_key or fact.fact_id)))
    return render_chunk_text_template_result(fact, ontology, rng, text_style=text_style)


def render_canonical_chunk_text(
    fact: ClinicalFact,
    ontology: MedicalOntology,
    text_style: ChunkTextStyle = 'semantic_hardened',
    surface_group: ChunkSurfaceGroup | None = None,
) -> str:
    rng = Random(stable_seed(str(fact.chunk_reuse_key or fact.fact_id)))
    return render_chunk_text_template_result(
        fact,
        ontology,
        rng,
        text_style=text_style,
        surface_group=surface_group,
    ).text


def new_chunk_state(
    final_text: str,
    validation: ChunkValidation,
    rendered_template: RenderedChunkTemplate,
) -> ChunkState:
    provenance = rendered_template.provenance
    return ChunkState(
        final_text=final_text,
        outer_template_family=provenance.outer_template_family,
        outer_template_id=provenance.outer_template_id,
        axis_template_family=provenance.axis_template_family,
        axis_template_id=provenance.axis_template_id,
    )


def finalize_chunk_row(
    cfg: ExperimentCfg,
    fact: ClinicalFact,
    ontology: MedicalOntology,
    index: int,
    state: ChunkState,
) -> ChunkRow:
    final_text = state.final_text
    validation = validate_chunk_text(
        final_text,
        fact,
        ontology,
        text_style=cfg.generation.chunk_text_style,
    )
    if validation.hard_errors:
        raise RuntimeError('; '.join(validation.hard_errors))

    word_count = len(final_text.split())
    word_errors = word_count_errors(
        word_count,
        min_words=cfg.generation.chunk_pools.chunk_min_words,
        max_words=cfg.generation.chunk_pools.chunk_max_words,
        tolerance=cfg.generation.chunk_pools.chunk_word_tolerance,
    )
    if word_errors:
        raise RuntimeError('; '.join(word_errors))

    return ChunkRow.from_state(fact, chunk_id=chunk_id(index), state=state)


def chunk_id(index: int) -> str:
    return f'chunk_{index + 1:07d}'
