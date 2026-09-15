from __future__ import annotations

from dataclasses import dataclass
from random import Random

from experiments.medical_dataset_gen.dataset_generation.chunk_templates import (
    RenderedChunkTemplate,
    render_chunk_text_template_result,
    validate_chunk_text,
)
from experiments.medical_dataset_gen.dataset_generation.schemas import (
    ChunkRow,
    ChunkTextStyle,
    ClinicalFact,
    MedicalOntology,
)
from experiments.medical_dataset_gen.utils.deterministic_ids import stable_seed
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg


@dataclass(frozen=True)
class ChunkState:
    final_text: str
    outer_template_family: str | None = None
    outer_template_id: str | None = None
    axis_template_family: str | None = None
    axis_template_id: str | None = None


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


def new_chunk_state(
    final_text: str,
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

    return ChunkRow(
        **fact.model_dump(mode='python'),
        chunk_id=chunk_id(index),
        text=state.final_text,
        approx_words=len(state.final_text.split()),
        outer_template_family=state.outer_template_family,
        outer_template_id=state.outer_template_id,
        axis_template_family=state.axis_template_family,
        axis_template_id=state.axis_template_id,
    )


def chunk_id(index: int) -> str:
    return f'chunk_{index + 1:07d}'
