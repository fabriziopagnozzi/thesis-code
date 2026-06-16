from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

type FacetAxis = Literal['treatment_duration', 'rehab_outcome']
type ClusterRole = Literal['dominant_gold', 'complementary_gold', 'hard_distractor']
type QueryType = Literal['subgroup_comparison', 'outcome_synthesis']
type Split = Literal['train', 'validation', 'test']
type PatientSex = Literal['female', 'male']
type SubgroupAxis = Literal['demographic', 'comorbidity']
type SubgroupKey = str  # todo: enforce literals maybe
type ConditionKey = str
type ClinicalAxisKey = str


class BenchmarkModel(BaseModel):
    model_config = ConfigDict(extra='forbid')

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)

    def get(self, key: str, default: object = None) -> object:
        return getattr(self, key, default)


class ConditionOntology(BenchmarkModel):
    display: str
    terms: list[str]
    treatments: list[str]
    duration_treatments: list[str] | None = None
    duration_days: dict[str, tuple[int, int]]
    rehab_outcomes: dict[str, list[str]]


class SubgroupOntology(BenchmarkModel):
    axis: SubgroupAxis
    label: str
    field: str
    value: str
    aliases: list[str] = Field(default_factory=list)

    def prefixed_fields(self, prefix: str, subgroup_id: str) -> dict[str, object]:
        return {
            f'{prefix}_id': subgroup_id,
            f'{prefix}_label': self.label,
            f'{prefix}_axis': self.axis,
            f'{prefix}_field': self.field,
            f'{prefix}_value': self.value,
        }


class ClinicalAxisOntology(BenchmarkModel):
    label: str
    exact_terms: list[str]
    synonym_terms: list[str]
    bins: list[str]


class MedicalOntology(BenchmarkModel):
    conditions: dict[ConditionKey, ConditionOntology]
    subgroups: dict[SubgroupKey, SubgroupOntology]
    clinical_axes: dict[ClinicalAxisKey, ClinicalAxisOntology]


class QueryPlanFacet(BenchmarkModel):
    facet_id: str
    condition_id: str
    condition_display: str
    subgroup_id: str
    subgroup_label: str
    subgroup_axis: SubgroupAxis
    subgroup_field: str
    subgroup_value: str
    axis: FacetAxis
    value_bin: str
    cluster_id: str
    cluster_role: ClusterRole
    target_gold_chunks: int


class QueryLogicalForm(BenchmarkModel):
    query_type: QueryType = Field(alias='type')
    condition: str
    subgroups: list[str]
    axes: list[FacetAxis]
    facets: list[str]
    dominant_facet_id: str

    model_config = ConfigDict(extra='forbid', populate_by_name=True)


class QueryPlanSpec(BenchmarkModel):
    query_type: QueryType
    condition_key: str
    condition_display: str
    subgroup_a_id: str
    subgroup_a: SubgroupOntology
    subgroup_b_id: str
    subgroup_b: SubgroupOntology


class QueryPlan(BenchmarkModel):
    query_id: str
    plan_seed: int
    split: Split
    query_type: QueryType
    template_id: str
    condition_id: str
    condition_display: str
    subgroup_a_id: str
    subgroup_a_label: str
    subgroup_a_axis: SubgroupAxis
    subgroup_a_field: str
    subgroup_a_value: str
    subgroup_b_id: str
    subgroup_b_label: str
    subgroup_b_axis: SubgroupAxis
    subgroup_b_field: str
    subgroup_b_value: str
    dominant_facet_id: str
    n_facets: int
    gold_chunks_total: int
    distractor_chunks: int
    facets: list[QueryPlanFacet]
    logical_form: QueryLogicalForm

    @model_validator(mode='before')
    @classmethod
    def _parse_legacy_row(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        row = dict(data)
        if 'facets' not in row and 'facets_json' in row:
            row['facets'] = [
                QueryPlanFacet.model_validate(item) for item in json.loads(row.pop('facets_json'))
            ]
        if 'logical_form' not in row and 'logical_form_json' in row:
            row['logical_form'] = QueryLogicalForm.model_validate(
                json.loads(row.pop('logical_form_json'))
            )
        return row

    def _json_columns(self) -> tuple[str, str]:
        facets_json = json.dumps(
            [facet.model_dump(mode='json') for facet in self.facets], sort_keys=True
        )
        logical_form_json = json.dumps(
            self.logical_form.model_dump(mode='json', by_alias=True), sort_keys=True
        )
        return facets_json, logical_form_json

    def to_row(self) -> dict[str, object]:
        row = self.model_dump(mode='python', exclude={'facets', 'logical_form'})
        row['facets_json'], row['logical_form_json'] = self._json_columns()
        return row

    def to_query_row(self, query_text: str) -> dict[str, object]:
        facets_json, logical_form_json = self._json_columns()
        return {
            'query_id': self.query_id,
            'query_type': self.query_type,
            'template_id': self.template_id,
            'condition_id': self.condition_id,
            'condition_display': self.condition_display,
            'subgroup_a_id': self.subgroup_a_id,
            'subgroup_a_label': self.subgroup_a_label,
            'subgroup_b_id': self.subgroup_b_id,
            'subgroup_b_label': self.subgroup_b_label,
            'dominant_facet_id': self.dominant_facet_id,
            'split': self.split,
            'n_facets': self.n_facets,
            'facets_json': facets_json,
            'logical_form_json': logical_form_json,
            'query_text': query_text,
        }

    def to_answer_row(
        self,
        *,
        answer_text: str,
        facet_summaries: dict[str, str],
        facet_answer_objects: list[dict[str, object]],
        supporting_fact_ids: list[str],
    ) -> dict[str, object]:
        return {
            'query_id': self.query_id,
            'answer_text': answer_text,
            'facet_summaries_json': json.dumps(facet_summaries, sort_keys=True),
            'answer_facts_json': json.dumps(facet_answer_objects, sort_keys=True),
            'supporting_fact_ids_json': json.dumps(supporting_fact_ids, sort_keys=True),
            'supporting_facet_ids_json': json.dumps(
                [facet.facet_id for facet in self.facets], sort_keys=True
            ),
        }


class ClinicalFact(BenchmarkModel):
    query_id: str
    source_query_id: str
    fact_id: str
    chunk_reuse_key: str
    facet_id: str | None
    target_facet_id: str
    cluster_id: str
    cluster_role: ClusterRole
    condition_id: str
    condition_display: str
    subgroup_id: str
    subgroup_label: str
    subgroup_axis: SubgroupAxis
    subgroup_field: str
    subgroup_value: str
    axis: FacetAxis
    value_bin: str
    duration_days: int | None
    treatment: str | None
    rehab_outcome: str | None
    is_gold: bool
    distractor_type: str | None
    admission_id: str
    patient_id: str
    patient_age: int
    patient_sex: PatientSex
    clinical_subgroup_phrase: str
    note_style: str
    split: Split
    must_mention: list[str] = Field(default_factory=list)
    must_not_mention: list[str] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def _parse_legacy_row(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        row = dict(data)
        if 'must_mention' not in row and 'must_mention_json' in row:
            row['must_mention'] = json.loads(row.pop('must_mention_json'))
        if 'must_not_mention' not in row and 'must_not_mention_json' in row:
            row['must_not_mention'] = json.loads(row.pop('must_not_mention_json'))
        return row

    @model_validator(mode='after')
    def _validate_axis_specific_fields(self) -> ClinicalFact:
        if self.axis == 'treatment_duration':
            if self.duration_days is None:
                raise ValueError('treatment duration facts require duration_days')
            if self.treatment is None:
                raise ValueError('treatment duration facts require treatment')
            if self.rehab_outcome is not None:
                raise ValueError('treatment duration facts must not include rehab_outcome')
        elif self.axis == 'rehab_outcome':
            if self.rehab_outcome is None:
                raise ValueError('rehab outcome facts require rehab_outcome')
            if self.duration_days is not None:
                raise ValueError('rehab outcome facts must not include duration_days')
            if self.treatment is not None:
                raise ValueError('rehab outcome facts must not include treatment')
        else:
            raise ValueError(f'unsupported axis: {self.axis}')

        if self.is_gold and self.facet_id is None:
            raise ValueError('gold facts require facet_id')
        if not self.is_gold and self.facet_id is not None:
            raise ValueError('distractor facts must not have facet_id')
        return self


class ChunkGenerationCacheEntry(BenchmarkModel):
    cache_version: int
    fact_id: str
    fact_chunk_reuse_key: str | None = None
    chunk_generation_cache_key: str
    text: str
    text_generation_source: Literal['llm', 'fallback']
    llm_attempted: bool
    llm_rejected: bool


@dataclass
class ChunkState:
    final_text: str
    text_generation_source: Literal['llm', 'fallback', 'cache']
    llm_attempted: bool
    llm_rejected: bool
    cache_hit: bool = False
    cache_hit_kind: Literal['miss', 'fact_id', 'reuse_key'] = 'miss'
    validation_soft_warnings: list[str] = field(default_factory=list)


class ChunkRow(ClinicalFact):
    chunk_id: str
    text: str
    approx_words: int
    text_generation_source: Literal['llm', 'fallback', 'cache']
    llm_attempted: bool
    llm_rejected: bool
    generation_cache_hit: bool
    generation_cache_hit_kind: Literal['miss', 'fact_id', 'reuse_key']
    validation_soft_warning_count: int
    validation_soft_warnings_json: str

    @classmethod
    def from_fact(
        cls,
        fact: ClinicalFact,
        *,
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
        return cls(
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

    @classmethod
    def from_state(cls, fact: ClinicalFact, *, chunk_id: str, state: ChunkState) -> ChunkRow:
        return cls.from_fact(
            fact,
            chunk_id=chunk_id,
            final_text=state.final_text,
            word_count=len(state.final_text.split()),
            text_generation_source=state.text_generation_source,
            llm_attempted=state.llm_attempted,
            llm_rejected=state.llm_rejected,
            cache_hit=state.cache_hit,
            cache_hit_kind=state.cache_hit_kind,
            validation_soft_warnings=list(state.validation_soft_warnings),
        )


class RehabClosingSentences(BenchmarkModel):
    home_rehab: list[str]
    inpatient_rehab: list[str]
    persistent_deficit: dict[str, list[str]]


class ChunkTemplateUtils(BenchmarkModel):
    condition_presentations: dict[str, list[str]]
    condition_status_phrases: dict[str, list[str]]
    duration_closing_sentences: dict[str, list[str]]
    functional_status_phrases: dict[str, dict[str, list[str]]]
    rehab_closing_sentences: RehabClosingSentences
    hidden_benchmark_terms: list[str]
    duration_course_nouns: list[str]
    duration_phrase_templates: list[str]
    duration_response_verbs: list[str]
    duration_chunk_templates: list[str]
    rehab_transitions: list[str]
    rehab_outcome_verbs: list[str]
    rehab_chunk_templates: list[str]
    subgroup_terms: dict[str, list[str]]
    rehab_language_terms: list[str]
    rehab_bin_terms: dict[str, list[str]]
    persistent_deficit_descriptor_terms: list[str]
    persistent_deficit_rehab_terms: list[str]
    meaningful_token_stopwords: list[str]
