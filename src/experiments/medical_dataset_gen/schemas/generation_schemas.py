from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

type ClinicalAxisKey = str
type ClinicalAxis = Literal[
    'treatment_duration',
    'rehab_outcome',
    'complication_burden',
    'acute_clinical_course',
    'care_intensity',
]
CLINICAL_AXIS_LIST = list[ClinicalAxis](get_args(ClinicalAxis.__value__))

type ClusterRole = Literal[
    'calibrated_primary_gold',
    'primary_gold',
    'secondary_gold',
    'hard_distractor',
    'background_outlier',
]
CLUSTER_ROLE_LIST = list[ClusterRole](get_args(ClusterRole.__value__))

type DistractorStr = Literal[
    'same_condition_wrong_subgroup',
    'same_subgroup_wrong_condition',
    'same_axis_wrong_condition',
]
DISTRACTOR_TYPES = list[DistractorStr](get_args(DistractorStr.__value__))

type QueryType = Literal['prioritized_subgroup_comparison']

type PlanCalibrationMode = Literal['rotating', 'embedding_calibrated']

type ChunkPoolScope = Literal['query_local', 'same_condition', 'full_corpus']

type SubgroupAxis = Literal['demographic', 'comorbidity']
type SubgroupKey = str

type DataSplit = Literal['train', 'validation', 'test']
type PatientSex = Literal['female', 'male']

type ConditionKey = Literal[
    'encephalitis_myelitis',
    'pneumonia',
    'ischemic_stroke',
    'heart_failure',
    'bacterial_meningitis',
    'multiple_sclerosis_relapse',
    'spinal_epidural_abscess',
    'pulmonary_embolism',
    'copd_exacerbation',
    'infective_endocarditis',
    'osteomyelitis',
    'pyelonephritis_sepsis',
    'diabetic_foot_infection',
    'cirrhosis_encephalopathy',
    'lupus_nephritis',
    'severe_cellulitis',
    'ulcerative_colitis_flare',
]


class QueryOutputRow(TypedDict):
    query_id: str
    evidence_profile_id: str
    pool_id: str
    outcome_profile_id: str
    query_type: QueryType
    template_id: str
    condition_id: str
    condition_display: str
    subgroup_a_id: str
    subgroup_a_label: str
    subgroup_b_id: str
    subgroup_b_label: str
    cohort_contrast_id: str
    cohort_dimension_id: str
    primary_axis: ClinicalAxis
    secondary_axis: ClinicalAxis
    calibrated_primary_facet_id: str
    split: DataSplit
    n_facets: int
    facets_json: str
    logical_form_json: str
    query_text: str


class GoldAnswerOutputRow(TypedDict):
    query_id: str
    evidence_profile_id: str
    pool_id: str
    answer_text: str
    facet_summaries_json: str
    answer_facts_json: str
    supporting_fact_ids_json: str
    supporting_facet_ids_json: str


class BenchmarkModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class TreatmentDurationPayload(BenchmarkModel):
    axis: Literal['treatment_duration']
    duration_days: int
    treatment: str


class RehabOutcomePayload(BenchmarkModel):
    axis: Literal['rehab_outcome']
    outcome: str


class ComplicationBurdenPayload(BenchmarkModel):
    axis: Literal['complication_burden']
    detail: str


class AcuteClinicalCoursePayload(BenchmarkModel):
    axis: Literal['acute_clinical_course']
    detail: str


class CareIntensityPayload(BenchmarkModel):
    axis: Literal['care_intensity']
    detail: str


type AxisFactPayload = Annotated[
    TreatmentDurationPayload
    | RehabOutcomePayload
    | ComplicationBurdenPayload
    | AcuteClinicalCoursePayload
    | CareIntensityPayload,
    Field(discriminator='axis'),
]


def parse_axis_payload(value: str) -> AxisFactPayload:
    payload = json.loads(value)
    axis = payload.get('axis')
    model_by_axis = {
        'treatment_duration': TreatmentDurationPayload,
        'rehab_outcome': RehabOutcomePayload,
        'complication_burden': ComplicationBurdenPayload,
        'acute_clinical_course': AcuteClinicalCoursePayload,
        'care_intensity': CareIntensityPayload,
    }
    try:
        model = model_by_axis[axis]
    except KeyError as exc:
        raise ValueError(f'unsupported axis payload: {axis!r}') from exc
    return model.model_validate(payload)


class AnswerSourceFact(BenchmarkModel):
    model_config = ConfigDict(extra='ignore')

    query_id: str
    facet_id: str
    axis: ClinicalAxis
    value_bin: str
    axis_payload_json: str
    facet_priority: Literal['primary', 'secondary'] | None
    fact_id: str

    @model_validator(mode='after')
    def _validate_axis_payload(self) -> AnswerSourceFact:
        payload = parse_axis_payload(self.axis_payload_json)
        if payload.axis != self.axis:
            raise ValueError(f'axis payload {payload.axis!r} does not match {self.axis!r}')
        return self


class AnswerFact(BenchmarkModel):
    facet_id: str
    subgroup_label: str
    axis: ClinicalAxis
    summary: str
    supporting_fact_ids: list[str]


class ConditionOntology(BenchmarkModel):
    display: str
    terms: list[str]
    treatments: list[str]
    duration_treatments: list[str] | None = None
    duration_days: dict[str, tuple[int, int]]
    rehab_outcomes: dict[str, list[str]]
    new_axis_values: dict[ClinicalAxisKey, dict[str, list[str]]] = Field(default_factory=dict)


class SubgroupOntology(BenchmarkModel):
    dimension_id: str
    level_id: str
    axis: SubgroupAxis
    label: str
    field: str
    value: str
    is_reference: bool
    aliases: list[str] = Field(default_factory=list)
    surface_phrases: list[str] = Field(default_factory=list)

    def prefixed_fields(self, prefix: str, subgroup_id: str) -> dict[str, object]:
        return {
            f'{prefix}_id': subgroup_id,
            f'{prefix}_label': self.label,
            f'{prefix}_axis': self.axis,
            f'{prefix}_field': self.field,
            f'{prefix}_value': self.value,
            f'{prefix}_dimension_id': self.dimension_id,
            f'{prefix}_level_id': self.level_id,
            f'{prefix}_is_reference': self.is_reference,
        }


class CohortContrast(BenchmarkModel):
    id: str
    dimension_id: str
    cohort_a_id: str
    cohort_b_id: str


class AxisPairProfile(BenchmarkModel):
    id: str
    cohort_a_bins: tuple[str, str]
    cohort_b_bins: tuple[str, str]


class ClinicalAxisOntology(BenchmarkModel):
    label: str
    exact_terms: list[str]
    synonym_terms: list[str]
    bins: list[str]


class MedicalOntology(BenchmarkModel):
    conditions: dict[ConditionKey, ConditionOntology]
    subgroups: dict[SubgroupKey, SubgroupOntology]
    clinical_axes: dict[ClinicalAxisKey, ClinicalAxisOntology]
    cohort_contrasts: list[CohortContrast]
    axis_pair_profiles: dict[str, list[AxisPairProfile]]


class QueryPlanFacet(BenchmarkModel):
    facet_id: str
    condition_id: ConditionKey
    condition_display: str
    subgroup_id: str
    subgroup_label: str
    subgroup_axis: SubgroupAxis
    subgroup_field: str
    subgroup_value: str
    axis: ClinicalAxis
    value_bin: str
    cluster_id: str
    cluster_role: ClusterRole
    target_gold_chunks: int
    priority: Literal['primary', 'secondary']


class QueryLogicalForm(BenchmarkModel):
    query_type: QueryType = Field(alias='type')
    condition: str
    subgroups: list[str]
    axes: list[ClinicalAxis]
    facets: list[str]
    primary_axis: ClinicalAxis
    secondary_axis: ClinicalAxis
    calibrated_primary_facet_id: str

    model_config = ConfigDict(extra='forbid', populate_by_name=True)


class QueryPlanSpec(BenchmarkModel):
    evidence_profile_id: str
    cohort_contrast_id: str
    cohort_dimension_id: str
    axis_a: ClinicalAxis
    axis_b: ClinicalAxis
    profile_id: str
    cohort_a_bins: tuple[str, str]
    cohort_b_bins: tuple[str, str]
    condition_key: ConditionKey
    condition_display: str
    subgroup_a_id: str
    subgroup_a: SubgroupOntology
    subgroup_b_id: str
    subgroup_b: SubgroupOntology


class QueryPlan(BenchmarkModel):
    query_id: str
    evidence_profile_id: str
    pool_id: str
    outcome_profile_id: str
    plan_seed: int
    split: DataSplit
    query_type: QueryType
    template_id: str
    condition_id: ConditionKey
    condition_display: str
    subgroup_a_id: str
    subgroup_a_label: str
    subgroup_a_axis: SubgroupAxis
    subgroup_a_field: str
    subgroup_a_value: str
    subgroup_a_dimension_id: str
    subgroup_a_level_id: str
    subgroup_a_is_reference: bool
    subgroup_b_id: str
    subgroup_b_label: str
    subgroup_b_axis: SubgroupAxis
    subgroup_b_field: str
    subgroup_b_value: str
    subgroup_b_dimension_id: str
    subgroup_b_level_id: str
    subgroup_b_is_reference: bool
    cohort_contrast_id: str
    cohort_dimension_id: str
    primary_axis: ClinicalAxis
    secondary_axis: ClinicalAxis
    calibrated_primary_facet_id: str
    n_facets: int
    gold_chunks_total: int
    distractor_chunks: int
    facets: list[QueryPlanFacet]
    logical_form: QueryLogicalForm

    @model_validator(mode='before')
    @classmethod
    def _hydrate_json_columns(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        hydrated = dict(value)
        facets_json = hydrated.pop('facets_json', None)
        logical_form_json = hydrated.pop('logical_form_json', None)
        if 'facets' not in hydrated and facets_json is not None:
            hydrated['facets'] = json.loads(facets_json)
        if 'logical_form' not in hydrated and logical_form_json is not None:
            hydrated['logical_form'] = json.loads(logical_form_json)
        return hydrated

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

    def to_query_row(self, query_text: str) -> QueryOutputRow:
        facets_json, logical_form_json = self._json_columns()
        return {
            'query_id': self.query_id,
            'evidence_profile_id': self.evidence_profile_id,
            'pool_id': self.pool_id,
            'outcome_profile_id': self.outcome_profile_id,
            'query_type': self.query_type,
            'template_id': self.template_id,
            'condition_id': self.condition_id,
            'condition_display': self.condition_display,
            'subgroup_a_id': self.subgroup_a_id,
            'subgroup_a_label': self.subgroup_a_label,
            'subgroup_b_id': self.subgroup_b_id,
            'subgroup_b_label': self.subgroup_b_label,
            'cohort_contrast_id': self.cohort_contrast_id,
            'cohort_dimension_id': self.cohort_dimension_id,
            'primary_axis': self.primary_axis,
            'secondary_axis': self.secondary_axis,
            'calibrated_primary_facet_id': self.calibrated_primary_facet_id,
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
        facet_answer_objects: list[AnswerFact],
        supporting_fact_ids: list[str],
    ) -> GoldAnswerOutputRow:
        return {
            'query_id': self.query_id,
            'evidence_profile_id': self.evidence_profile_id,
            'pool_id': self.pool_id,
            'answer_text': answer_text,
            'facet_summaries_json': json.dumps(facet_summaries, sort_keys=True),
            'answer_facts_json': json.dumps(
                [fact.model_dump(mode='json') for fact in facet_answer_objects], sort_keys=True
            ),
            'supporting_fact_ids_json': json.dumps(supporting_fact_ids, sort_keys=True),
            'supporting_facet_ids_json': json.dumps(
                [facet.facet_id for facet in self.facets], sort_keys=True
            ),
        }


class ClinicalFact(BenchmarkModel):
    query_id: str
    evidence_profile_id: str
    pool_id: str
    primary_axis: ClinicalAxis
    secondary_axis: ClinicalAxis
    calibrated_primary_facet_id: str
    fact_id: str
    chunk_reuse_key: str
    facet_id: str | None
    target_facet_id: str | None
    cluster_id: str
    cluster_role: ClusterRole
    condition_id: ConditionKey
    condition_display: str
    subgroup_id: str
    subgroup_label: str
    subgroup_axis: SubgroupAxis
    subgroup_field: str
    subgroup_value: str
    subgroup_dimension_id: str
    subgroup_level_id: str
    subgroup_is_reference: bool
    axis: ClinicalAxis
    value_bin: str
    axis_payload_json: str
    facet_priority: Literal['primary', 'secondary'] | None
    is_gold: bool
    distractor_type: str | None
    admission_id: str
    patient_id: str
    patient_age: int
    patient_sex: PatientSex
    clinical_subgroup_phrase: str
    note_style: str
    split: DataSplit
    must_mention: list[str] = Field(default_factory=list)
    must_not_mention: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def _validate_axis_payload(self) -> ClinicalFact:
        payload = parse_axis_payload(self.axis_payload_json)
        if payload.axis != self.axis:
            raise ValueError(f'axis payload {payload.axis!r} does not match {self.axis!r}')
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
    validation_soft_warnings: list[str] = field(default_factory=list[str])


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
    duration_focus_phrases: list[str]
    note_style_templates: dict[str, list[str]]
    axis_closing_sentences: dict[ClinicalAxisKey, list[str]]
    axis_sentence_templates: dict[ClinicalAxisKey, list[str]]
    axis_bin_terms: dict[ClinicalAxisKey, dict[str, list[str]]]
    condition_new_axis_values: dict[str, dict[ClinicalAxisKey, dict[str, list[str]]]]


class QueryTemplateSpec(BenchmarkModel):
    id: str
    template: str


class AnswerTemplateSpec(BenchmarkModel):
    template: str


class QueryTemplateData(BenchmarkModel):
    query_templates: dict[QueryType, list[QueryTemplateSpec]]
    answer_templates: dict[QueryType, AnswerTemplateSpec]
