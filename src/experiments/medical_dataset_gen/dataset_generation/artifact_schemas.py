"""Schemas for latent plans and persisted benchmark artifacts."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import Field, model_validator

from experiments.medical_dataset_gen.dataset_generation.payload_schemas import (
    axis_payload_required_phrase,
    validate_axis_payload,
)
from experiments.medical_dataset_gen.dataset_generation.schema_types import (
    BenchmarkPydanticModel,
    ChunkSurfaceGroup,
    ClinicalAxis,
    ClusterRole,
    CohortContrastFamily,
    ConditionAnchor,
    ConditionKey,
    DataSplit,
    PatientSex,
    QueryType,
    SubgroupAxis,
)


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
    cohort_contrast_family: CohortContrastFamily
    cohort_dimension_id: str
    primary_axis: ClinicalAxis
    secondary_axis: ClinicalAxis
    dominant_primary_facet_id: str
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


class AnswerFact(BenchmarkPydanticModel):
    facet_id: str
    subgroup_label: str
    axis: ClinicalAxis
    summary: str
    supporting_fact_ids: list[str]


class QueryPlanFacet(BenchmarkPydanticModel):
    facet_id: str
    subgroup_id: str
    subgroup_label: str
    axis: ClinicalAxis
    value_bin: str
    cluster_id: str
    cluster_role: ClusterRole
    target_gold_chunks: int
    priority: Literal['primary', 'secondary']


class QueryLogicalForm(BenchmarkPydanticModel):
    query_type: QueryType = Field(alias='type')
    condition: str
    subgroups: list[str]
    axes: list[ClinicalAxis]
    facets: list[str]
    cohort_contrast_family: CohortContrastFamily
    primary_axis: ClinicalAxis
    secondary_axis: ClinicalAxis
    dominant_primary_facet_id: str

class QueryPlan(BenchmarkPydanticModel):
    query_id: str
    evidence_profile_id: str
    pool_id: str
    outcome_profile_id: str
    plan_seed: int
    split: DataSplit
    query_type: QueryType
    condition_id: ConditionKey
    condition_display: str
    subgroup_a_id: str
    subgroup_a_label: str
    subgroup_b_id: str
    subgroup_b_label: str
    cohort_contrast_id: str
    cohort_contrast_family: CohortContrastFamily
    cohort_dimension_id: str
    primary_axis: ClinicalAxis
    secondary_axis: ClinicalAxis
    dominant_primary_facet_id: str
    facets: list[QueryPlanFacet]
    logical_form: QueryLogicalForm


class ClinicalFact(BenchmarkPydanticModel):
    query_id: str
    evidence_profile_id: str
    pool_id: str
    primary_axis: ClinicalAxis
    secondary_axis: ClinicalAxis
    dominant_primary_facet_id: str
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
    axis_bin_term: str
    axis_payload_json: str
    condition_anchor: ConditionAnchor
    facet_priority: Literal['primary', 'secondary'] | None
    is_gold: bool
    distractor_type: str | None
    admission_id: str
    patient_age: int
    patient_sex: PatientSex
    clinical_subgroup_phrase: str
    note_style: str
    chunk_surface_group: ChunkSurfaceGroup
    split: DataSplit

    @model_validator(mode='after')
    def _validate_axis_payload(self) -> ClinicalFact:
        payload = validate_axis_payload(self.axis, self.axis_payload_json)
        required_phrase = axis_payload_required_phrase(payload)
        expected_anchor: ConditionAnchor = (
            'axis_evidence'
            if self.condition_display.casefold() in required_phrase.casefold()
            else 'outer_template'
        )
        if self.condition_anchor != expected_anchor:
            raise ValueError(
                f'condition_anchor {self.condition_anchor!r} does not match '
                f'payload-derived anchor {expected_anchor!r}'
            )
        return self


class ChunkRow(ClinicalFact):
    chunk_id: str
    text: str
    approx_words: int
    outer_template_family: str | None = None
    outer_template_id: str | None = None
    axis_template_family: str | None = None
    axis_template_id: str | None = None
