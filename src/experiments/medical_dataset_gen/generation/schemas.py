import json
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FacetAxis = Literal['treatment_duration', 'rehab_outcome']
ClusterRole = Literal['dominant_gold', 'complementary_gold', 'hard_distractor']
QueryType = Literal['subgroup_comparison', 'outcome_synthesis']
Split = Literal['train', 'validation', 'test']
PatientSex = Literal['female', 'male']
SubgroupAxis = Literal['demographic', 'comorbidity']


class BenchmarkModel(BaseModel):
    model_config = ConfigDict(extra='forbid')

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)

    def get(self, key: str, default: object = None) -> object:
        return getattr(self, key, default)


class ClinicalConditionOntology(BenchmarkModel):
    display: str
    terms: list[str]
    treatments: list[str]
    duration_treatments: list[str] | None = None
    duration_days: dict[str, tuple[int, int]]
    rehab_outcomes: dict[str, list[str]]


class ClinicalSubgroupOntology(BenchmarkModel):
    axis: SubgroupAxis
    label: str
    field: str
    value: str
    aliases: list[str] = Field(default_factory=list)


class ClinicalAxisOntology(BenchmarkModel):
    label: str
    exact_terms: list[str]
    synonym_terms: list[str]
    bins: list[str]


class MedicalOntology(BenchmarkModel):
    conditions: dict[str, ClinicalConditionOntology]
    subgroups: dict[str, ClinicalSubgroupOntology]
    clinical_axes: dict[str, ClinicalAxisOntology]


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
    condition_id: str
    condition_display: str
    subgroup_a_id: str
    subgroup_a: ClinicalSubgroupOntology
    subgroup_b_id: str
    subgroup_b: ClinicalSubgroupOntology


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

    def to_row(self) -> dict[str, object]:
        row = self.model_dump(mode='python', exclude={'facets', 'logical_form'})
        row['facets_json'] = json.dumps(
            [facet.model_dump(mode='json') for facet in self.facets], sort_keys=True
        )
        row['logical_form_json'] = json.dumps(
            self.logical_form.model_dump(mode='json', by_alias=True), sort_keys=True
        )
        return row


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
