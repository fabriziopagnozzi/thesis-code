"""Schemas for the authored clinical ontology and admissibility policies."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from experiments.medical_dataset_gen.dataset_generation.ontology_validation import (
    validate_ontology,
)
from experiments.medical_dataset_gen.dataset_generation.schema_types import (
    BenchmarkPydanticModel,
    ClinicalAxis,
    ConditionKey,
    PatientSex,
    QueryFocusMode,
    SubgroupAxis,
    SubgroupKey,
)


class TreatmentDurationCourse(BenchmarkPydanticModel):
    surface_forms: list[str] = Field(min_length=1)
    bins: dict[str, list[int]]

    @model_validator(mode='after')
    def _validate_day_bins(self) -> TreatmentDurationCourse:
        seen: dict[int, str] = {}
        for bin_id, days in self.bins.items():
            if not days:
                raise ValueError(f'treatment-duration bin {bin_id!r} must not be empty')
            if any(day < 1 for day in days):
                raise ValueError(f'treatment-duration bin {bin_id!r} contains nonpositive days')
            if len(days) != len(set(days)):
                raise ValueError(f'treatment-duration bin {bin_id!r} repeats a duration day')
            for day in days:
                previous = seen.get(day)
                if previous is not None:
                    raise ValueError(
                        f'treatment-duration day {day} appears in both {previous!r} and {bin_id!r}'
                    )
                seen[day] = bin_id
        return self


class TreatmentDurationAxisValues(BenchmarkPydanticModel):
    axis: Literal['treatment_duration']
    treatments: dict[str, TreatmentDurationCourse]

    @property
    def bins(self) -> dict[str, list[int]]:
        merged: dict[str, list[int]] = {}
        for treatment in self.treatments.values():
            for bin_id, days in treatment.bins.items():
                merged.setdefault(bin_id, []).extend(days)
        return merged

    @model_validator(mode='after')
    def _validate_treatment_courses(self) -> TreatmentDurationAxisValues:
        if not self.treatments:
            raise ValueError('treatment_duration must define at least one treatment course')
        expected_bins: set[str] | None = None
        for treatment_id, treatment in self.treatments.items():
            if expected_bins is None:
                expected_bins = set(treatment.bins)
            elif set(treatment.bins) != expected_bins:
                raise ValueError(
                    'all treatment-duration courses must define the same value bins; '
                    f'{treatment_id!r} differs'
                )
        return self


class RehabOutcomeAxisValues(BenchmarkPydanticModel):
    axis: Literal['rehab_outcome']
    bins: dict[str, list[str]]


class ComplicationBurdenAxisValues(BenchmarkPydanticModel):
    axis: Literal['complication_burden']
    bins: dict[str, list[str]]


class AcuteClinicalCourseAxisValues(BenchmarkPydanticModel):
    axis: Literal['acute_clinical_course']
    bins: dict[str, list[str]]


class CareIntensityAxisValues(BenchmarkPydanticModel):
    axis: Literal['care_intensity']
    bins: dict[str, list[str]]


class DiagnosticEvidenceAxisValues(BenchmarkPydanticModel):
    axis: Literal['diagnostic_evidence_type']
    bins: dict[str, list[str]]


type ConditionAxisValues = Annotated[
    TreatmentDurationAxisValues
    | RehabOutcomeAxisValues
    | ComplicationBurdenAxisValues
    | AcuteClinicalCourseAxisValues
    | CareIntensityAxisValues
    | DiagnosticEvidenceAxisValues,
    Field(discriminator='axis'),
]


class SubgroupOntology(BenchmarkPydanticModel):
    dimension_id: str
    level_id: str
    axis: SubgroupAxis
    label: str
    field: str
    value: str
    is_reference: bool
    patient_age_range: tuple[int, int] | None = None
    patient_sex: PatientSex | None = None
    aliases: list[str] = Field(default_factory=list)
    surface_phrases: list[str] = Field(default_factory=list)


class CohortContrast(BenchmarkPydanticModel):
    id: str
    dimension_id: str
    cohort_a_id: str
    cohort_b_id: str


class DistinctComorbidityContrast(BenchmarkPydanticModel):
    id: str
    cohort_a_id: str
    cohort_b_id: str


class ConditionOntology(BenchmarkPydanticModel):
    display: str
    allowed_comorbidity_contrast_ids: list[str] = Field(default_factory=list)
    allowed_distinct_comorbidity_contrasts: list[DistinctComorbidityContrast] = Field(
        default_factory=list
    )
    terms: list[str]
    axis_values: dict[ClinicalAxis, ConditionAxisValues]

    @model_validator(mode='after')
    def _axis_keys_match_payloads(self) -> ConditionOntology:
        for axis, values in self.axis_values.items():
            if axis != values.axis:
                raise ValueError(f'axis_values key {axis!r} does not match payload {values.axis!r}')
        return self


class AxisPairProfile(BenchmarkPydanticModel):
    id: str
    cohort_a_bins: tuple[str, str]
    cohort_b_bins: tuple[str, str]


class AxisQueryFocus(BenchmarkPydanticModel):
    list: str
    natural: str

    def text_for(self, mode: QueryFocusMode) -> str:
        if mode == 'list':
            return self.list
        return self.natural


class ClinicalAxisOntology(BenchmarkPydanticModel):
    label: str
    query_label: str
    allow_as_primary: bool
    query_focus: AxisQueryFocus
    exact_terms: list[str]
    synonym_terms: list[str]
    bins: list[str]
    bin_terms: dict[str, list[str]]

    @model_validator(mode='after')
    def _validate_bin_terms(self) -> ClinicalAxisOntology:
        if set(self.bin_terms) != set(self.bins):
            raise ValueError('bin_terms must define every declared clinical-axis bin')
        if not self.query_label.strip():
            raise ValueError('clinical-axis query_label must not be empty')
        if any(not terms for terms in self.bin_terms.values()):
            raise ValueError('clinical-axis bin_terms must not contain empty term lists')
        return self


class AxisPairConditionOverride(BenchmarkPydanticModel):
    condition_id: ConditionKey
    allowed_primary_axes: list[ClinicalAxis] | None = None
    blocked_profile_ids: list[str] = Field(default_factory=list)
    rationale: str | None = None


class AxisPairOntology(BenchmarkPydanticModel):
    axes: tuple[ClinicalAxis, ClinicalAxis]
    profiles: list[AxisPairProfile]
    allowed_primary_axes: list[ClinicalAxis] | None = None
    blocked_profile_ids: list[str] = Field(default_factory=list)
    rationale: str | None = None
    condition_overrides: list[AxisPairConditionOverride] = Field(default_factory=list)

    @model_validator(mode='after')
    def _validate_generation_policy(self) -> AxisPairOntology:
        axis_set = set(self.axes)
        profile_ids = {profile.id for profile in self.profiles}
        if self.allowed_primary_axes is not None and not set(self.allowed_primary_axes) <= axis_set:
            raise ValueError('allowed_primary_axes must be a subset of the axis-pair members')
        unknown_blocked = set(self.blocked_profile_ids) - profile_ids
        if unknown_blocked:
            unknown = ', '.join(sorted(unknown_blocked))
            raise ValueError(f'axis-pair policy blocks unknown profiles: {unknown}')
        seen_conditions: set[str] = set()
        for override in self.condition_overrides:
            if override.condition_id in seen_conditions:
                raise ValueError('axis-pair condition overrides must be unique per condition')
            seen_conditions.add(override.condition_id)
            if (
                override.allowed_primary_axes is not None
                and not set(override.allowed_primary_axes) <= axis_set
            ):
                raise ValueError(
                    'axis-pair condition override allowed_primary_axes must stay within the pair'
                )
            unknown_override_profiles = set(override.blocked_profile_ids) - profile_ids
            if unknown_override_profiles:
                unknown = ', '.join(sorted(unknown_override_profiles))
                raise ValueError(f'axis-pair condition override blocks unknown profiles: {unknown}')
        return self


class PatientDefaults(BenchmarkPydanticModel):
    age_range: tuple[int, int]


class MedicalOntology(BenchmarkPydanticModel):
    patient_defaults: PatientDefaults
    conditions: dict[ConditionKey, ConditionOntology]
    subgroups: dict[SubgroupKey, SubgroupOntology]
    clinical_axes: dict[ClinicalAxis, ClinicalAxisOntology]
    cohort_contrasts: list[CohortContrast]
    axis_pairs: list[AxisPairOntology]

    @model_validator(mode='after')
    def _validate_references(self) -> MedicalOntology:
        validate_ontology(self)
        return self
