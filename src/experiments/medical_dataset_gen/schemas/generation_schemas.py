from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from typing import Annotated, Literal, TypedDict, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

type ClinicalAxis = Literal[
    'treatment_duration',
    'rehab_outcome',
    'complication_burden',
    'acute_clinical_course',
    'care_intensity',
    'diagnostic_evidence_type',
]
CLINICAL_AXIS_LIST = list[ClinicalAxis](get_args(ClinicalAxis.__value__))

type ClusterRole = Literal[
    'dominant_primary_gold',
    'primary_gold',
    'secondary_gold',
    'niche_gold',
    'hard_distractor',
    'background_outlier',
]

type QueryType = Literal['prioritized_subgroup_comparison']
type CohortContrastFamily = Literal[
    'demographic',
    'comorbidity_present_absent',
    'distinct_comorbidity',
]

type PlanCalibrationMode = Literal['rotating', 'embedding_calibrated']
type ChunkPoolScope = Literal['query_local']
type ChunkTextStyle = Literal['ontology_explicit', 'semantic_hardened']
type QueryFocusMode = Literal['list', 'natural']
QUERY_FOCUS_MODE_LIST = list[QueryFocusMode](get_args(QueryFocusMode.__value__))
type QueryStructure = Literal['unbalanced', 'balanced']
QUERY_STRUCTURE_LIST = list[QueryStructure](get_args(QueryStructure.__value__))
type ChunkSurfaceGroup = Literal['seen', 'heldout']
CHUNK_SURFACE_GROUP_LIST = list[ChunkSurfaceGroup](get_args(ChunkSurfaceGroup.__value__))
type ChunkSurfacePolicy = Literal['split_heldout', 'seen_only', 'heldout_only']
type AxisTemplateFamily = Literal[
    'direct_fact',
    'temporal_course',
    'clinical_assessment',
    'contrast_or_alternative',
    'fragmented_or_split_evidence',
]
AXIS_TEMPLATE_FAMILY_LIST = list[AxisTemplateFamily](
    get_args(AxisTemplateFamily.__value__)
)
SEEN_AXIS_TEMPLATE_FAMILIES = AXIS_TEMPLATE_FAMILY_LIST
HELDOUT_AXIS_TEMPLATE_FAMILIES = list[AxisTemplateFamily](AXIS_TEMPLATE_FAMILY_LIST[2:])
type SubgroupAxis = Literal['demographic', 'comorbidity']
type SubgroupKey = str
type ConditionKey = str
type PatientSex = Literal['female', 'male']

type DataSplit = Literal['validation', 'test']


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


class BenchmarkPydanticModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class TreatmentDurationPayload(BenchmarkPydanticModel):
    axis: Literal['treatment_duration']
    duration_days: int
    treatment: str
    treatment_course_id: str | None = None


class RehabOutcomePayload(BenchmarkPydanticModel):
    axis: Literal['rehab_outcome']
    outcome: str


class ComplicationBurdenPayload(BenchmarkPydanticModel):
    axis: Literal['complication_burden']
    detail: str


class AcuteClinicalCoursePayload(BenchmarkPydanticModel):
    axis: Literal['acute_clinical_course']
    detail: str


class CareIntensityPayload(BenchmarkPydanticModel):
    axis: Literal['care_intensity']
    detail: str


class DiagnosticEvidencePayload(BenchmarkPydanticModel):
    axis: Literal['diagnostic_evidence_type']
    detail: str


type AxisFactPayload = Annotated[
    TreatmentDurationPayload
    | RehabOutcomePayload
    | ComplicationBurdenPayload
    | AcuteClinicalCoursePayload
    | CareIntensityPayload
    | DiagnosticEvidencePayload,
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
        'diagnostic_evidence_type': DiagnosticEvidencePayload,
    }
    try:
        model = model_by_axis[axis]
    except KeyError as exc:
        raise ValueError(f'unsupported axis payload: {axis!r}') from exc
    return model.model_validate(payload)


class AnswerSourceFact(BenchmarkPydanticModel):
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


class AnswerFact(BenchmarkPydanticModel):
    facet_id: str
    subgroup_label: str
    axis: ClinicalAxis
    summary: str
    supporting_fact_ids: list[str]


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
    presentations: list[str]
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
    allow_as_primary: bool
    query_focus: AxisQueryFocus
    exact_terms: list[str]
    synonym_terms: list[str]
    bins: list[str]
    bin_terms: dict[str, list[str]]

    @model_validator(mode='before')
    @classmethod
    def _normalize_legacy_query_focus(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        hydrated = dict(value)
        query_focus = hydrated.get('query_focus')
        if isinstance(query_focus, str):
            hydrated['query_focus'] = {'list': query_focus, 'natural': query_focus}
        return hydrated

    @model_validator(mode='after')
    def _validate_bin_terms(self) -> ClinicalAxisOntology:
        if set(self.bin_terms) != set(self.bins):
            raise ValueError('bin_terms must define every declared clinical-axis bin')
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
        declared_axes = set(self.clinical_axes)
        contrast_by_id: dict[str, CohortContrast] = {}
        for contrast in self.cohort_contrasts:
            if contrast.id in contrast_by_id:
                raise ValueError(f'duplicate cohort contrast id: {contrast.id!r}')
            contrast_by_id[contrast.id] = contrast

        for condition_id, condition in self.conditions.items():
            if set(condition.axis_values) != declared_axes:
                raise ValueError(f'condition {condition_id!r} must define every clinical axis')
            for axis, values in condition.axis_values.items():
                if set(values.bins) != set(self.clinical_axes[axis].bins):
                    raise ValueError(f'condition {condition_id!r} has incomplete bins for {axis!r}')
            allowed_ids = condition.allowed_comorbidity_contrast_ids
            if len(allowed_ids) != len(set(allowed_ids)):
                raise ValueError(
                    f'condition {condition_id!r} repeats allowed comorbidity contrast ids'
                )
            unknown_allowed = set(allowed_ids) - set(contrast_by_id)
            if unknown_allowed:
                unknown = ', '.join(sorted(unknown_allowed))
                raise ValueError(
                    f'condition {condition_id!r} allows unknown comorbidity contrasts: {unknown}'
                )
            for contrast_id in allowed_ids:
                contrast = contrast_by_id[contrast_id]
                cohorts = [
                    self.subgroups[contrast.cohort_a_id],
                    self.subgroups[contrast.cohort_b_id],
                ]
                if not _is_comorbidity_present_absent_contrast(cohorts):
                    raise ValueError(
                        f'condition {condition_id!r} allows non-comorbidity present/absent '
                        f'contrast {contrast_id!r}'
                    )
            _validate_distinct_comorbidity_contrasts(
                condition_id,
                condition,
                self.subgroups,
                contrast_by_id,
            )

        for contrast in self.cohort_contrasts:
            cohorts = [self.subgroups[contrast.cohort_a_id], self.subgroups[contrast.cohort_b_id]]
            if any(cohort.dimension_id != contrast.dimension_id for cohort in cohorts):
                raise ValueError(f'contrast {contrast.id!r} mixes cohort dimensions')

        _validate_absent_subgroup_surface_forms(self.subgroups)

        declared_pairs = {frozenset(pair.axes) for pair in self.axis_pairs}
        expected_pairs = {frozenset(pair) for pair in combinations(declared_axes, 2)}
        if declared_pairs != expected_pairs or len(declared_pairs) != len(self.axis_pairs):
            raise ValueError('axis_pairs must contain each unordered clinical-axis pair once')
        for pair in self.axis_pairs:
            if len(pair.profiles) < 2:
                raise ValueError('each axis pair must define at least two joint profiles')
            left, right = pair.axes
            candidate_primary_axes = (
                list(pair.axes) if pair.allowed_primary_axes is None else pair.allowed_primary_axes
            )
            explicitly_suppressed = pair.allowed_primary_axes == []
            if (
                not explicitly_suppressed
                and not any(
                    self.clinical_axes[axis].allow_as_primary for axis in candidate_primary_axes
                )
            ):
                raise ValueError(f'axis pair {left!r}/{right!r} has no permitted primary axis')
            profile_ids = [profile.id for profile in pair.profiles]
            if len(profile_ids) != len(set(profile_ids)):
                raise ValueError(f'axis pair {left!r}/{right!r} reuses a profile id')
            for profile in pair.profiles:
                cohort_pairs = zip(profile.cohort_a_bins, profile.cohort_b_bins, strict=True)
                if any(a == b for a, b in cohort_pairs):
                    raise ValueError(f'profile {profile.id!r} must differ on both axes')
                if not all(
                    bins[index] in self.clinical_axes[axis].bins
                    for index, axis in enumerate((left, right))
                    for bins in (profile.cohort_a_bins, profile.cohort_b_bins)
                ):
                    raise ValueError(f'profile {profile.id!r} contains an unknown axis bin')
        return self


_BANNED_NEGATIVE_SUBTYPE_MODIFIERS = frozenset({
    'complicated',
    'metastatic',
    'mild',
    'uncomplicated',
})


def _is_comorbidity_present_absent_contrast(cohorts: list[SubgroupOntology]) -> bool:
    if len(cohorts) != 2:
        return False
    if any(cohort.axis != 'comorbidity' for cohort in cohorts):
        return False
    return {cohort.level_id for cohort in cohorts} == {'present', 'absent'}


def _validate_distinct_comorbidity_contrasts(
    condition_id: str,
    condition: ConditionOntology,
    subgroups: dict[SubgroupKey, SubgroupOntology],
    contrast_by_id: dict[str, CohortContrast],
) -> None:
    contrast_ids = [contrast.id for contrast in condition.allowed_distinct_comorbidity_contrasts]
    if len(contrast_ids) != len(set(contrast_ids)):
        raise ValueError(
            f'condition {condition_id!r} repeats allowed distinct comorbidity contrast ids'
        )

    present_absent_id_by_present_subgroup: dict[str, str] = {}
    for contrast in contrast_by_id.values():
        cohorts = {
            contrast.cohort_a_id: subgroups[contrast.cohort_a_id],
            contrast.cohort_b_id: subgroups[contrast.cohort_b_id],
        }
        if not _is_comorbidity_present_absent_contrast(list(cohorts.values())):
            continue
        present_id = next(
            subgroup_id
            for subgroup_id, subgroup in cohorts.items()
            if subgroup.level_id == 'present'
        )
        present_absent_id_by_present_subgroup[present_id] = contrast.id

    allowed_present_absent_ids = set(condition.allowed_comorbidity_contrast_ids)
    for contrast in condition.allowed_distinct_comorbidity_contrasts:
        if contrast.cohort_a_id == contrast.cohort_b_id:
            raise ValueError(
                f'condition {condition_id!r} distinct comorbidity contrast {contrast.id!r} '
                'uses the same cohort twice'
            )
        unknown_cohorts = {
            subgroup_id
            for subgroup_id in (contrast.cohort_a_id, contrast.cohort_b_id)
            if subgroup_id not in subgroups
        }
        if unknown_cohorts:
            unknown = ', '.join(sorted(unknown_cohorts))
            raise ValueError(
                f'condition {condition_id!r} distinct comorbidity contrast {contrast.id!r} '
                f'references unknown cohorts: {unknown}'
            )

        cohort_a = subgroups[contrast.cohort_a_id]
        cohort_b = subgroups[contrast.cohort_b_id]
        if cohort_a.axis != 'comorbidity' or cohort_b.axis != 'comorbidity':
            raise ValueError(
                f'condition {condition_id!r} distinct comorbidity contrast {contrast.id!r} '
                'must use comorbidity cohorts'
            )
        if cohort_a.level_id != 'present' or cohort_b.level_id != 'present':
            raise ValueError(
                f'condition {condition_id!r} distinct comorbidity contrast {contrast.id!r} '
                'must use present comorbidity cohorts'
            )
        if cohort_a.dimension_id == cohort_b.dimension_id:
            raise ValueError(
                f'condition {condition_id!r} distinct comorbidity contrast {contrast.id!r} '
                'must compare different comorbidity dimensions'
            )

        required_present_absent_ids = {
            present_absent_id_by_present_subgroup.get(contrast.cohort_a_id),
            present_absent_id_by_present_subgroup.get(contrast.cohort_b_id),
        }
        if None in required_present_absent_ids:
            raise ValueError(
                f'condition {condition_id!r} distinct comorbidity contrast {contrast.id!r} '
                'uses a present cohort without a matching present/absent contrast'
            )
        missing_allowlist_ids = required_present_absent_ids - allowed_present_absent_ids
        if missing_allowlist_ids:
            missing = ', '.join(sorted(str(item) for item in missing_allowlist_ids))
            raise ValueError(
                f'condition {condition_id!r} distinct comorbidity contrast {contrast.id!r} '
                f'requires allowlisted present/absent contrasts: {missing}'
            )


def _validate_absent_subgroup_surface_forms(subgroups: dict[SubgroupKey, SubgroupOntology]) -> None:
    subgroups_by_dimension: dict[str, list[SubgroupOntology]] = {}
    for subgroup in subgroups.values():
        subgroups_by_dimension.setdefault(subgroup.dimension_id, []).append(subgroup)

    for dimension_id, dimension_subgroups in subgroups_by_dimension.items():
        present_terms = [
            term
            for subgroup in dimension_subgroups
            if subgroup.axis == 'comorbidity' and subgroup.level_id == 'present'
            for term in _negative_subtype_banned_terms(subgroup)
        ]
        for subgroup in dimension_subgroups:
            if subgroup.axis != 'comorbidity' or subgroup.level_id != 'absent':
                continue
            for form in _subgroup_human_forms(subgroup):
                normalized = _normalize_subgroup_text(form)
                if not _looks_like_negative_subgroup_form(normalized):
                    continue
                direct_modifier = next(
                    (
                        modifier
                        for modifier in _BANNED_NEGATIVE_SUBTYPE_MODIFIERS
                        if modifier in normalized.split()
                    ),
                    None,
                )
                if direct_modifier is not None:
                    raise ValueError(
                        f'absent subgroup for {dimension_id!r} uses negative subtype wording: '
                        f'{form!r}; use the broad category instead'
                    )
                if not present_terms:
                    continue
                matched = next((term for term in present_terms if term in normalized), None)
                if matched is not None:
                    raise ValueError(
                        f'absent subgroup for {dimension_id!r} uses negative subtype wording: '
                        f'{form!r}; use the broad category instead'
                    )


def _negative_subtype_banned_terms(subgroup: SubgroupOntology) -> list[str]:
    terms: list[str] = []
    for form in _subgroup_human_forms(subgroup):
        term = _positive_subgroup_core_term(form)
        if term is None:
            continue
        first_word = term.split(maxsplit=1)[0]
        if first_word in _BANNED_NEGATIVE_SUBTYPE_MODIFIERS or ' without ' in f' {term} ':
            terms.append(term)
    return terms


def _subgroup_human_forms(subgroup: SubgroupOntology) -> list[str]:
    return [subgroup.label, *subgroup.aliases, *subgroup.surface_phrases]


def _positive_subgroup_core_term(text: str) -> str | None:
    normalized = _normalize_subgroup_text(text)
    prefixes = (
        'patients with ',
        'patient with ',
        'with ',
        'history of ',
    )
    for prefix in prefixes:
        if normalized.startswith(prefix):
            return normalized.removeprefix(prefix).strip()
    return normalized or None


def _looks_like_negative_subgroup_form(text: str) -> bool:
    return (
        text.startswith(('no ', 'without ', 'patients without ', 'patient without '))
        or ' no ' in f' {text} '
        or ' without ' in f' {text} '
    )


def _normalize_subgroup_text(text: str) -> str:
    return ' '.join(str(text).casefold().replace('-', ' ').split())


class QueryPlanFacet(BenchmarkPydanticModel):
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

    model_config = ConfigDict(extra='forbid', populate_by_name=True)


class QueryPlanSpec(BenchmarkPydanticModel):
    evidence_profile_id: str
    cohort_contrast_id: str
    cohort_contrast_family: CohortContrastFamily
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


class QueryPlan(BenchmarkPydanticModel):
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
    cohort_contrast_family: CohortContrastFamily
    cohort_dimension_id: str
    primary_axis: ClinicalAxis
    secondary_axis: ClinicalAxis
    dominant_primary_facet_id: str
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
            'cohort_contrast_family': self.cohort_contrast_family,
            'cohort_dimension_id': self.cohort_dimension_id,
            'primary_axis': self.primary_axis,
            'secondary_axis': self.secondary_axis,
            'dominant_primary_facet_id': self.dominant_primary_facet_id,
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
    facet_priority: Literal['primary', 'secondary'] | None
    is_gold: bool
    distractor_type: str | None
    admission_id: str
    patient_id: str
    patient_age: int
    patient_sex: PatientSex
    clinical_subgroup_phrase: str
    note_style: str
    chunk_surface_group: ChunkSurfaceGroup = 'seen'
    split: DataSplit
    must_mention: list[str] = Field(default_factory=list)
    must_not_mention: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def _validate_axis_payload(self) -> ClinicalFact:
        payload = parse_axis_payload(self.axis_payload_json)
        if payload.axis != self.axis:
            raise ValueError(f'axis payload {payload.axis!r} does not match {self.axis!r}')
        return self


class ChunkGenerationCacheEntry(BenchmarkPydanticModel):
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
    outer_template_family: str | None = None
    outer_template_id: str | None = None
    axis_template_family: str | None = None
    axis_template_id: str | None = None


class ChunkRow(ClinicalFact):
    chunk_id: str
    text: str
    approx_words: int
    text_generation_source: Literal['llm', 'fallback', 'cache']
    llm_attempted: bool
    llm_rejected: bool
    generation_cache_hit: bool
    generation_cache_hit_kind: Literal['miss', 'fact_id', 'reuse_key']
    outer_template_family: str | None = None
    outer_template_id: str | None = None
    axis_template_family: str | None = None
    axis_template_id: str | None = None
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
        outer_template_family: str | None = None,
        outer_template_id: str | None = None,
        axis_template_family: str | None = None,
        axis_template_id: str | None = None,
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
            outer_template_family=outer_template_family,
            outer_template_id=outer_template_id,
            axis_template_family=axis_template_family,
            axis_template_id=axis_template_id,
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
            outer_template_family=state.outer_template_family,
            outer_template_id=state.outer_template_id,
            axis_template_family=state.axis_template_family,
            axis_template_id=state.axis_template_id,
        )


class TemplateSpec(BenchmarkPydanticModel):
    id: str
    template: str


class SurfaceTemplateBucket(BenchmarkPydanticModel):
    seen: list[TemplateSpec] = Field(min_length=1)
    heldout: list[TemplateSpec] = Field(min_length=1)

    def templates_for_group(self, surface_group: ChunkSurfaceGroup) -> list[TemplateSpec]:
        if surface_group == 'seen':
            return self.seen
        return self.heldout


class CohortEvidenceTemplates(BenchmarkPydanticModel):
    comorbidity_present: SurfaceTemplateBucket
    comorbidity_reference: SurfaceTemplateBucket


class SemanticAxisBinTemplates(BenchmarkPydanticModel):
    seen: dict[AxisTemplateFamily, list[TemplateSpec]]
    heldout: dict[AxisTemplateFamily, list[TemplateSpec]]

    @model_validator(mode='after')
    def _validate_surface_counts(self) -> SemanticAxisBinTemplates:
        self._validate_group('seen', self.seen, SEEN_AXIS_TEMPLATE_FAMILIES)
        self._validate_group('heldout', self.heldout, HELDOUT_AXIS_TEMPLATE_FAMILIES)
        return self

    @staticmethod
    def _validate_group(
        surface_group: ChunkSurfaceGroup,
        templates: dict[AxisTemplateFamily, list[TemplateSpec]],
        expected_families: list[AxisTemplateFamily],
    ) -> None:
        expected = set(expected_families)
        actual = set(templates)
        if actual != expected:
            raise ValueError(
                f'semantic_hardened {surface_group} families must be {sorted(expected)}; '
                f'got {sorted(actual)}'
            )
        empty = [family for family, specs in templates.items() if not specs]
        if empty:
            raise ValueError(f'semantic_hardened {surface_group} families are empty: {empty}')

    def template_specs(
        self,
        surface_group: ChunkSurfaceGroup,
    ) -> list[tuple[AxisTemplateFamily, TemplateSpec]]:
        templates = self.seen if surface_group == 'seen' else self.heldout
        families = SEEN_AXIS_TEMPLATE_FAMILIES if surface_group == 'seen' else HELDOUT_AXIS_TEMPLATE_FAMILIES
        return [
            (family, spec)
            for family in families
            for spec in templates.get(family, [])
        ]


class ChunkAxisSentenceTemplates(BenchmarkPydanticModel):
    ontology_explicit: dict[ClinicalAxis, dict[str, list[str]]]
    semantic_hardened: dict[ClinicalAxis, dict[str, SemanticAxisBinTemplates]]


class ChunkTemplateUtils(BenchmarkPydanticModel):
    hidden_benchmark_terms: list[str]
    duration_phrase_templates: list[str]
    note_style_templates: dict[str, SurfaceTemplateBucket]
    cohort_evidence_templates: CohortEvidenceTemplates
    axis_sentence_templates: ChunkAxisSentenceTemplates

    @model_validator(mode='after')
    def _validate_semantic_hardened_template_inventory(self) -> ChunkTemplateUtils:
        template_ids: set[str] = set()
        duplicate_ids: set[str] = set()
        for bucket in self.note_style_templates.values():
            for spec in [*bucket.seen, *bucket.heldout]:
                if spec.id in template_ids:
                    duplicate_ids.add(spec.id)
                template_ids.add(spec.id)
        for bucket in (
            self.cohort_evidence_templates.comorbidity_present,
            self.cohort_evidence_templates.comorbidity_reference,
        ):
            for spec in [*bucket.seen, *bucket.heldout]:
                if spec.id in template_ids:
                    duplicate_ids.add(spec.id)
                template_ids.add(spec.id)
        for axis, bins in self.axis_sentence_templates.semantic_hardened.items():
            seen_families_by_bin: set[frozenset[AxisTemplateFamily]] = set()
            heldout_families_by_bin: set[frozenset[AxisTemplateFamily]] = set()
            for value_bin, grouped in bins.items():
                seen_specs = grouped.template_specs('seen')
                heldout_specs = grouped.template_specs('heldout')
                total = len(seen_specs) + len(heldout_specs)
                if len(seen_specs) != 5 or len(heldout_specs) != 3 or total != 8:
                    raise ValueError(
                        f'semantic_hardened {axis}/{value_bin} must define exactly '
                        f'5 seen and 3 heldout templates; got {len(seen_specs)} seen '
                        f'and {len(heldout_specs)} heldout'
                    )
                seen_families_by_bin.add(frozenset(family for family, _ in seen_specs))
                heldout_families_by_bin.add(frozenset(family for family, _ in heldout_specs))
                for _, spec in [*seen_specs, *heldout_specs]:
                    if spec.id in template_ids:
                        duplicate_ids.add(spec.id)
                    template_ids.add(spec.id)
            if len(seen_families_by_bin) != 1 or len(heldout_families_by_bin) != 1:
                raise ValueError(
                    f'semantic_hardened template families must be identical across bins '
                    f'for axis {axis}'
                )
        if duplicate_ids:
            raise ValueError(f'duplicate chunk template ids: {sorted(duplicate_ids)}')
        return self


class QueryTemplateSpec(BenchmarkPydanticModel):
    id: str
    template: str


class AnswerTemplateSpec(BenchmarkPydanticModel):
    template: str


class QueryTemplateData(BenchmarkPydanticModel):
    query_templates: dict[QueryStructure, dict[QueryFocusMode, list[QueryTemplateSpec]]]
    answer_template: AnswerTemplateSpec

    @model_validator(mode='before')
    @classmethod
    def _normalize_legacy_query_templates(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        hydrated = dict(value)
        query_templates = hydrated.get('query_templates')
        if isinstance(query_templates, list):
            hydrated['query_templates'] = {
                structure: {mode: query_templates for mode in QUERY_FOCUS_MODE_LIST}
                for structure in QUERY_STRUCTURE_LIST
            }
        elif isinstance(query_templates, dict):
            normalized: dict[str, object] = {}
            changed = False
            for structure, specs_by_mode in query_templates.items():
                if isinstance(specs_by_mode, list):
                    normalized[str(structure)] = {
                        mode: specs_by_mode for mode in QUERY_FOCUS_MODE_LIST
                    }
                    changed = True
                else:
                    normalized[str(structure)] = specs_by_mode
            if changed:
                hydrated['query_templates'] = normalized
        return hydrated

    @model_validator(mode='after')
    def _validate_query_template_structures(self) -> QueryTemplateData:
        if set(self.query_templates) != set(QUERY_STRUCTURE_LIST):
            raise ValueError(
                f'query_templates must define exactly these structures: {QUERY_STRUCTURE_LIST}'
            )
        expected_ids: list[str] | None = None
        for structure in QUERY_STRUCTURE_LIST:
            specs_by_mode = self.query_templates[structure]
            if set(specs_by_mode) != set(QUERY_FOCUS_MODE_LIST):
                raise ValueError(
                    f'query_templates.{structure} must define exactly these focus modes: '
                    f'{QUERY_FOCUS_MODE_LIST}'
                )
            for focus_mode in QUERY_FOCUS_MODE_LIST:
                specs = specs_by_mode[focus_mode]
                ids = [spec.id for spec in specs]
                if len(ids) != len(set(ids)):
                    raise ValueError(
                        f'duplicate query template ids for {structure}/{focus_mode}: {ids}'
                    )
                if expected_ids is None:
                    expected_ids = ids
                elif ids != expected_ids:
                    raise ValueError(
                        'query template structures and focus modes must define the same ids '
                        'in the same order'
                    )
        return self
