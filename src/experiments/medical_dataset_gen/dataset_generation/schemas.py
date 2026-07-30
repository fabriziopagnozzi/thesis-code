from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from experiments.medical_dataset_gen.utils.global_utils import get_literals

type ClinicalAxis = Literal[
    'treatment_duration',
    'rehab_outcome',
    'complication_burden',
    'acute_clinical_course',
    'care_intensity',
    'diagnostic_evidence_type',
]
CLINICAL_AXIS_LIST = list[ClinicalAxis](get_literals(ClinicalAxis))


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

type ChunkPoolScope = Literal['query_local']
type ChunkTextStyle = Literal['ontology_explicit', 'semantic_hardened']
CHUNK_TEXT_STYLE_LIST = list[ChunkTextStyle](get_literals(ChunkTextStyle))

type QueryFocusMode = Literal['list', 'natural']
QUERY_FOCUS_MODE_LIST = list[QueryFocusMode](get_literals(QueryFocusMode))

type QueryStructure = Literal['unbalanced', 'balanced', 'label_only']
QUERY_STRUCTURE_LIST = list[QueryStructure](get_literals(QueryStructure))
LABEL_ONLY_CANONICAL_FOCUS_MODE: QueryFocusMode = 'natural'

type QueryWordingMode = tuple[QueryStructure, QueryFocusMode]
QUERY_WORDING_MODE_LIST: list[QueryWordingMode] = [
    ('unbalanced', 'list'),
    ('unbalanced', 'natural'),
    ('balanced', 'list'),
    ('balanced', 'natural'),
    ('label_only', LABEL_ONLY_CANONICAL_FOCUS_MODE),
]


def query_focus_modes_for_structure(structure: QueryStructure) -> list[QueryFocusMode]:
    """Return the meaningful focus modes without inventing label-only subvariants."""
    if structure == 'label_only':
        return [LABEL_ONLY_CANONICAL_FOCUS_MODE]
    return QUERY_FOCUS_MODE_LIST


def canonical_query_focus_mode(
    structure: QueryStructure,
    focus_mode: QueryFocusMode,
) -> QueryFocusMode:
    if structure == 'label_only':
        return LABEL_ONLY_CANONICAL_FOCUS_MODE
    return focus_mode


type ChunkSurfaceGroup = Literal['seen', 'heldout']
CHUNK_SURFACE_GROUP_LIST = list[ChunkSurfaceGroup](get_literals(ChunkSurfaceGroup))
type ChunkSurfacePolicy = Literal['split_heldout', 'seen_only', 'heldout_only']
type ConditionAnchor = Literal['outer_template', 'axis_evidence']
type AxisTemplateFamily = Literal[
    'direct_fact',
    'temporal_course',
    'clinical_assessment',
    'contrast_or_alternative',
]
AXIS_TEMPLATE_FAMILY_LIST = list[AxisTemplateFamily](get_literals(AxisTemplateFamily))
SEEN_AXIS_TEMPLATE_FAMILIES = AXIS_TEMPLATE_FAMILY_LIST
HELDOUT_AXIS_TEMPLATE_FAMILIES = AXIS_TEMPLATE_FAMILY_LIST
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


def axis_payload_required_phrase(payload: AxisFactPayload) -> str:
    if isinstance(payload, TreatmentDurationPayload):
        return f'{payload.duration_days} days of {payload.treatment}'
    if isinstance(payload, RehabOutcomePayload):
        return payload.outcome
    return payload.detail


def _validated_axis_payload(axis: ClinicalAxis, axis_payload_json: str) -> AxisFactPayload:
    payload = parse_axis_payload(axis_payload_json)
    if payload.axis != axis:
        raise ValueError(f'axis payload {payload.axis!r} does not match {axis!r}')
    return payload


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
        _validated_axis_payload(self.axis, self.axis_payload_json)
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
    # This is deliberately independent from the human-readable ontology label.
    # Query templates use it as the stable benchmark-facing axis wording.
    query_label: str
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
        declared_axes = set(self.clinical_axes)
        contrast_by_id = _cohort_contrasts_by_id(self.cohort_contrasts)
        _validate_condition_references(
            self.conditions,
            self.clinical_axes,
            self.subgroups,
            contrast_by_id,
            declared_axes,
        )
        _validate_cohort_contrast_dimensions(self.cohort_contrasts, self.subgroups)
        _validate_absent_subgroup_surface_forms(self.subgroups)
        _validate_axis_pair_inventory(self.axis_pairs, self.clinical_axes, declared_axes)
        return self


_BANNED_NEGATIVE_SUBTYPE_MODIFIERS = frozenset({
    'complicated',
    'metastatic',
    'mild',
    'uncomplicated',
})


def _cohort_contrasts_by_id(contrasts: list[CohortContrast]) -> dict[str, CohortContrast]:
    contrast_by_id: dict[str, CohortContrast] = {}
    for contrast in contrasts:
        if contrast.id in contrast_by_id:
            raise ValueError(f'duplicate cohort contrast id: {contrast.id!r}')
        contrast_by_id[contrast.id] = contrast
    return contrast_by_id


def _validate_condition_references(
    conditions: dict[ConditionKey, ConditionOntology],
    clinical_axes: dict[ClinicalAxis, ClinicalAxisOntology],
    subgroups: dict[SubgroupKey, SubgroupOntology],
    contrast_by_id: dict[str, CohortContrast],
    declared_axes: set[ClinicalAxis],
) -> None:
    for condition_id, condition in conditions.items():
        if set(condition.axis_values) != declared_axes:
            raise ValueError(f'condition {condition_id!r} must define every clinical axis')
        for axis, values in condition.axis_values.items():
            if set(values.bins) != set(clinical_axes[axis].bins):
                raise ValueError(f'condition {condition_id!r} has incomplete bins for {axis!r}')

        allowed_ids = condition.allowed_comorbidity_contrast_ids
        if len(allowed_ids) != len(set(allowed_ids)):
            raise ValueError(f'condition {condition_id!r} repeats allowed comorbidity contrast ids')
        unknown_allowed = set(allowed_ids) - set(contrast_by_id)
        if unknown_allowed:
            unknown = ', '.join(sorted(unknown_allowed))
            raise ValueError(
                f'condition {condition_id!r} allows unknown comorbidity contrasts: {unknown}'
            )

        for contrast_id in allowed_ids:
            contrast = contrast_by_id[contrast_id]
            cohorts = [subgroups[contrast.cohort_a_id], subgroups[contrast.cohort_b_id]]
            if not _is_comorbidity_present_absent_contrast(cohorts):
                raise ValueError(
                    f'condition {condition_id!r} allows non-comorbidity present/absent '
                    f'contrast {contrast_id!r}'
                )

        _validate_distinct_comorbidity_contrasts(
            condition_id,
            condition,
            subgroups,
            contrast_by_id,
        )


def _validate_cohort_contrast_dimensions(
    contrasts: list[CohortContrast],
    subgroups: dict[SubgroupKey, SubgroupOntology],
) -> None:
    for contrast in contrasts:
        cohorts = [subgroups[contrast.cohort_a_id], subgroups[contrast.cohort_b_id]]
        if any(cohort.dimension_id != contrast.dimension_id for cohort in cohorts):
            raise ValueError(f'contrast {contrast.id!r} mixes cohort dimensions')


def _validate_axis_pair_inventory(
    axis_pairs: list[AxisPairOntology],
    clinical_axes: dict[ClinicalAxis, ClinicalAxisOntology],
    declared_axes: set[ClinicalAxis],
) -> None:
    declared_pairs = {frozenset(pair.axes) for pair in axis_pairs}
    expected_pairs = {frozenset(pair) for pair in combinations(declared_axes, 2)}
    if declared_pairs != expected_pairs or len(declared_pairs) != len(axis_pairs):
        raise ValueError('axis_pairs must contain each unordered clinical-axis pair once')

    for pair in axis_pairs:
        _validate_axis_pair(pair, clinical_axes)


def _validate_axis_pair(
    pair: AxisPairOntology,
    clinical_axes: dict[ClinicalAxis, ClinicalAxisOntology],
) -> None:
    if len(pair.profiles) < 2:
        raise ValueError('each axis pair must define at least two joint profiles')

    left, right = pair.axes
    candidate_primary_axes = (
        list(pair.axes) if pair.allowed_primary_axes is None else pair.allowed_primary_axes
    )
    explicitly_suppressed = pair.allowed_primary_axes == []
    if not explicitly_suppressed and not any(
        clinical_axes[axis].allow_as_primary for axis in candidate_primary_axes
    ):
        raise ValueError(f'axis pair {left!r}/{right!r} has no permitted primary axis')

    profile_ids = [profile.id for profile in pair.profiles]
    if len(profile_ids) != len(set(profile_ids)):
        raise ValueError(f'axis pair {left!r}/{right!r} reuses a profile id')

    for profile in pair.profiles:
        _validate_axis_pair_profile(profile, left, right, clinical_axes)


def _validate_axis_pair_profile(
    profile: AxisPairProfile,
    left: ClinicalAxis,
    right: ClinicalAxis,
    clinical_axes: dict[ClinicalAxis, ClinicalAxisOntology],
) -> None:
    cohort_pairs = zip(profile.cohort_a_bins, profile.cohort_b_bins, strict=True)
    if any(a == b for a, b in cohort_pairs):
        raise ValueError(f'profile {profile.id!r} must differ on both axes')
    if not all(
        bins[index] in clinical_axes[axis].bins
        for index, axis in enumerate((left, right))
        for bins in (profile.cohort_a_bins, profile.cohort_b_bins)
    ):
        raise ValueError(f'profile {profile.id!r} contains an unknown axis bin')


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

    def to_query_row(self, query_text: str, *, template_id: str | None = None) -> QueryOutputRow:
        facets_json, logical_form_json = self._json_columns()
        return {
            'query_id': self.query_id,
            'evidence_profile_id': self.evidence_profile_id,
            'pool_id': self.pool_id,
            'outcome_profile_id': self.outcome_profile_id,
            'query_type': self.query_type,
            'template_id': template_id or self.template_id,
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
    condition_anchor: ConditionAnchor = 'outer_template'
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
        payload = _validated_axis_payload(self.axis, self.axis_payload_json)
        if 'condition_anchor' in self.model_fields_set:
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


@dataclass
class ChunkState:
    final_text: str
    outer_template_family: str | None = None
    outer_template_id: str | None = None
    axis_template_family: str | None = None
    axis_template_id: str | None = None


class ChunkRow(ClinicalFact):
    chunk_id: str
    text: str
    approx_words: int
    outer_template_family: str | None = None
    outer_template_id: str | None = None
    axis_template_family: str | None = None
    axis_template_id: str | None = None

    @classmethod
    def from_fact(
        cls,
        fact: ClinicalFact,
        *,
        chunk_id: str,
        final_text: str,
        word_count: int,
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
            outer_template_family=outer_template_family,
            outer_template_id=outer_template_id,
            axis_template_family=axis_template_family,
            axis_template_id=axis_template_id,
        )

    @classmethod
    def from_state(cls, fact: ClinicalFact, *, chunk_id: str, state: ChunkState) -> ChunkRow:
        return cls.from_fact(
            fact,
            chunk_id=chunk_id,
            final_text=state.final_text,
            word_count=len(state.final_text.split()),
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


class NoteStyleTemplates(BenchmarkPydanticModel):
    outer_template: dict[str, SurfaceTemplateBucket]
    axis_evidence: dict[str, SurfaceTemplateBucket]

    @model_validator(mode='after')
    def _validate_matching_families(self) -> NoteStyleTemplates:
        if set(self.outer_template) != set(self.axis_evidence):
            raise ValueError(
                'note-style families must match for outer_template and axis_evidence anchors'
            )
        return self

    def templates_for_anchor(
        self,
        condition_anchor: ConditionAnchor,
    ) -> dict[str, SurfaceTemplateBucket]:
        if condition_anchor == 'outer_template':
            return self.outer_template
        return self.axis_evidence


class PairedAxisSentenceTemplates(BenchmarkPydanticModel):
    """Shared clinical evidence wording for deterministic v4 chunks.

    The template renders the same clinical evidence sentence for both chunk
    styles. The simple style adds an authored interpretation sentence from the
    top-level template data; the hardened style stops after this evidence.
    """

    seen: dict[AxisTemplateFamily, list[TemplateSpec]]
    heldout: dict[AxisTemplateFamily, list[TemplateSpec]]

    @model_validator(mode='after')
    def _validate_surface_counts(self) -> PairedAxisSentenceTemplates:
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
                f'paired axis templates {surface_group} families must be {sorted(expected)}; '
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
        families = (
            SEEN_AXIS_TEMPLATE_FAMILIES
            if surface_group == 'seen'
            else HELDOUT_AXIS_TEMPLATE_FAMILIES
        )
        return [(family, spec) for family in families for spec in templates.get(family, [])]


class ChunkAxisSentenceTemplates(BenchmarkPydanticModel):
    paired: dict[ClinicalAxis, PairedAxisSentenceTemplates]


class ChunkTemplateUtils(BenchmarkPydanticModel):
    hidden_benchmark_terms: list[str]
    note_style_templates: NoteStyleTemplates
    cohort_evidence_templates: CohortEvidenceTemplates
    treatment_course_templates: dict[str, SurfaceTemplateBucket]
    axis_sentence_templates: ChunkAxisSentenceTemplates
    simple_interpretations: dict[ClinicalAxis, dict[str, SurfaceTemplateBucket]]

    @model_validator(mode='after')
    def _validate_template_inventory(self) -> ChunkTemplateUtils:
        template_ids: set[str] = set()
        duplicate_ids: set[str] = set()
        for templates_by_family in (
            self.note_style_templates.outer_template,
            self.note_style_templates.axis_evidence,
        ):
            for bucket in templates_by_family.values():
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
        if not self.treatment_course_templates:
            raise ValueError('treatment_course_templates must not be empty')
        for bucket in self.treatment_course_templates.values():
            for spec in [*bucket.seen, *bucket.heldout]:
                if spec.id in template_ids:
                    duplicate_ids.add(spec.id)
                template_ids.add(spec.id)
        expected_axes = set(CLINICAL_AXIS_LIST)
        actual_axes = set(self.axis_sentence_templates.paired)
        if actual_axes != expected_axes:
            raise ValueError(
                f'paired axis templates must define every clinical axis; '
                f'missing={sorted(expected_axes - actual_axes)}, '
                f'unexpected={sorted(actual_axes - expected_axes)}'
            )
        for axis, paired in self.axis_sentence_templates.paired.items():
            seen_specs = paired.template_specs('seen')
            heldout_specs = paired.template_specs('heldout')
            if len(seen_specs) != len(SEEN_AXIS_TEMPLATE_FAMILIES) or len(heldout_specs) != len(
                HELDOUT_AXIS_TEMPLATE_FAMILIES
            ):
                raise ValueError(
                    f'paired {axis} templates must define exactly '
                    f'{len(SEEN_AXIS_TEMPLATE_FAMILIES)} seen and '
                    f'{len(HELDOUT_AXIS_TEMPLATE_FAMILIES)} heldout families'
                )
            for _, spec in [*seen_specs, *heldout_specs]:
                if spec.id in template_ids:
                    duplicate_ids.add(spec.id)
                template_ids.add(spec.id)
        if set(self.simple_interpretations) != expected_axes:
            raise ValueError('simple_interpretations must define every clinical axis')
        for axis, interpretations_by_bin in self.simple_interpretations.items():
            if not interpretations_by_bin:
                raise ValueError(f'simple_interpretations for {axis!r} must not be empty')
            for value_bin, bucket in interpretations_by_bin.items():
                if not value_bin.strip():
                    raise ValueError(f'simple_interpretations for {axis!r} contains an empty bin')
                for spec in [*bucket.seen, *bucket.heldout]:
                    if spec.id in template_ids:
                        duplicate_ids.add(spec.id)
                    template_ids.add(spec.id)
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
                structure: {
                    mode: query_templates for mode in query_focus_modes_for_structure(structure)
                }
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
            expected_focus_modes = query_focus_modes_for_structure(structure)
            if set(specs_by_mode) != set(expected_focus_modes):
                raise ValueError(
                    f'query_templates.{structure} must define exactly these focus modes: '
                    f'{expected_focus_modes}'
                )
            for focus_mode in expected_focus_modes:
                specs = specs_by_mode[focus_mode]
                ids = [spec.id for spec in specs]
                if not ids:
                    raise ValueError(f'query_templates.{structure}.{focus_mode} must not be empty')
                if len(ids) != len(set(ids)):
                    raise ValueError(
                        f'duplicate query template ids for {structure}/{focus_mode}: {ids}'
                    )
                if structure == 'label_only':
                    continue
                if expected_ids is None:
                    expected_ids = ids
                elif ids != expected_ids:
                    raise ValueError(
                        'query template structures and focus modes must define the same ids '
                        'in the same order'
                    )
        return self
