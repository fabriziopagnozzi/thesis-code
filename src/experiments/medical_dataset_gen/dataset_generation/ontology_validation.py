"""Cross-reference and admissibility validation for the clinical ontology."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

from experiments.medical_dataset_gen.dataset_generation.schema_types import ClinicalAxis

if TYPE_CHECKING:
    from experiments.medical_dataset_gen.dataset_generation.ontology_schemas import (
        AxisPairOntology,
        AxisPairProfile,
        ClinicalAxisOntology,
        CohortContrast,
        ConditionOntology,
        MedicalOntology,
        SubgroupOntology,
    )


_BANNED_NEGATIVE_SUBTYPE_MODIFIERS = frozenset(
    {
        'complicated',
        'metastatic',
        'mild',
        'uncomplicated',
    }
)


def validate_ontology(ontology: MedicalOntology) -> None:
    declared_axes = set(ontology.clinical_axes)
    contrast_by_id = _cohort_contrasts_by_id(ontology.cohort_contrasts)
    _validate_condition_references(
        ontology.conditions,
        ontology.clinical_axes,
        ontology.subgroups,
        contrast_by_id,
        declared_axes,
    )
    _validate_cohort_contrast_dimensions(ontology.cohort_contrasts, ontology.subgroups)
    _validate_absent_subgroup_surface_forms(ontology.subgroups)
    _validate_axis_pair_inventory(ontology.axis_pairs, ontology.clinical_axes, declared_axes)


def _cohort_contrasts_by_id(contrasts: list[CohortContrast]) -> dict[str, CohortContrast]:
    contrast_by_id: dict[str, CohortContrast] = {}
    for contrast in contrasts:
        if contrast.id in contrast_by_id:
            raise ValueError(f'duplicate cohort contrast id: {contrast.id!r}')
        contrast_by_id[contrast.id] = contrast
    return contrast_by_id


def _validate_condition_references(
    conditions: dict[str, ConditionOntology],
    clinical_axes: dict[ClinicalAxis, ClinicalAxisOntology],
    subgroups: dict[str, SubgroupOntology],
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
    subgroups: dict[str, SubgroupOntology],
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
    subgroups: dict[str, SubgroupOntology],
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


def _validate_absent_subgroup_surface_forms(subgroups: dict[str, SubgroupOntology]) -> None:
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
