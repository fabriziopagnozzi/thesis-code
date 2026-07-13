from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from experiments.medical_dataset_gen.schemas.generation_schemas import (
    AxisPairOntology,
    AxisPairProfile,
    ClinicalAxis,
    CohortContrast,
    CohortContrastFamily,
    ConditionKey,
    ConditionOntology,
    MedicalOntology,
    SubgroupKey,
    SubgroupOntology,
)
from experiments.medical_dataset_gen.schemas.global_config_schemas import (
    ExperimentCfg,
)
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths


@dataclass(frozen=True)
class ResolvedAxisPairGenerationPolicy:
    allowed_primary_axes: frozenset[ClinicalAxis]
    blocked_profile_ids: frozenset[str]
    rationale: str | None


@dataclass(frozen=True)
class ResolvedCohortContrast:
    id: str
    family: CohortContrastFamily
    dimension_id: str
    cohort_a_id: str
    cohort_b_id: str


def load_ontology(cfg: ExperimentCfg) -> MedicalOntology:
    path = _ontology_path(cfg)
    with open(path) as f:
        raw_ontology = yaml.safe_load(f)

    if not isinstance(raw_ontology, dict):
        raise ValueError('Ontology file must contain a mapping at the top level')
    return MedicalOntology.model_validate(raw_ontology)


def get_selected_conditions(
    ontology: MedicalOntology, n_conditions: int
) -> list[tuple[ConditionKey, ConditionOntology]]:
    items = list(ontology.conditions.items())
    if n_conditions > len(items):
        raise ValueError(f'Config asks for {n_conditions} conditions but ontology has {len(items)}')
    return items[:n_conditions]


def make_subgroup_pairs_for_condition(
    ontology: MedicalOntology,
    condition_id: ConditionKey,
) -> list[
    tuple[
        ResolvedCohortContrast,
        tuple[SubgroupKey, SubgroupOntology],
        tuple[SubgroupKey, SubgroupOntology],
    ]
]:
    condition = ontology.conditions[condition_id]
    allowed_comorbidity_contrasts = set(condition.allowed_comorbidity_contrast_ids)
    resolved: list[
        tuple[
            ResolvedCohortContrast,
            tuple[SubgroupKey, SubgroupOntology],
            tuple[SubgroupKey, SubgroupOntology],
        ]
    ] = []
    for contrast in ontology.cohort_contrasts:
        if _is_global_demographic_contrast(ontology, contrast):
            family: CohortContrastFamily = 'demographic'
        elif contrast.id in allowed_comorbidity_contrasts:
            family = 'comorbidity_present_absent'
        else:
            continue
        resolved.append(
            (
                ResolvedCohortContrast(
                    id=contrast.id,
                    family=family,
                    dimension_id=contrast.dimension_id,
                    cohort_a_id=contrast.cohort_a_id,
                    cohort_b_id=contrast.cohort_b_id,
                ),
                (contrast.cohort_a_id, ontology.subgroups[contrast.cohort_a_id]),
                (contrast.cohort_b_id, ontology.subgroups[contrast.cohort_b_id]),
            )
        )

    for contrast in condition.allowed_distinct_comorbidity_contrasts:
        cohort_a = ontology.subgroups[contrast.cohort_a_id]
        cohort_b = ontology.subgroups[contrast.cohort_b_id]
        dimension_id = f'{cohort_a.dimension_id}_vs_{cohort_b.dimension_id}'
        resolved.append(
            (
                ResolvedCohortContrast(
                    id=contrast.id,
                    family='distinct_comorbidity',
                    dimension_id=dimension_id,
                    cohort_a_id=contrast.cohort_a_id,
                    cohort_b_id=contrast.cohort_b_id,
                ),
                (contrast.cohort_a_id, cohort_a),
                (contrast.cohort_b_id, cohort_b),
            )
        )
    return resolved


def _is_global_demographic_contrast(
    ontology: MedicalOntology,
    contrast: CohortContrast,
) -> bool:
    return (
        ontology.subgroups[contrast.cohort_a_id].axis == 'demographic'
        and ontology.subgroups[contrast.cohort_b_id].axis == 'demographic'
    )


def get_axis_pair_profiles(
    ontology: MedicalOntology, left: ClinicalAxis, right: ClinicalAxis
) -> list[AxisPairProfile]:
    pair = get_axis_pair_ontology(ontology, left, right)
    if pair.axes == (left, right):
        return pair.profiles
    return [
        profile.model_copy(
            update={
                'cohort_a_bins': tuple(reversed(profile.cohort_a_bins)),
                'cohort_b_bins': tuple(reversed(profile.cohort_b_bins)),
            }
        )
        for profile in pair.profiles
    ]


def get_axis_pair_ontology(
    ontology: MedicalOntology, left: ClinicalAxis, right: ClinicalAxis
) -> AxisPairOntology:
    requested = {left, right}
    for pair in ontology.axis_pairs:
        if set(pair.axes) == requested:
            return pair
    raise KeyError(f'missing clinical axis pair: {left}, {right}')


def resolve_axis_pair_generation_policy(
    ontology: MedicalOntology,
    cfg: ExperimentCfg | None,
    condition_id: ConditionKey,
    left: ClinicalAxis,
    right: ClinicalAxis,
) -> ResolvedAxisPairGenerationPolicy:
    pair = get_axis_pair_ontology(ontology, left, right)
    pair_profiles = get_axis_pair_profiles(ontology, left, right)
    allowed_primary_axes = (
        set(pair.axes) if pair.allowed_primary_axes is None else set(pair.allowed_primary_axes)
    )
    blocked_profile_ids = set(pair.blocked_profile_ids)
    rationale = pair.rationale
    for override in pair.condition_overrides:
        if override.condition_id != condition_id:
            continue
        if override.allowed_primary_axes is not None:
            allowed_primary_axes = set(override.allowed_primary_axes)
        blocked_profile_ids.update(override.blocked_profile_ids)
        if override.rationale:
            rationale = override.rationale
        break
    if cfg is not None:
        config_override = cfg.generation.axis_pair_policy_override(left, right)
        if config_override is not None:
            known_profile_ids = {profile.id for profile in pair_profiles}
            unknown_blocked = set(config_override.blocked_profile_ids) - known_profile_ids
            if unknown_blocked:
                unknown = ', '.join(sorted(unknown_blocked))
                raise ValueError(
                    'generation.axis_pair_policy_overrides blocks unknown profiles: '
                    f'{left}/{right}: {unknown}'
                )
            if config_override.allowed_primary_axes is not None:
                allowed_primary_axes = set(config_override.allowed_primary_axes)
            blocked_profile_ids.update(config_override.blocked_profile_ids)
            if config_override.rationale:
                rationale = config_override.rationale
            for override in config_override.condition_overrides:
                if override.condition_id != condition_id:
                    continue
                unknown_override_profiles = set(override.blocked_profile_ids) - known_profile_ids
                if unknown_override_profiles:
                    unknown = ', '.join(sorted(unknown_override_profiles))
                    raise ValueError(
                        'generation.axis_pair_policy_overrides.condition_overrides blocks '
                        f'unknown profiles: {left}/{right}: {unknown}'
                    )
                if override.allowed_primary_axes is not None:
                    allowed_primary_axes = set(override.allowed_primary_axes)
                blocked_profile_ids.update(override.blocked_profile_ids)
                if override.rationale:
                    rationale = override.rationale
                break
    return ResolvedAxisPairGenerationPolicy(
        allowed_primary_axes=frozenset(allowed_primary_axes),
        blocked_profile_ids=frozenset(blocked_profile_ids),
        rationale=rationale,
    )


def other_subgroups(
    ontology: MedicalOntology, excluded_ids: set[str]
) -> list[tuple[SubgroupKey, SubgroupOntology]]:
    return [
        (sid, subgroup) for sid, subgroup in ontology.subgroups.items() if sid not in excluded_ids
    ]


def outlier_subgroups(
    ontology: MedicalOntology, excluded_ids: set[str]
) -> list[tuple[SubgroupKey, SubgroupOntology]]:
    return [
        (sid, subgroup)
        for sid, subgroup in ontology.subgroups.items()
        if sid not in excluded_ids and _is_outlier_eligible_subgroup(sid, subgroup)
    ]


def _is_outlier_eligible_subgroup(subgroup_id: str, subgroup: SubgroupOntology) -> bool:
    if subgroup_id.startswith('no_'):
        return False
    return not (subgroup.axis == 'comorbidity' and subgroup.level_id == 'absent')


def other_conditions(
    ontology: MedicalOntology, excluded_id: str
) -> list[tuple[ConditionKey, ConditionOntology]]:
    return [
        (cid, condition) for cid, condition in ontology.conditions.items() if cid != excluded_id
    ]


def get_axis_bins(ontology: MedicalOntology, axis: ClinicalAxis) -> list[str]:
    try:
        bins = ontology.clinical_axes[axis].bins
    except KeyError as exc:
        raise ValueError(f'Ontology is missing clinical axis metadata for {axis}') from exc
    if not bins:
        raise ValueError(f'Ontology axis {axis} must declare at least one value bin')
    return bins


def exclusive_distinct_subgroup_label(
    included: SubgroupOntology,
    excluded: SubgroupOntology,
) -> str:
    return f'{included.label} but without {subgroup_core_phrase(excluded)}'


def exclusive_distinct_subgroup_phrase(
    included: SubgroupOntology,
    excluded: SubgroupOntology,
) -> str:
    return f'{subgroup_core_phrase(included)} without {subgroup_core_phrase(excluded)}'


def subgroup_core_phrase(subgroup: SubgroupOntology) -> str:
    special_cases = {
        'immunosuppression': 'immunosuppression',
        'frailty': 'baseline frailty',
    }
    if subgroup.value in special_cases:
        return special_cases[subgroup.value]

    for form in [subgroup.label, *subgroup.aliases, *subgroup.surface_phrases]:
        core = _strip_patient_prefixes(form)
        if not core or core.startswith('patients '):
            continue
        if core.endswith(' patients'):
            continue
        return core
    return subgroup.label


def _strip_patient_prefixes(text: str) -> str:
    normalized = ' '.join(text.split())
    for prefix in (
        'patients with ',
        'patient with ',
        'patients receiving ',
        'history of ',
        'with ',
    ):
        if normalized.lower().startswith(prefix):
            return normalized[len(prefix) :].strip()
    return normalized


def _ontology_path(cfg: ExperimentCfg) -> Path:
    if cfg.generation.ontology_path:
        return Path(cfg.generation.ontology_path)
    return MedicalDatasetGenPaths.default_ontology_path
