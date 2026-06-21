from __future__ import annotations

from pathlib import Path

import yaml

from experiments.medical_dataset_gen.schemas.generation_schemas import (
    CLINICAL_AXIS_LIST,
    ClinicalAxis,
    CohortContrast,
    ConditionKey,
    ConditionOntology,
    MedicalOntology,
    SubgroupKey,
    SubgroupOntology,
)
from experiments.medical_dataset_gen.utils.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
)


def load_ontology(cfg: ExperimentCfg) -> MedicalOntology:
    path = _ontology_path(cfg)
    with open(path) as f:
        raw_ontology = yaml.safe_load(f)

    if not isinstance(raw_ontology, dict):
        raise ValueError('Ontology file must contain a mapping at the top level')
    design_path = MedicalDatasetGenPaths.root / 'data_templates' / 'v2_benchmark_design.yaml'
    with open(design_path) as f:
        design = yaml.safe_load(f) or {}
    axis_template_path = (
        MedicalDatasetGenPaths.root / 'data_templates' / 'chunk_v2_axis_templates.yaml'
    )
    with open(axis_template_path) as f:
        axis_templates = yaml.safe_load(f) or {}

    raw_ontology['subgroups'] = design['cohort_levels']
    raw_ontology['cohort_contrasts'] = design['cohort_contrasts']
    raw_ontology['axis_pair_profiles'] = design['axis_pair_profiles']
    raw_ontology['clinical_axes'].update(
        {
            'complication_burden': {
                'label': 'in-hospital complication burden',
                'exact_terms': ['complication burden', 'in-hospital complications'],
                'synonym_terms': ['safety outcome', 'adverse-event burden'],
                'bins': ['none_documented', 'limited_complication', 'major_complication'],
            },
            'acute_clinical_course': {
                'label': 'acute clinical course',
                'exact_terms': ['acute clinical course', 'clinical response trajectory'],
                'synonym_terms': ['inpatient response pattern', 'stabilization trajectory'],
                'bins': ['early_stabilization', 'delayed_improvement', 'refractory_or_worsening'],
            },
            'care_intensity': {
                'label': 'care intensity',
                'exact_terms': ['care intensity', 'level of care'],
                'synonym_terms': ['monitoring needs', 'inpatient resource use'],
                'bins': ['ward_level', 'monitored_or_procedural', 'intensive_care'],
            },
        }
    )
    condition_values = axis_templates['condition_new_axis_values']
    for condition_id, condition in raw_ontology['conditions'].items():
        condition['new_axis_values'] = condition_values[condition_id]

    ontology = MedicalOntology.model_validate(raw_ontology)
    _validate_v2_ontology(ontology)
    return ontology


def _validate_v2_ontology(ontology: MedicalOntology) -> None:
    if list(ontology.clinical_axes) != CLINICAL_AXIS_LIST:
        raise ValueError(
            f'v2 clinical axes must be ordered as {CLINICAL_AXIS_LIST}, '
            f'got {list(ontology.clinical_axes)}'
        )
    if len(ontology.cohort_contrasts) != 16:
        raise ValueError(f'v2 requires 16 cohort contrasts, got {len(ontology.cohort_contrasts)}')
    expected_pairs = {
        axis_pair_key(left, right)
        for index, left in enumerate(CLINICAL_AXIS_LIST)
        for right in CLINICAL_AXIS_LIST[index + 1 :]
    }
    if set(ontology.axis_pair_profiles) != expected_pairs:
        raise ValueError('v2 axis-pair profiles must cover all ten unordered axis pairs')
    for pair_key, profiles in ontology.axis_pair_profiles.items():
        if len(profiles) < 2:
            raise ValueError(f'axis pair {pair_key!r} requires at least two balanced profiles')
        left_axis, right_axis = pair_key.split('__')
        valid_left = set(ontology.clinical_axes[left_axis].bins)
        valid_right = set(ontology.clinical_axes[right_axis].bins)
        for profile in profiles:
            if profile.cohort_a_bins[0] not in valid_left or profile.cohort_b_bins[0] not in valid_left:
                raise ValueError(f'profile {pair_key}/{profile.id} has an invalid left-axis bin')
            if profile.cohort_a_bins[1] not in valid_right or profile.cohort_b_bins[1] not in valid_right:
                raise ValueError(f'profile {pair_key}/{profile.id} has an invalid right-axis bin')
            if any(
                left == right
                for left, right in zip(
                    profile.cohort_a_bins, profile.cohort_b_bins, strict=True
                )
            ):
                raise ValueError(f'profile {pair_key}/{profile.id} must differ on both axes')
    for contrast in ontology.cohort_contrasts:
        left = ontology.subgroups[contrast.cohort_a_id]
        right = ontology.subgroups[contrast.cohort_b_id]
        if (
            left.dimension_id != contrast.dimension_id
            or right.dimension_id != contrast.dimension_id
        ):
            raise ValueError(f'contrast {contrast.id!r} mixes cohort dimensions')
    for condition_id, condition in ontology.conditions.items():
        for axis in CLINICAL_AXIS_LIST[2:]:
            bins = condition.new_axis_values.get(axis, {})
            expected_bins = set(ontology.clinical_axes[axis].bins)
            if set(bins) != expected_bins:
                raise ValueError(
                    f'condition {condition_id!r} axis {axis!r} must define {expected_bins}'
                )


def get_selected_conditions(
    ontology: MedicalOntology, n_conditions: int
) -> list[tuple[ConditionKey, ConditionOntology]]:
    items = list(ontology.conditions.items())
    if n_conditions > len(items):
        raise ValueError(f'Config asks for {n_conditions} conditions but ontology has {len(items)}')
    return items[:n_conditions]


def make_subgroup_pairs(
    ontology: MedicalOntology,
) -> list[
    tuple[
        CohortContrast, tuple[SubgroupKey, SubgroupOntology], tuple[SubgroupKey, SubgroupOntology]
    ]
]:
    return [
        (
            contrast,
            (contrast.cohort_a_id, ontology.subgroups[contrast.cohort_a_id]),
            (contrast.cohort_b_id, ontology.subgroups[contrast.cohort_b_id]),
        )
        for contrast in ontology.cohort_contrasts
    ]


def axis_pair_key(left: ClinicalAxis, right: ClinicalAxis) -> str:
    left_index = CLINICAL_AXIS_LIST.index(left)
    right_index = CLINICAL_AXIS_LIST.index(right)
    first, second = (left, right) if left_index < right_index else (right, left)
    return f'{first}__{second}'


def other_subgroups(
    ontology: MedicalOntology, excluded_ids: set[str]
) -> list[tuple[SubgroupKey, SubgroupOntology]]:
    return [
        (sid, subgroup) for sid, subgroup in ontology.subgroups.items() if sid not in excluded_ids
    ]


def other_conditions(
    ontology: MedicalOntology, excluded_id: str
) -> list[tuple[ConditionKey, ConditionOntology]]:
    return [
        (cid, condition) for cid, condition in ontology.conditions.items() if cid != excluded_id
    ]


def get_axes_keys(ontology: MedicalOntology) -> list[str]:
    return list(ontology.clinical_axes.keys())


def axis_label(ontology: MedicalOntology, axis_id: ClinicalAxis) -> str:
    return ontology.clinical_axes[axis_id].label


def get_axis_bins(ontology: MedicalOntology, axis: ClinicalAxis) -> list[str]:
    try:
        bins = ontology.clinical_axes[axis].bins
    except KeyError as exc:
        raise ValueError(f'Ontology is missing clinical axis metadata for {axis}') from exc
    if not bins:
        raise ValueError(f'Ontology axis {axis} must declare at least one value bin')
    return bins


def _ontology_path(cfg: ExperimentCfg) -> Path:
    if cfg.generation.ontology_path:
        return Path(cfg.generation.ontology_path)
    return MedicalDatasetGenPaths.default_ontology_path
