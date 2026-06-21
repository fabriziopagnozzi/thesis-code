from __future__ import annotations

from pathlib import Path

import yaml

from experiments.medical_dataset_gen.schemas.generation_schemas import (
    AxisPairProfile,
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
    return MedicalOntology.model_validate(raw_ontology)


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


def get_axis_pair_profiles(
    ontology: MedicalOntology, left: ClinicalAxis, right: ClinicalAxis
) -> list[AxisPairProfile]:
    requested = {left, right}
    for pair in ontology.axis_pairs:
        if set(pair.axes) == requested:
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
    raise KeyError(f'missing clinical axis pair: {left}, {right}')


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
