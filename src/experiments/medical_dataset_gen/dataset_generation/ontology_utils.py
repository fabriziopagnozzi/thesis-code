from __future__ import annotations

from pathlib import Path

import yaml

from experiments.medical_dataset_gen.schemas.generation_schemas import (
    ClinicalAxis,
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
    ontology = MedicalOntology.model_validate(raw_ontology)
    return ontology


def get_selected_conditions(
    ontology: MedicalOntology, n_conditions: int
) -> list[tuple[str, ConditionOntology]]:
    items = list(ontology.conditions.items())
    if n_conditions > len(items):
        raise ValueError(f'Config asks for {n_conditions} conditions but ontology has {len(items)}')
    return items[:n_conditions]


def make_subgroup_pairs(
    ontology: MedicalOntology,
) -> list[tuple[tuple[SubgroupKey, SubgroupOntology], tuple[SubgroupKey, SubgroupOntology]]]:
    items = list(ontology.subgroups.items())
    pairs = []
    for i, left in enumerate(items):
        for right in items[i + 1 :]:
            pairs.append((left, right))
    return pairs


def other_subgroups(
    ontology: MedicalOntology, excluded_ids: set[str]
) -> list[tuple[str, SubgroupOntology]]:
    return [
        (sid, subgroup) for sid, subgroup in ontology.subgroups.items() if sid not in excluded_ids
    ]


def other_conditions(
    ontology: MedicalOntology, excluded_id: str
) -> list[tuple[str, ConditionOntology]]:
    return [
        (cid, condition) for cid, condition in ontology.conditions.items() if cid != excluded_id
    ]


def get_axes_keys(ontology: MedicalOntology) -> list[str]:
    return list(ontology.clinical_axes.keys())


def axis_label(ontology: MedicalOntology, axis_id: str) -> str:
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
