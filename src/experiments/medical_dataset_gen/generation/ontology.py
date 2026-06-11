"""Load and query the benchmark ontology used to build synthetic medical data."""

from pathlib import Path
from typing import Any

import yaml

from experiments.medical_dataset_gen.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
)


def load_ontology(cfg: ExperimentCfg) -> dict[str, Any]:
    path = _ontology_path(cfg)
    with open(path) as f:
        ontology = yaml.safe_load(f)

    required = {'conditions', 'subgroups', 'clinical_axes'}
    missing = required - set(ontology)
    if missing:
        raise ValueError(f'Ontology missing required keys: {sorted(missing)}')
    return ontology


def selected_conditions(ontology: dict[str, Any], n_conditions: int) -> list[tuple[str, dict]]:
    items = list(ontology['conditions'].items())
    if n_conditions > len(items):
        raise ValueError(f'Config asks for {n_conditions} conditions but ontology has {len(items)}')
    return items[:n_conditions]


def subgroup_pairs(ontology: dict[str, Any]) -> list[tuple[tuple[str, dict], tuple[str, dict]]]:
    items = list(ontology['subgroups'].items())
    pairs = []
    for i, left in enumerate(items):
        for right in items[i + 1 :]:
            pairs.append((left, right))
    return pairs


def other_subgroups(
    ontology: dict[str, Any],
    excluded_ids: set[str],
) -> list[tuple[str, dict]]:
    return [
        (sid, subgroup)
        for sid, subgroup in ontology['subgroups'].items()
        if sid not in excluded_ids
    ]


def other_conditions(
    ontology: dict[str, Any],
    excluded_id: str,
) -> list[tuple[str, dict]]:
    return [
        (cid, condition) for cid, condition in ontology['conditions'].items() if cid != excluded_id
    ]


def axis_ids(ontology: dict[str, Any]) -> list[str]:
    return list(ontology['clinical_axes'].keys())


def axis_label(ontology: dict[str, Any], axis_id: str) -> str:
    return ontology['clinical_axes'][axis_id]['label']


def _ontology_path(cfg: ExperimentCfg) -> Path:
    if cfg.generation.ontology_path:
        return Path(cfg.generation.ontology_path)
    return MedicalDatasetGenPaths.default_ontology_path
