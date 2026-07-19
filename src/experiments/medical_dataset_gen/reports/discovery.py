from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import yaml

from experiments.medical_dataset_gen.reports.analysis_constants import (
    EXPERIMENT_FAMILIES,
    EXPERIMENT_FAMILY_LABELS,
    ExperimentFamilyId,
)
from experiments.medical_dataset_gen.reports.models import ExperimentRecord
from experiments.medical_dataset_gen.schemas.global_config_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.exp_naming import (
    child_experiment_names,
    resolve_experiment_name,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    load_config,
    load_raw_experiment_config,
    paths_for,
)


def discover_experiments(
    results_dir: Path,
    *,
    include_scrapped: bool,
    requested_experiments: Sequence[str],
    experiment_regex: str | None = None,
    exclude_experiment_regex: str | None = None,
    warnings: list[str],
) -> list[ExperimentRecord]:
    candidate_names = (
        _requested_experiment_names(results_dir, requested_experiments, warnings)
        if requested_experiments
        else _artifact_experiment_names(results_dir)
    )
    if experiment_regex is not None:
        pattern = re.compile(experiment_regex)
        candidate_names = [name for name in candidate_names if pattern.search(name)]
    if exclude_experiment_regex is not None:
        exclude_pattern = re.compile(exclude_experiment_regex)
        candidate_names = [name for name in candidate_names if not exclude_pattern.search(name)]
    records: list[ExperimentRecord] = []
    seen: set[str] = set()
    for name in candidate_names:
        if name in seen:
            continue
        seen.add(name)
        parts = Path(name).parts
        if not parts:
            continue
        if parts[0].startswith('_'):
            continue
        if _is_legacy_all_query_experiment(parts):
            warnings.append(f'{name}: skipped legacy _allq query-scope variant')
            continue
        if parts[0] == '00_scrapped' and not include_scrapped:
            continue
        if len(parts) > 2:
            warnings.append(f'{name}: skipped because subexperiments support only one child level')
            continue
        stats_path = results_dir / name / 'evaluation_stats.parquet'
        if not stats_path.is_file():
            warnings.append(f'{name}: skipped because evaluation_stats.parquet is missing')
            continue
        record = load_experiment_record(results_dir, name, warnings=warnings)
        records.append(record)
    return sorted(records, key=lambda record: record.name)


def _is_legacy_all_query_experiment(parts: Sequence[str]) -> bool:
    return len(parts) == 2 and parts[1].endswith('_allq')


def _artifact_experiment_names(results_dir: Path) -> list[str]:
    return sorted(
        path.parent.relative_to(results_dir).as_posix()
        for path in results_dir.glob('**/evaluation_stats.parquet')
        if path.is_file()
    )


def _requested_experiment_names(
    results_dir: Path,
    requested_experiments: Sequence[str],
    warnings: list[str],
) -> list[str]:
    names: list[str] = []
    for raw_name in requested_experiments:
        try:
            resolved = resolve_experiment_name(raw_name, results_dir=results_dir)
        except Exception as exc:
            warnings.append(f'{raw_name}: could not resolve requested experiment ({exc})')
            continue
        if (results_dir / resolved / 'evaluation_stats.parquet').is_file():
            names.append(resolved)
            continue
        children = [
            child
            for child in child_experiment_names(resolved, results_dir=results_dir)
            if (results_dir / child / 'evaluation_stats.parquet').is_file()
        ]
        if children:
            names.extend(children)
        else:
            warnings.append(f'{resolved}: requested experiment has no completed eval artifact')
    return names


def load_experiment_record(
    results_dir: Path,
    name: str,
    *,
    warnings: list[str],
) -> ExperimentRecord:
    cfg: ExperimentCfg | None = None
    config_error: str | None = None
    try:
        cfg = load_config(name)
    except Exception as exc:
        try:
            cfg = _load_config_with_report_compatibility(name)
        except Exception:
            config_error = str(exc)
            warnings.append(
                f'{name}: config could not be loaded, using local artifacts only ({exc})'
            )

    paths = paths_for(cfg) if cfg is not None else MedicalDatasetGenPaths(name)
    parts = Path(name).parts
    is_subexperiment = len(parts) == 2
    distribution_id = parts[0]
    run_label = parts[1] if is_subexperiment else 'parent'
    family_id, family_label = _load_experiment_family(
        results_dir=results_dir,
        name=name,
        distribution_id=distribution_id,
        warnings=warnings,
    )
    return ExperimentRecord(
        name=name,
        experiment_dir=results_dir / name,
        distribution_id=distribution_id,
        run_label=run_label,
        is_subexperiment=is_subexperiment,
        cfg=cfg,
        paths=paths,
        config_error=config_error,
        family_id=family_id,
        family_label=family_label,
    )


def _load_experiment_family(
    *,
    results_dir: Path,
    name: str,
    distribution_id: str,
    warnings: list[str],
) -> tuple[ExperimentFamilyId, str]:
    candidate_paths = [
        results_dir / name / '_exp_family.yaml',
        results_dir / distribution_id / '_exp_family.yaml',
    ]
    for path in candidate_paths:
        if not path.is_file():
            continue
        try:
            raw: object = yaml.safe_load(path.read_text()) or {}
        except Exception as exc:
            warnings.append(f'{name}: could not read experiment family metadata at {path} ({exc})')
            return 'unknown', EXPERIMENT_FAMILY_LABELS['unknown']
        if not isinstance(raw, Mapping):
            warnings.append(f'{name}: experiment family metadata at {path} is not a mapping')
            return 'unknown', EXPERIMENT_FAMILY_LABELS['unknown']
        family_id = raw.get('family_id')
        if not isinstance(family_id, str) or family_id not in EXPERIMENT_FAMILIES:
            warnings.append(
                f'{name}: experiment family metadata at {path} has invalid family_id {family_id!r}'
            )
            return 'unknown', EXPERIMENT_FAMILY_LABELS['unknown']
        typed_family_id = cast(ExperimentFamilyId, family_id)
        family_label = raw.get('family_label')
        return (
            typed_family_id,
            family_label
            if isinstance(family_label, str)
            else EXPERIMENT_FAMILY_LABELS[typed_family_id],
        )
    return 'unknown', EXPERIMENT_FAMILY_LABELS['unknown']


def _load_config_with_report_compatibility(exp_name: str) -> ExperimentCfg:
    paths = MedicalDatasetGenPaths(exp_name)
    raw = load_raw_experiment_config(paths)
    evaluation = raw.get('evaluation')
    if isinstance(evaluation, Mapping):
        sanitized_evaluation = dict(evaluation)
        sanitized_evaluation.pop('fac_loc_mmr_comparison_kernels', None)
        raw = {**raw, 'evaluation': sanitized_evaluation}
    cfg = ExperimentCfg.model_validate(raw)
    cfg.global_.output_experiment = exp_name
    return cfg
