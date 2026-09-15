from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from experiments.medical_dataset_gen.reports.analysis_constants import (
    EXPERIMENT_FAMILIES,
    EXPERIMENT_FAMILY_LABELS,
    ExperimentFamilyId,
)
from experiments.medical_dataset_gen.reports.models import ExperimentRecord
from experiments.medical_dataset_gen.reports.report_io import report_io_workers
from experiments.medical_dataset_gen.suites.core import (
    LogicalSuite,
    SuiteManifest,
    SuiteManifestCell,
    load_logical_suite,
    load_logical_suite_family,
    load_suite_manifest,
    load_suite_spec,
    suite_root,
    suite_spec_root,
)
from experiments.medical_dataset_gen.suites.runtime import SuiteRuntime, suite_where_matches
from experiments.medical_dataset_gen.utils.exp_naming import (
    child_experiment_names,
    resolve_experiment_name,
)
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    load_config,
    paths_for,
)

_VERSION_DIR_RE = re.compile(r'^v[0-9]+$')


@dataclass(frozen=True)
class ReportSuiteSelection:
    logical_suite: LogicalSuite
    suite_ids: tuple[str, ...]


def discover_suite_experiments(
    results_dir: Path,
    *,
    suite_id: str,
    where: str | None,
    warnings: list[str],
    suite_base_id: str | None = None,
    suite_regex: str | None = None,
    logical_suite: LogicalSuite | None = None,
) -> list[ExperimentRecord]:
    """Load completed v5-suite cells solely from their manifest.

    No result-path regular expression or directory-name parsing participates in
    suite discovery.  This keeps the reporting population identical to the
    declared experiment design.
    """
    if logical_suite is None:
        logical_suite = load_report_logical_suite(
            results_dir=results_dir,
            suite_id=suite_id,
            suite_base_id=suite_base_id,
            suite_regex=suite_regex,
            warnings=warnings,
        ).logical_suite
    manifest = logical_suite.manifest
    runtimes = {
        root: SuiteRuntime.load(results_dir=root.parents[2], suite_id=root.name)
        for root in set(logical_suite.roots_by_cell_id.values())
    }
    selected_cells = [
        cell
        for cell in manifest.cells
        if cell.status == 'completed'
        and (where is None or not where.strip() or suite_where_matches(cell, where))
    ]
    if not selected_cells:
        return []
    worker_count = report_io_workers(len(selected_cells))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        loaded = executor.map(
            lambda cell: _load_suite_experiment_record(
                logical_suite=logical_suite,
                runtimes=runtimes,
                cell=cell,
            ),
            selected_cells,
        )
        records: list[ExperimentRecord] = []
        for record, record_warnings in loaded:
            warnings.extend(record_warnings)
            if record is not None:
                records.append(record)
    return sorted(records, key=lambda record: record.name)


def _load_suite_experiment_record(
    *,
    logical_suite: LogicalSuite,
    runtimes: Mapping[Path, SuiteRuntime],
    cell: SuiteManifestCell,
) -> tuple[ExperimentRecord | None, tuple[str, ...]]:
    root = logical_suite.roots_by_cell_id[cell.cell_id]
    local_warnings: list[str] = []
    cfg: ExperimentCfg | None = None
    config_error: str | None = None
    try:
        runtime = runtimes[root]
        cfg = runtime.load_config(cell)
        paths = runtime.paths(
            cell,
            cfg,
            prevalidated_data_root=logical_suite.data_roots_by_cell_id[cell.cell_id],
        )
    except Exception as exc:
        config_error = str(exc)
        local_warnings.append(f'{cell.cell_id}: could not load suite config ({exc})')
        # The stats path remains explicit even if a manually edited snapshot no longer parses.
        paths = MedicalDatasetGenPaths(cell.name, artifact_root=root / cell.result_root)
    stats_path = paths.table_path('evaluation_stats')
    if not stats_path.is_file():
        local_warnings.append(
            f'{cell.cell_id}: skipped because evaluation_stats.parquet is missing'
        )
        return None, tuple(local_warnings)
    return (
        ExperimentRecord(
            name=cell.name,
            experiment_dir=paths.experiment_dir,
            distribution_id=cell.distribution_id,
            run_label=cell.run_profile_id,
            is_subexperiment=True,
            cfg=cfg,
            paths=paths,
            config_error=config_error,
            family_id=cast(ExperimentFamilyId, cell.family_id),
            family_label=cell.family_label,
            origin=cell.origin,
            include_in_family_summary=cell.include_in_family_summary,
            factors={str(key): value for key, value in cell.factors.items()},
            tags=tuple(cell.tags),
            analysis_blocks=tuple(cell.analysis_blocks),
            analysis_tier=cell.analysis_tier,
            run_profile_factors=dict(cell.run_profile_factors),
        ),
        tuple(local_warnings),
    )


def load_report_logical_suite(
    *,
    results_dir: Path,
    suite_id: str,
    suite_base_id: str | None,
    suite_regex: str | None,
    warnings: list[str],
) -> ReportSuiteSelection:
    """Resolve one suite or a base suite plus all derived suite specs."""
    if suite_base_id is None:
        return ReportSuiteSelection(
            logical_suite=load_logical_suite(results_dir, suite_id),
            suite_ids=(suite_id,),
        )
    if suite_id != suite_base_id:
        raise ValueError('suite_id and suite_base_id must identify the same report base')
    derived_suite_ids = _derived_suite_ids_from_specs(
        results_dir=results_dir,
        base_suite_id=suite_base_id,
        suite_regex=suite_regex,
        warnings=warnings,
    )
    return ReportSuiteSelection(
        logical_suite=load_logical_suite_family(
            results_dir,
            base_suite_id=suite_base_id,
            derived_suite_ids=derived_suite_ids,
        ),
        suite_ids=(suite_base_id, *derived_suite_ids),
    )


def _derived_suite_ids_from_specs(
    *,
    results_dir: Path,
    base_suite_id: str,
    suite_regex: str | None,
    warnings: list[str],
) -> tuple[str, ...]:
    """Find materialized derived suites whose specs point at the base suite."""
    compiled_regex = re.compile(suite_regex) if suite_regex is not None else None
    candidates: dict[str, Path] = {}
    spec_paths = sorted((*suite_spec_root().glob('*.yaml'), *suite_spec_root().glob('*.yml')))
    for spec_path in spec_paths:
        try:
            spec = load_suite_spec(spec_path)
        except Exception as exc:
            warnings.append(f'{spec_path.name}: could not parse suite spec ({exc})')
            continue
        if spec.origin != 'derived' or spec.source is None:
            continue
        if spec.source.suite_id != base_suite_id or spec.suite_id == base_suite_id:
            continue
        if compiled_regex is not None and compiled_regex.search(spec.suite_id) is None:
            continue
        candidates[spec.suite_id] = spec_path

    materialized: list[str] = []
    for suite_id in sorted(candidates):
        manifest_path = suite_root(results_dir, suite_id) / 'suite_manifest.json'
        if not manifest_path.is_file():
            warnings.append(
                f'{suite_id}: derived suite spec found but manifest is not materialized'
            )
            continue
        # Parse once here so a malformed manifest is reported with its suite ID
        # before the family loader attempts to combine it.
        try:
            load_suite_manifest(results_dir, suite_id)
        except Exception as exc:
            warnings.append(f'{suite_id}: could not load derived suite manifest ({exc})')
            continue
        materialized.append(suite_id)
    if compiled_regex is not None and not materialized:
        warnings.append(
            f'--suite-regex {suite_regex!r} selected no materialized derived suites for '
            f'{base_suite_id}'
        )
    return tuple(materialized)


def suite_cells_matching_where(manifest: SuiteManifest, where: str | None) -> set[str]:
    """Return manifest cell IDs targeted by a suite ``--where`` expression.

    Reporting uses this separately from completed-artifact discovery: strict
    contrast validation should reject an absent member of a contrast that the
    user requested, but should not treat an intentionally out-of-scope member
    as a failed smoke cross.
    """
    if where is None or not where.strip():
        return {cell.cell_id for cell in manifest.cells}
    return {cell.cell_id for cell in manifest.cells if suite_where_matches(cell, where)}


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
        _requested_experiment_names(
            results_dir,
            requested_experiments,
            warnings=warnings,
        )
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
        if parts[0] == '00_scrapped' and not include_scrapped:
            continue
        if len(parts) > 2:
            warnings.append(f'{name}: skipped because subexperiments support only one child level')
            continue
        stats_path = _evaluation_stats_path(results_dir, name)
        if not stats_path.is_file():
            warnings.append(f'{name}: skipped because evaluation_stats.parquet is missing')
            continue
        record = load_experiment_record(
            results_dir,
            name,
            warnings=warnings,
        )
        records.append(record)
    return sorted(records, key=lambda record: record.name)


def _artifact_experiment_names(results_dir: Path) -> list[str]:
    return sorted(
        {
            _artifact_experiment_name(path, results_dir)
            for path in results_dir.glob('**/v5/evaluation_stats.parquet')
            if path.is_file()
        }
    )


def _artifact_experiment_name(path: Path, results_dir: Path) -> str:
    artifact_dir = path.parent.relative_to(results_dir)
    parts = artifact_dir.parts
    if parts and _VERSION_DIR_RE.fullmatch(parts[-1]):
        return Path(*parts[:-1]).as_posix()
    return artifact_dir.as_posix()


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
        if _evaluation_stats_path(results_dir, resolved).is_file():
            names.append(resolved)
            continue
        children = [
            child
            for child in child_experiment_names(resolved, results_dir=results_dir)
            if _evaluation_stats_path(results_dir, child).is_file()
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
        config_error = str(exc)
        warnings.append(f'{name}: config could not be loaded, using local artifacts only ({exc})')

    stats_path = _evaluation_stats_path(results_dir, name)
    paths = (
        paths_for(cfg)
        if cfg is not None
        else MedicalDatasetGenPaths(name, local_artifact_version=stats_path.parent.name)
    )
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
        experiment_dir=paths.experiment_dir,
        distribution_id=distribution_id,
        run_label=run_label,
        is_subexperiment=is_subexperiment,
        cfg=cfg,
        paths=paths,
        config_error=config_error,
        family_id=family_id,
        family_label=family_label,
    )


def _evaluation_stats_path(
    results_dir: Path,
    name: str,
) -> Path:
    return results_dir / name / 'v5' / 'evaluation_stats.parquet'


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
