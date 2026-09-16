"""Suite manifest loading, source pinning, and report-facing composition."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from experiments.medical_dataset_gen.suites.contracts import (
    DerivedSuiteSource,
    SuiteManifest,
    SuiteManifestCell,
    SuiteSpec,
    canonical_manifest,
)
from experiments.medical_dataset_gen.suites.io import (
    json_text,
    read_yaml_mapping,
    safe_relative,
    sha256_file,
    write_text_atomically,
)
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths


@dataclass(frozen=True)
class LogicalSuite:
    manifest: SuiteManifest
    roots_by_cell_id: dict[str, Path]
    data_roots_by_cell_id: dict[str, Path]


def suite_spec_root() -> Path:
    return MedicalDatasetGenPaths.root / 'experiment_specs'


def suite_root(results_dir: Path, suite_id: str) -> Path:
    from experiments.medical_dataset_gen.suites.contracts import validate_identifier

    validate_identifier(suite_id, 'suite_id')
    return results_dir / 'v5' / 'suites' / suite_id


def load_suite_spec(suite_id_or_path: str | Path) -> SuiteSpec:
    path = Path(suite_id_or_path)
    if not path.suffix:
        path = suite_spec_root() / f'{path}.yaml'
    return SuiteSpec.model_validate(read_yaml_mapping(path))


def load_suite_manifest(results_dir: Path, suite_id: str) -> SuiteManifest:
    path = suite_root(results_dir, suite_id) / 'suite_manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'missing suite manifest: {path}')
    manifest = canonical_manifest(json.loads(path.read_text()))
    if manifest.suite_id != suite_id:
        raise ValueError(
            f'manifest suite_id {manifest.suite_id!r} does not match directory {suite_id!r}'
        )
    return manifest


def write_suite_manifest(root: Path, manifest: SuiteManifest) -> None:
    write_text_atomically(
        root / 'suite_manifest.json',
        json_text(manifest.model_dump(mode='json')),
    )


def load_pinned_source_manifest(
    *, results_dir: Path, source: DerivedSuiteSource
) -> tuple[Path, SuiteManifest]:
    root = suite_root(results_dir, source.suite_id)
    path = root / 'suite_manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'missing derived-suite source manifest: {path}')
    observed = sha256_file(path)
    if observed != source.manifest_sha256:
        raise ValueError(
            f'{source.suite_id}: source manifest hash changed; expected '
            f'{source.manifest_sha256}, observed {observed}'
        )
    manifest = canonical_manifest(json.loads(path.read_text()))
    if manifest.origin != 'native':
        raise ValueError(f'{source.suite_id}: derived suites require a native source suite')
    return root, manifest


def validated_derived_source_cell(
    *,
    cell: SuiteManifestCell,
    source: DerivedSuiteSource,
    source_cells: Mapping[str, SuiteManifestCell],
) -> SuiteManifestCell:
    if (
        cell.source_suite_id != source.suite_id
        or cell.source_manifest_sha256 != source.manifest_sha256
    ):
        raise ValueError(f'{cell.cell_id}: source reference disagrees with derived suite contract')
    source_cell = source_cells.get(cast(str, cell.source_cell_id))
    if source_cell is None:
        raise ValueError(f'{cell.cell_id}: source cell {cell.source_cell_id!r} is missing')
    if source_cell.status != 'completed':
        raise ValueError(f'{cell.cell_id}: source cell {source_cell.cell_id} is not completed')
    if source_cell.distribution_id != cell.distribution_id:
        raise ValueError(f'{cell.cell_id}: source distribution does not match derived distribution')
    if source_cell.dataset_sha256 != cell.source_dataset_sha256:
        raise ValueError(f'{cell.cell_id}: source dataset hash changed')
    if cell.dataset_sha256 != cell.source_dataset_sha256:
        raise ValueError(f'{cell.cell_id}: derived dataset hash differs from its source')
    return source_cell


def resolve_derived_source_cell(
    *, root: Path, cell: SuiteManifestCell
) -> tuple[Path, SuiteManifestCell]:
    if cell.origin != 'derived':
        raise ValueError(f'{cell.cell_id}: cell is not source-backed')
    manifest = load_suite_manifest(root.parents[2], root.name)
    if manifest.source is None:
        raise ValueError(f'{cell.cell_id}: derived suite manifest has no source contract')
    source_root, source_manifest = load_pinned_source_manifest(
        results_dir=root.parents[2], source=manifest.source
    )
    return source_root, validated_derived_source_cell(
        cell=cell,
        source=manifest.source,
        source_cells={candidate.cell_id: candidate for candidate in source_manifest.cells},
    )


def load_logical_suite(results_dir: Path, suite_id: str) -> LogicalSuite:
    root = suite_root(results_dir, suite_id)
    manifest = load_suite_manifest(results_dir, suite_id)
    if manifest.origin == 'native':
        return LogicalSuite(
            manifest=manifest,
            roots_by_cell_id={cell.cell_id: root for cell in manifest.cells},
            data_roots_by_cell_id={
                cell.cell_id: safe_relative(root, cell.data_root) for cell in manifest.cells
            },
        )
    assert manifest.source is not None
    source_root, source_manifest = load_pinned_source_manifest(
        results_dir=results_dir, source=manifest.source
    )
    selected = set(manifest.source.distribution_ids)
    source_cells = [cell for cell in source_manifest.cells if cell.distribution_id in selected]
    incomplete = [cell.cell_id for cell in source_cells if cell.status != 'completed']
    if incomplete:
        raise ValueError(f'{suite_id}: incomplete source cells: {incomplete[:8]}')
    source_by_id = {cell.cell_id: cell for cell in source_manifest.cells}
    resolved = {
        cell.cell_id: validated_derived_source_cell(
            cell=cell, source=manifest.source, source_cells=source_by_id
        )
        for cell in manifest.cells
    }
    all_cells = source_cells + list(manifest.cells)
    ids = [cell.cell_id for cell in all_cells]
    if len(ids) != len(set(ids)):
        raise ValueError(f'{suite_id}: source and derived cells must have distinct IDs')
    logical = manifest.model_copy(
        update={
            'distributions': source_manifest.distributions,
            'run_profiles': source_manifest.run_profiles + manifest.run_profiles,
            'cells': all_cells,
            'comparison_groups': source_manifest.comparison_groups,
        }
    )
    return LogicalSuite(
        manifest=logical,
        roots_by_cell_id={
            **{cell.cell_id: source_root for cell in source_cells},
            **{cell.cell_id: root for cell in manifest.cells},
        },
        data_roots_by_cell_id={
            **{cell.cell_id: safe_relative(source_root, cell.data_root) for cell in source_cells},
            **{
                cell.cell_id: safe_relative(source_root, resolved[cell.cell_id].data_root)
                for cell in manifest.cells
            },
        },
    )


def load_logical_suite_family(
    results_dir: Path,
    *,
    base_suite_id: str,
    derived_suite_ids: Sequence[str],
) -> LogicalSuite:
    base_root = suite_root(results_dir, base_suite_id)
    base = load_suite_manifest(results_dir, base_suite_id)
    if base.origin != 'native':
        raise ValueError(f'{base_suite_id}: suite-base must identify a native suite')
    base_hash = sha256_file(base_root / 'suite_manifest.json')
    cells = list(base.cells)
    profiles = list(base.run_profiles)
    roots = {cell.cell_id: base_root for cell in base.cells}
    data_roots = {cell.cell_id: safe_relative(base_root, cell.data_root) for cell in base.cells}
    base_cells = {cell.cell_id: cell for cell in base.cells}
    cell_ids = set(base_cells)
    profile_ids = {profile.run_profile_id for profile in profiles}
    seen: set[str] = set()
    for suite_id in derived_suite_ids:
        if suite_id == base_suite_id or suite_id in seen:
            continue
        seen.add(suite_id)
        derived = load_suite_manifest(results_dir, suite_id)
        if derived.origin != 'derived' or derived.source is None:
            raise ValueError(f'{suite_id}: expected a derived suite')
        if derived.source.suite_id != base_suite_id or derived.source.manifest_sha256 != base_hash:
            raise ValueError(f'{suite_id}: source contract does not match {base_suite_id}')
        derived_root = suite_root(results_dir, suite_id)
        local_cell_ids = {cell.cell_id for cell in derived.cells}
        local_profile_ids = {profile.run_profile_id for profile in derived.run_profiles}
        if cell_ids & local_cell_ids:
            raise ValueError(f'{suite_id}: derived cell IDs collide with the base suite')
        if profile_ids & local_profile_ids:
            raise ValueError(f'{suite_id}: derived run-profile IDs collide with the base suite')
        resolved = {
            cell.cell_id: validated_derived_source_cell(
                cell=cell, source=derived.source, source_cells=base_cells
            )
            for cell in derived.cells
        }
        cells.extend(derived.cells)
        profiles.extend(derived.run_profiles)
        roots.update({cell.cell_id: derived_root for cell in derived.cells})
        data_roots.update(
            {
                cell.cell_id: safe_relative(base_root, resolved[cell.cell_id].data_root)
                for cell in derived.cells
            }
        )
        cell_ids.update(local_cell_ids)
        profile_ids.update(local_profile_ids)
    return LogicalSuite(
        manifest=base.model_copy(update={'cells': cells, 'run_profiles': profiles}),
        roots_by_cell_id=roots,
        data_roots_by_cell_id=data_roots,
    )
