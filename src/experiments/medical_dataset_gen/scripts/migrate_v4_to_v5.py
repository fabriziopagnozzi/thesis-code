"""Transactionally migrate completed legacy v4 artifacts into a v5 suite.

The migration is intentionally a file-layout/provenance operation.  It never
calls a generator, embedder, or evaluator.  The default mode is a no-write
inventory; callers must pass ``--execute`` to move any artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from experiments.medical_dataset_gen.suites.core import (
    MANIFEST_VERSION,
    SUITE_LAYOUT_VERSION,
    SuiteManifest,
    SuiteManifestCell,
    _dataset_hash,
    _sha256_json,
    suite_root,
)
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths

DEFAULT_SUITE_ID = 'thesis_v4_migrated'
MIGRATION_VERSION = 1
_AUDITED_CONFUNDED_DISTRIBUTIONS = {
    'BAL_L01_balanced_dense_high_purity',
    'BAL_S03_no_niche_dense_secondary',
    'DOM_L01_dominant_primary_skew_ultra',
    'BG_L01_single_outlier_pressure_large',
    'BG_L02_paired_back_outliers_islands_large',
    'BG_M02_background_topology_singletons',
    'BG_M03_control_outlier_pressure',
}


@dataclass(frozen=True)
class LegacyRun:
    distribution_dir: Path
    run_dir: Path

    @property
    def distribution_id(self) -> str:
        return self.distribution_dir.name

    @property
    def run_profile_id(self) -> str:
        return self.run_dir.name

    @property
    def artifact_dir(self) -> Path:
        return self.run_dir / 'v4'


@dataclass(frozen=True)
class LegacyDistribution:
    directory: Path
    runs: tuple[LegacyRun, ...]


@dataclass(frozen=True)
class MoveOperation:
    source: Path
    staged_destination: Path
    final_destination: Path
    kind: str


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    results_dir = args.results_dir.expanduser().resolve()
    suite_id = str(args.suite_id)
    target = suite_root(results_dir, suite_id)
    if args.rollback:
        rollback_migration(results_dir=results_dir, suite_id=suite_id)
        return 0
    if target.exists():
        if (target / 'suite_manifest.json').is_file():
            print(f'{target}: already migrated; no action taken')
            return 0
        raise FileExistsError(f'migration target exists without a suite manifest: {target}')

    inventory = inventory_v4_artifacts(results_dir)
    _print_inventory(inventory)
    if not args.skip_cache:
        candidates = _qwen_cache_signature_dirs(
            MedicalDatasetGenPaths.cache_dir / 'chunk_embeddings'
        )
        print(
            'Qwen 0.6B cache signatures to adopt: '
            + (', '.join(path.name for path in candidates) if candidates else 'none')
        )
        for candidate in candidates:
            _validate_embedding_cache(candidate)
    if not args.execute:
        print('dry-run complete; no files were changed')
        return 0
    execute_migration(
        results_dir=results_dir,
        suite_id=suite_id,
        inventory=inventory,
        include_cache=not args.skip_cache,
    )
    print(f'migration complete: {target}')
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Move completed v4 benchmark artifacts into the immutable v5 suite layout.'
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        '--dry-run',
        action='store_true',
        help='Inventory and validate only (the default); do not rename any artifacts.',
    )
    mode.add_argument('--execute', action='store_true', help='Perform the transactional migration.')
    mode.add_argument(
        '--rollback', action='store_true', help='Restore an unmodified migrated suite.'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=MedicalDatasetGenPaths.results_dir,
        help='Legacy results root.',
    )
    parser.add_argument('--suite-id', default=DEFAULT_SUITE_ID)
    parser.add_argument(
        '--skip-cache', action='store_true', help='Do not move the reusable chunk-embedding cache.'
    )
    return parser


def inventory_v4_artifacts(results_dir: Path) -> tuple[LegacyDistribution, ...]:
    if not results_dir.is_dir():
        raise FileNotFoundError(f'missing results directory: {results_dir}')
    distributions: list[LegacyDistribution] = []
    for parent in sorted(results_dir.iterdir()):
        if not parent.is_dir() or parent.name.startswith('_') or parent.name == 'v5':
            continue
        shared = parent / '_shared_v4'
        if not shared.is_dir():
            continue
        runs: list[LegacyRun] = []
        for child in sorted(parent.iterdir()):
            if not child.is_dir() or child.name.startswith('_'):
                continue
            artifact_dir = child / 'v4'
            if not artifact_dir.is_dir():
                continue
            _validate_completed_v4_run(artifact_dir)
            runs.append(LegacyRun(distribution_dir=parent, run_dir=child))
        if not runs:
            raise ValueError(
                f'{parent}: _shared_v4 exists but no completed v4 child artifacts found'
            )
        distributions.append(LegacyDistribution(directory=parent, runs=tuple(runs)))
    if not distributions:
        raise ValueError(f'no completed v4 distributions found under {results_dir}')
    return tuple(distributions)


def execute_migration(
    *,
    results_dir: Path,
    suite_id: str,
    inventory: Sequence[LegacyDistribution] | None = None,
    include_cache: bool = True,
) -> None:
    target = suite_root(results_dir, suite_id)
    if target.exists():
        manifest = target / 'suite_manifest.json'
        if manifest.is_file():
            print(f'{target}: already migrated; no action taken')
            return
        raise FileExistsError(f'migration target exists without a suite manifest: {target}')
    inventory = tuple(inventory or inventory_v4_artifacts(results_dir))
    target_parent = target.parent
    target_parent.mkdir(parents=True, exist_ok=True)
    _assert_same_filesystem(results_dir, target_parent)

    cache_source = MedicalDatasetGenPaths.cache_dir / 'chunk_embeddings'
    cache_target = MedicalDatasetGenPaths.cache_dir / 'v5' / 'chunk_embeddings'
    cache_sources = _qwen_cache_signature_dirs(cache_source) if include_cache else []
    if cache_sources:
        _assert_same_filesystem(cache_source, cache_target.parent)
        conflicts = [path for path in cache_sources if (cache_target / path.name).exists()]
        if conflicts:
            raise FileExistsError(f'v5 embedding cache signatures already exist: {conflicts}')

    staging = target_parent / f'.{suite_id}.staging.{os.getpid()}'
    if staging.exists():
        raise FileExistsError(f'stale migration staging directory exists: {staging}')
    cache_staging = cache_target.parent / f'.chunk_embeddings.staging.{os.getpid()}'
    operations = _move_operations(inventory, staging, target)
    operations = [
        *operations,
        *[
            MoveOperation(
                source=source,
                staged_destination=cache_staging / source.name,
                final_destination=cache_target / source.name,
                kind='embedding_cache',
            )
            for source in cache_sources
        ],
    ]

    journal = {
        'migration_version': MIGRATION_VERSION,
        'suite_id': suite_id,
        'created_at': datetime.now(UTC).isoformat(),
        'status': 'staging',
        'operations': [_operation_json(operation) for operation in operations],
    }
    staging.mkdir(parents=True)
    _write_json(staging / 'migration_journal.json', journal)
    moved: list[MoveOperation] = []
    try:
        _write_configuration_snapshots(staging, inventory)
        for operation in operations:
            operation.staged_destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(operation.source, operation.staged_destination)
            moved.append(operation)
            journal['status'] = 'moving'
            journal['completed_operations'] = len(moved)
            _write_json(staging / 'migration_journal.json', journal)

        file_manifest = _build_file_manifest(staging, include_cache=False)
        _write_json(staging / 'file_manifest.json', file_manifest)
        cache_operations = [operation for operation in moved if operation.kind == 'embedding_cache']
        if cache_operations:
            for cache_operation in cache_operations:
                _validate_embedding_cache(cache_operation.staged_destination)
            _write_json(
                staging / 'cache_manifest.json',
                _build_cache_manifest(cache_operations),
            )
        _validate_staged_layout(staging, inventory)
        manifest = _build_suite_manifest(staging, suite_id, inventory)
        _write_json(staging / 'suite_manifest.json', manifest.model_dump(mode='json'))
        journal['status'] = 'validated'
        _write_json(staging / 'migration_journal.json', journal)

        # Publish result artifacts first.  The independent cache move follows
        # immediately and any failure restores both namespaces.
        os.replace(staging, target)
        for operation in moved:
            if operation.kind == 'embedding_cache':
                operation.final_destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(operation.staged_destination, operation.final_destination)
        journal_path = target / 'migration_journal.json'
        journal['status'] = 'published'
        journal['published_at'] = datetime.now(UTC).isoformat()
        _write_json(journal_path, journal)
    except Exception:
        _restore_after_failed_migration(
            target=target,
            staging=staging,
            moved=moved,
            operations=operations,
        )
        raise


def rollback_migration(*, results_dir: Path, suite_id: str) -> None:
    target = suite_root(results_dir, suite_id)
    journal_path = target / 'migration_journal.json'
    manifest_path = target / 'file_manifest.json'
    if not journal_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f'{target}: not a reversible v4 migration')
    journal = _read_json_mapping(journal_path)
    if (
        journal.get('migration_version') != MIGRATION_VERSION
        or journal.get('status') != 'published'
    ):
        raise ValueError(f'{target}: migration journal is not in a published reversible state')
    _assert_no_new_attempts(target)
    suite_manifest = _read_json_mapping(target / 'suite_manifest.json')
    for cell in suite_manifest.get('cells', []):
        if isinstance(cell, Mapping) and cell.get('extra_evaluation_attempts'):
            raise RuntimeError('rollback refused: suite records new evaluation attempts')
    errors = _verify_file_manifest(target, _read_json_mapping(manifest_path))
    if errors:
        raise RuntimeError(
            'rollback refused because migrated artifacts changed:\n' + '\n'.join(errors)
        )
    cache_manifest_path = target / 'cache_manifest.json'
    if cache_manifest_path.is_file():
        cache_errors = _verify_cache_manifest(_read_json_mapping(cache_manifest_path))
        if cache_errors:
            raise RuntimeError(
                'rollback refused because migrated embedding cache changed:\n'
                + '\n'.join(cache_errors)
            )
    raw_operations = journal.get('operations')
    if not isinstance(raw_operations, list):
        raise ValueError(f'{journal_path}: invalid operations list')
    operations = [_operation_from_json(value) for value in raw_operations]
    for operation in reversed(operations):
        current = operation.final_destination
        if not current.exists():
            raise FileNotFoundError(f'rollback source is missing: {current}')
        if operation.source.exists():
            raise FileExistsError(f'rollback target already exists: {operation.source}')
        operation.source.parent.mkdir(parents=True, exist_ok=True)
        os.replace(current, operation.source)
    shutil.rmtree(target)
    print(f'rollback complete: restored {len(operations)} source artifact directories')


def _validate_completed_v4_run(artifact_dir: Path) -> None:
    required = ('evaluation_stats.parquet', 'evaluation_results.parquet')
    missing = [name for name in required if not (artifact_dir / name).is_file()]
    if missing:
        raise ValueError(f'{artifact_dir}: incomplete v4 evaluation, missing {missing}')


def _move_operations(
    inventory: Sequence[LegacyDistribution], staging: Path, target: Path
) -> list[MoveOperation]:
    operations: list[MoveOperation] = []
    for distribution in inventory:
        distribution_id = distribution.directory.name
        data_stage = staging / 'distributions' / distribution_id / 'data' / 'schema-v4'
        data_final = target / data_stage.relative_to(staging)
        operations.append(
            MoveOperation(
                source=distribution.directory / '_shared_v4',
                staged_destination=data_stage,
                final_destination=data_final,
                kind='shared_data',
            )
        )
        for run in distribution.runs:
            attempt_stage = (
                staging
                / 'distributions'
                / distribution_id
                / 'runs'
                / run.run_profile_id
                / 'attempts'
                / 'migrated-v4'
            )
            operations.append(
                MoveOperation(
                    source=run.artifact_dir,
                    staged_destination=attempt_stage,
                    final_destination=target / attempt_stage.relative_to(staging),
                    kind='evaluation_attempt',
                )
            )
    return operations


def _write_configuration_snapshots(staging: Path, inventory: Sequence[LegacyDistribution]) -> None:
    for distribution in inventory:
        dist_root = staging / 'distributions' / distribution.directory.name
        parent_text = (distribution.directory / '_config.yaml').read_text()
        family_text = _read_optional_text(distribution.directory / '_exp_family.yaml')
        (dist_root / 'config').mkdir(parents=True, exist_ok=True)
        (dist_root / 'config' / 'legacy_parent_config.yaml').write_text(parent_text)
        if family_text is not None:
            (dist_root / 'config' / 'legacy_family.yaml').write_text(family_text)
        _write_json(
            dist_root / 'provenance.json',
            {
                'layout_version': SUITE_LAYOUT_VERSION,
                'origin': 'migrated_v4',
                'source_distribution_path': str(distribution.directory),
                'source_parent_config_sha256': _sha256_text(parent_text),
                'source_family_metadata_sha256': _sha256_text(family_text)
                if family_text is not None
                else None,
                'effective_source_dataset_schema_version': 4,
            },
        )
        for run in distribution.runs:
            run_root = dist_root / 'runs' / run.run_profile_id
            run_root.mkdir(parents=True, exist_ok=True)
            child_text = (run.run_dir / '_subconfig.yaml').read_text()
            (run_root / 'legacy_subconfig.yaml').write_text(child_text)
            raw = _convert_legacy_config(
                _deep_merge(_read_yaml(parent_text), _read_yaml(child_text))
            )
            (run_root / 'resolved_config.yaml').write_text(yaml.safe_dump(raw, sort_keys=False))
            _write_json(
                run_root / 'provenance.json',
                {
                    'layout_version': SUITE_LAYOUT_VERSION,
                    'origin': 'migrated_v4',
                    'source_run_path': str(run.run_dir),
                    'source_artifact_path': str(run.artifact_dir),
                    'source_subconfig_sha256': _sha256_text(child_text),
                    'resolved_config_sha256': _sha256_json(raw),
                    'effective_source_dataset_schema_version': 4,
                    'converted_pool_composition': _converted_composition(raw),
                },
            )


def _build_suite_manifest(
    staging: Path,
    suite_id: str,
    inventory: Sequence[LegacyDistribution],
) -> SuiteManifest:
    cells: list[SuiteManifestCell] = []
    for distribution in inventory:
        family_id, family_label = _legacy_family(distribution.directory)
        for run in distribution.runs:
            dist_id = run.distribution_id
            run_id = run.run_profile_id
            run_root = staging / 'distributions' / dist_id / 'runs' / run_id
            raw = _read_yaml((run_root / 'resolved_config.yaml').read_text())
            cfg_sha = _sha256_json(raw)
            cell_id = f'{dist_id}__{run_id}'
            cells.append(
                SuiteManifestCell(
                    cell_id=cell_id,
                    name=f'{suite_id}/{dist_id}/{run_id}',
                    distribution_id=dist_id,
                    distribution_base_id=dist_id,
                    run_profile_id=run_id,
                    family_id=family_id,
                    family_label=family_label,
                    origin='migrated_v4',
                    dataset_schema_version=4,
                    evaluation_schema_version=4,
                    status='completed',
                    include_in_causal_summaries=(dist_id not in _AUDITED_CONFUNDED_DISTRIBUTIONS),
                    factors={
                        'legacy_distribution': dist_id,
                        'legacy_run_label': run_id,
                        'artifact_origin': 'migrated_v4',
                        'source_dataset_schema_version': 4,
                    },
                    data_root=str(
                        (Path('distributions') / dist_id / 'data' / 'schema-v4').as_posix()
                    ),
                    attempt_root=str(
                        (
                            Path('distributions')
                            / dist_id
                            / 'runs'
                            / run_id
                            / 'attempts'
                            / 'migrated-v4'
                        ).as_posix()
                    ),
                    resolved_config_path=str(
                        (
                            Path('distributions')
                            / dist_id
                            / 'runs'
                            / run_id
                            / 'resolved_config.yaml'
                        ).as_posix()
                    ),
                    config_sha256=cfg_sha,
                    dataset_sha256=_dataset_hash(raw),
                    run_profile_sha256=_run_profile_hash(raw),
                    artifact_manifest_path='file_manifest.json',
                )
            )
    return SuiteManifest(
        manifest_version=MANIFEST_VERSION,
        layout_version=SUITE_LAYOUT_VERSION,
        suite_id=suite_id,
        origin='migrated_v4',
        created_at=datetime.now(UTC).isoformat(),
        cells=cells,
        comparison_groups=[],
    )


def _convert_legacy_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Preserve legacy semantics while expressing pool topology explicitly."""
    converted = _deep_copy(raw)
    converted['dataset_schema_version'] = 4
    pools = converted.get('generation', {}).get('chunk_pools', {})
    if not isinstance(pools, dict):
        raise ValueError('legacy config lacks generation.chunk_pools')
    for key in ('dominant_primary', 'other_primary', 'secondary', 'niche'):
        pool = pools.get(key)
        if not isinstance(pool, dict):
            raise ValueError(f'legacy config lacks chunk pool {key!r}')
        size = pool.pop('size', None)
        if not isinstance(size, int) or size < 1:
            raise ValueError(f'legacy pool {key!r} has invalid size {size!r}')
        pool['num_clusters'] = 1
        pool['chunks_per_cluster'] = size
        distractors = pool.get('distractors', [])
        if not isinstance(distractors, list):
            raise ValueError(f'legacy pool {key!r} has invalid distractors')
        for distractor in distractors:
            if not isinstance(distractor, dict):
                raise ValueError(f'legacy pool {key!r} has non-mapping distractor')
            distractor_size = distractor.pop('size', None)
            if not isinstance(distractor_size, int) or distractor_size < 1:
                raise ValueError(f'legacy pool {key!r} has invalid distractor size')
            distractor['num_clusters'] = distractor_size
            distractor['chunks_per_cluster'] = 1
    backgrounds = pools.get('background_outliers', [])
    if not isinstance(backgrounds, list):
        raise ValueError('legacy background_outliers must be a list')
    for background in backgrounds:
        if not isinstance(background, dict):
            raise ValueError('legacy background outlier must be a mapping')
        per_cluster = background.pop('size', None)
        if not isinstance(per_cluster, int) or per_cluster < 1:
            raise ValueError('legacy background outlier has invalid size')
        background.setdefault('num_clusters', 1)
        background['chunks_per_cluster'] = per_cluster
    converted.setdefault('suite_metadata', {})
    # ``suite_metadata`` is not part of ExperimentCfg and is deliberately kept
    # outside the runtime snapshot to preserve a direct loader contract.
    converted.pop('suite_metadata', None)
    return converted


def _converted_composition(raw: Mapping[str, Any]) -> dict[str, Any]:
    pools = cast(Mapping[str, Any], cast(Mapping[str, Any], raw['generation'])['chunk_pools'])
    gold_pools = [
        cast(Mapping[str, Any], pools[name])
        for name in ('dominant_primary', 'other_primary', 'secondary', 'niche')
    ]
    gold_vector = [
        int(pool['num_clusters']) * int(pool['chunks_per_cluster']) for pool in gold_pools
    ]
    background = sum(
        int(spec['num_clusters']) * int(spec['chunks_per_cluster'])
        for spec in cast(list[Mapping[str, Any]], pools.get('background_outliers', []))
    )
    near_miss = sum(
        int(spec['num_clusters']) * int(spec['chunks_per_cluster'])
        for pool in gold_pools
        for spec in cast(list[Mapping[str, Any]], pool.get('distractors', []))
    )
    return {
        'gold_component_masses': gold_vector,
        'near_miss_mass': near_miss,
        'background_mass': background,
    }


def _build_file_manifest(staging: Path, *, include_cache: bool) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    excluded = {
        'suite_manifest.json',
        'file_manifest.json',
        'migration_journal.json',
        'legacy_parent_config.yaml',
        'legacy_subconfig.yaml',
        'legacy_family.yaml',
        'resolved_config.yaml',
    }
    for path in sorted(item for item in staging.rglob('*') if item.is_file()):
        if path.name in excluded:
            continue
        entry: dict[str, Any] = {
            'path': str(path.relative_to(staging).as_posix()),
            'size_bytes': path.stat().st_size,
            'sha256': _sha256_file(path),
        }
        if path.suffix == '.parquet':
            parquet = pq.ParquetFile(path)
            entry['parquet_schema_sha256'] = hashlib.sha256(
                parquet.schema_arrow.serialize().to_pybytes()
            ).hexdigest()
            entry['parquet_rows'] = parquet.metadata.num_rows
        files.append(entry)
    return {
        'layout_version': SUITE_LAYOUT_VERSION,
        'migration_version': MIGRATION_VERSION,
        'files': files,
    }


def _build_cache_manifest(operations: Sequence[MoveOperation]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for operation in operations:
        for path in sorted(
            item for item in operation.staged_destination.rglob('*') if item.is_file()
        ):
            files.append(
                {
                    'path': str(path.relative_to(operation.staged_destination).as_posix()),
                    'signature': operation.source.name,
                    'final_path': str(
                        (
                            operation.final_destination
                            / path.relative_to(operation.staged_destination)
                        ).resolve()
                    ),
                    'size_bytes': path.stat().st_size,
                    'sha256': _sha256_file(path),
                }
            )
    return {
        'layout_version': SUITE_LAYOUT_VERSION,
        'cache_namespace': 'v5',
        'cache_root': str(operations[0].final_destination.parent) if operations else None,
        'files': files,
    }


def _validate_embedding_cache(cache_root: Path) -> None:
    required = {
        'chunk_embedding_cache_key',
        'text_sha256',
        'embedding_signature',
        'embedding_signature_payload_json',
        'dimension',
        'embedding',
        'embedding_payload_sha256',
    }
    parquet_paths = sorted(cache_root.rglob('*.parquet'))
    if not parquet_paths:
        raise ValueError(f'embedding cache contains no parquet shards: {cache_root}')
    for path in parquet_paths:
        parquet = pq.ParquetFile(path)
        missing = sorted(required - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f'{path}: invalid embedding cache schema, missing {missing}')
        batches = parquet.iter_batches(
            batch_size=1,
            columns=['embedding_signature', 'embedding_signature_payload_json', 'dimension'],
        )
        try:
            row = next(batches).to_pylist()[0]
        except StopIteration:
            raise ValueError(f'{path}: empty embedding cache shard') from None
        signature = row['embedding_signature']
        if not isinstance(signature, str) or path.parent.name != signature:
            raise ValueError(f'{path}: signature does not match its cache directory')
        payload = json.loads(row['embedding_signature_payload_json'])
        if payload.get('model_name') != 'Qwen/Qwen3-Embedding-0.6B':
            raise ValueError(f'{path}: expected Qwen3-Embedding-0.6B cache payload')
        if row['dimension'] != 1024:
            raise ValueError(f'{path}: expected Qwen 0.6B dimension 1024')


def _qwen_cache_signature_dirs(cache_root: Path) -> list[Path]:
    if not cache_root.is_dir():
        return []
    candidates: list[Path] = []
    for signature_root in sorted(path for path in cache_root.iterdir() if path.is_dir()):
        model_name: str | None = None
        for shard in sorted(signature_root.glob('*.parquet')):
            try:
                batch = next(
                    pq.ParquetFile(shard).iter_batches(
                        batch_size=1,
                        columns=['embedding_signature_payload_json'],
                    )
                )
            except (OSError, StopIteration, pa.ArrowInvalid):
                continue
            row = batch.to_pylist()[0]
            try:
                payload = json.loads(row['embedding_signature_payload_json'])
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            model_name = payload.get('model_name')
            break
        if model_name == 'Qwen/Qwen3-Embedding-0.6B':
            candidates.append(signature_root)
    return candidates


def _verify_cache_manifest(manifest: Mapping[str, Any]) -> list[str]:
    files = manifest.get('files')
    if not isinstance(files, list):
        return ['cache manifest lacks files list']
    errors: list[str] = []
    for entry in files:
        if not isinstance(entry, Mapping):
            errors.append('invalid cache manifest file entry')
            continue
        path = entry.get('final_path')
        expected = entry.get('sha256')
        if not isinstance(path, str) or not isinstance(expected, str):
            errors.append('cache manifest entry lacks path/hash')
            continue
        resolved = Path(path)
        if not resolved.is_file():
            errors.append(f'missing: {resolved}')
        elif _sha256_file(resolved) != expected:
            errors.append(f'hash mismatch: {resolved}')
    return errors


def _validate_staged_layout(staging: Path, inventory: Sequence[LegacyDistribution]) -> None:
    for distribution in inventory:
        data_root = staging / 'distributions' / distribution.directory.name / 'data' / 'schema-v4'
        for relative in (
            Path('base/query_plans.parquet'),
            Path('base/clinical_facts.parquet'),
        ):
            if not (data_root / relative).is_file():
                raise FileNotFoundError(
                    f'migrated data missing required artifact: {data_root / relative}'
                )
        for run in distribution.runs:
            attempt = (
                staging
                / 'distributions'
                / run.distribution_id
                / 'runs'
                / run.run_profile_id
                / 'attempts'
                / 'migrated-v4'
            )
            _validate_completed_v4_run(attempt)


def _verify_file_manifest(root: Path, manifest: Mapping[str, Any]) -> list[str]:
    files = manifest.get('files')
    if not isinstance(files, list):
        return ['file manifest lacks files list']
    errors: list[str] = []
    for entry in files:
        if not isinstance(entry, Mapping):
            errors.append('invalid manifest file entry')
            continue
        relative = entry.get('path')
        expected = entry.get('sha256')
        if not isinstance(relative, str) or not isinstance(expected, str):
            errors.append('manifest file entry lacks path/hash')
            continue
        path = root / relative
        if not path.is_file():
            errors.append(f'missing: {relative}')
            continue
        if _sha256_file(path) != expected:
            errors.append(f'hash mismatch: {relative}')
    return errors


def _assert_no_new_attempts(target: Path) -> None:
    for attempts_root in target.glob('distributions/*/runs/*/attempts'):
        attempts = sorted(path.name for path in attempts_root.iterdir() if path.is_dir())
        if attempts != ['migrated-v4']:
            raise RuntimeError(
                f'rollback refused: {attempts_root} contains additional immutable attempts {attempts}'
            )


def _restore_after_failed_migration(
    *,
    target: Path,
    staging: Path,
    moved: Sequence[MoveOperation],
    operations: Sequence[MoveOperation],
) -> None:
    locations = [target, staging]
    for operation in reversed(moved):
        if operation.kind == 'embedding_cache':
            current = (
                operation.staged_destination
                if operation.staged_destination.exists()
                else operation.final_destination
                if operation.final_destination.exists()
                else None
            )
        else:
            relative = operation.staged_destination.relative_to(staging)
            current = next(
                (location / relative for location in locations if (location / relative).exists()),
                None,
            )
        if current is not None and not operation.source.exists():
            operation.source.parent.mkdir(parents=True, exist_ok=True)
            os.replace(current, operation.source)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    for operation in operations:
        if (
            operation.kind == 'embedding_cache'
            and operation.staged_destination.exists()
            and not operation.source.exists()
        ):
            os.replace(operation.staged_destination, operation.source)


def _legacy_family(distribution_dir: Path) -> tuple[str, str]:
    path = distribution_dir / '_exp_family.yaml'
    if not path.is_file():
        return 'unknown', 'Unknown'
    raw = _read_yaml(path.read_text())
    return str(raw.get('family_id', 'unknown')), str(raw.get('family_label', 'Unknown'))


def _run_profile_hash(raw: Mapping[str, Any]) -> str:
    return _sha256_json(
        {
            'embeddings': raw.get('embeddings'),
            'retrieval': raw.get('retrieval'),
            'evaluation': raw.get('evaluation'),
            'query_geometry': raw.get('query_geometry'),
        }
    )


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = _deep_copy(base)
    for key, value in overrides.items():
        if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(cast(Mapping[str, Any], merged[key]), value)
        else:
            merged[key] = _deep_copy(value)
    return merged


def _deep_copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _read_yaml(text: str) -> dict[str, Any]:
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ValueError('expected YAML mapping')
    return cast(dict[str, Any], raw)


def _read_optional_text(path: Path) -> str | None:
    return path.read_text() if path.is_file() else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _assert_same_filesystem(source: Path, target_parent: Path) -> None:
    probe = target_parent
    while not probe.exists():
        probe = probe.parent
    if source.stat().st_dev != probe.stat().st_dev:
        raise OSError(f'migration requires same filesystem: {source} -> {target_parent}')


def _operation_json(operation: MoveOperation) -> dict[str, str]:
    return {
        'source': str(operation.source),
        'staged_destination': str(operation.staged_destination),
        'final_destination': str(operation.final_destination),
        'kind': operation.kind,
    }


def _operation_from_json(raw: object) -> MoveOperation:
    if not isinstance(raw, Mapping):
        raise ValueError('invalid migration operation')
    values = [raw.get(key) for key in ('source', 'staged_destination', 'final_destination', 'kind')]
    if not all(isinstance(value, str) for value in values):
        raise ValueError('migration operation lacks string fields')
    source, staged, final, kind = cast(tuple[str, str, str, str], tuple(values))
    return MoveOperation(Path(source), Path(staged), Path(final), kind)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def _read_json_mapping(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f'expected JSON mapping: {path}')
    return cast(dict[str, Any], raw)


def _print_inventory(inventory: Sequence[LegacyDistribution]) -> None:
    runs = sum(len(distribution.runs) for distribution in inventory)
    print(f'v4 inventory: {len(inventory)} distributions, {runs} completed cells')
    for distribution in inventory:
        print(f'  {distribution.directory.name}: {len(distribution.runs)} cells')


if __name__ == '__main__':
    raise SystemExit(main())
