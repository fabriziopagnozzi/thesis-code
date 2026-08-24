"""Deduplicate suite embedding matrices into model-qualified shared artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from experiments.medical_dataset_gen.suites.core import load_suite_manifest, suite_root
from experiments.medical_dataset_gen.suites.runtime import (
    load_cell_config,
    suite_paths_for_cell,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    EMBEDDING_ARTIFACT_FILENAMES,
    MedicalDatasetGenPaths,
)

type EmbeddingPairKind = Literal['chunks', 'queries']
type PairArtifactName = Literal[
    'chunk_vectors', 'chunk_ids', 'query_vectors', 'query_ids'
]

_ARTIFACT_PAIRS: dict[
    EmbeddingPairKind, tuple[PairArtifactName, PairArtifactName]
] = {
    'chunks': ('chunk_vectors', 'chunk_ids'),
    'queries': ('query_vectors', 'query_ids'),
}
_HASH_BLOCK_BYTES = 16 * 1024 * 1024
_VECTOR_COMPARE_BLOCK_BYTES = 64 * 1024 * 1024
_MAX_VECTOR_ABS_DIFF = 1e-6


@dataclass(frozen=True)
class LegacyAttempt:
    cell_id: str
    attempt_root: Path


@dataclass(frozen=True)
class MigrationGroupReport:
    artifact_kind: EmbeddingPairKind
    shared_root: str
    source_attempt: str | None
    matching_attempts: int
    removed_files: int
    vector_sha256: str | None
    ids_sha256: str | None
    non_identical_vector_attempts: int
    max_vector_abs_diff: float


def migrate_suite_chunk_embeddings(
    *, results_dir: Path, suite_id: str, execute: bool, verify: bool = True
) -> list[MigrationGroupReport]:
    """Move one pair per resolved surface and delete redundant local pairs."""
    root = suite_root(results_dir, suite_id)
    manifest = load_suite_manifest(results_dir, suite_id)
    root_resolved = root.resolve()
    grouped_attempts: dict[tuple[EmbeddingPairKind, Path], list[LegacyAttempt]] = defaultdict(list)

    for cell in manifest.cells:
        cfg = load_cell_config(root, cell)
        paths = suite_paths_for_cell(root=root, cell=cell, cfg=cfg)
        base_attempt = _safe_suite_path(root, cell.attempt_root)
        attempts = [base_attempt]
        attempts.extend(base_attempt.parent / attempt_id for attempt_id in cell.extra_evaluation_attempts)
        for artifact_kind, (vector_artifact, _) in _ARTIFACT_PAIRS.items():
            shared_vector_path = paths.embeddings_paths(vector_artifact)
            if not shared_vector_path.resolve().is_relative_to(root_resolved):
                raise ValueError(
                    f'shared {artifact_kind} path escapes suite root: {shared_vector_path}'
                )
            grouped_attempts[(artifact_kind, shared_vector_path.parent)].extend(
                LegacyAttempt(cell_id=cell.cell_id, attempt_root=attempt_root)
                for attempt_root in attempts
            )

    reports: list[MigrationGroupReport] = []
    sorted_groups = sorted(
        grouped_attempts.items(), key=lambda item: (item[0][0], str(item[0][1]))
    )
    for group_index, ((artifact_kind, shared_root), attempts) in enumerate(
        sorted_groups, start=1
    ):
        artifacts = _ARTIFACT_PAIRS[artifact_kind]
        vector_artifact, ids_artifact = artifacts
        shared_paths: dict[PairArtifactName, Path] = {
            artifact: shared_root / EMBEDDING_ARTIFACT_FILENAMES[artifact]
            for artifact in artifacts
        }
        existing_shared = [path.exists() for path in shared_paths.values()]
        if any(existing_shared) and not all(existing_shared):
            raise RuntimeError(f'incomplete shared chunk artifact pair: {shared_root}')

        local_pairs: list[LegacyAttempt] = []
        for attempt in attempts:
            local_paths = _legacy_pair_paths(attempt.attempt_root, artifacts)
            existence = [path.exists() for path in local_paths.values()]
            if any(existence) and not all(existence):
                raise RuntimeError(f'incomplete local chunk artifact pair: {attempt.attempt_root}')
            if all(existence):
                local_pairs.append(attempt)

        if not local_pairs and not all(existing_shared):
            continue

        action = 'validating' if verify else 'processing'
        print(
            f'[migrate] {action} group {group_index}/{len(sorted_groups)}: '
            f'{artifact_kind} {shared_root.relative_to(root)} '
            f'({len(local_pairs)} local pair(s))',
            flush=True,
        )
        candidates: list[tuple[str, dict[PairArtifactName, Path]]] = (
            [('shared', shared_paths)] if all(existing_shared) else []
        )
        candidates.extend(
            (
                f'{attempt.cell_id}:{attempt.attempt_root.name}',
                _legacy_pair_paths(attempt.attempt_root, artifacts),
            )
            for attempt in local_pairs
        )
        representative_label, representative_paths = candidates[0]
        representative_ids_sha256: str | None = None
        representative_vector_sha256: str | None = None
        non_identical_vector_attempts = 0
        max_vector_abs_diff = 0.0
        if verify:
            ids_digests = {
                label: _sha256_file(paths[ids_artifact]) for label, paths in candidates
            }
            if len(set(ids_digests.values())) != 1:
                labels = ', '.join(sorted(ids_digests))
                raise RuntimeError(
                    f'{shared_root}: non-identical {artifact_kind} IDs across {labels}'
                )
            representative_ids_sha256 = ids_digests[representative_label]
            for label, paths in candidates[1:]:
                exact, observed_max_abs_diff = _compare_vector_arrays(
                    representative_paths[vector_artifact], paths[vector_artifact]
                )
                if observed_max_abs_diff > _MAX_VECTOR_ABS_DIFF:
                    raise RuntimeError(
                        f'{shared_root}: vector difference exceeds {_MAX_VECTOR_ABS_DIFF:g} '
                        f'between {representative_label} and {label} '
                        f'(max_abs_diff={observed_max_abs_diff:g})'
                    )
                if not exact:
                    non_identical_vector_attempts += 1
                max_vector_abs_diff = max(max_vector_abs_diff, observed_max_abs_diff)
            representative_vector_sha256 = _sha256_file(
                representative_paths[vector_artifact]
            )

        source_attempt = local_pairs[0] if local_pairs and not all(existing_shared) else None
        removed_files = 0
        if execute:
            shared_root.mkdir(parents=True, exist_ok=True)
            if source_attempt is not None:
                source_paths = _legacy_pair_paths(source_attempt.attempt_root, artifacts)
                for artifact in artifacts:
                    os.replace(source_paths[artifact], shared_paths[artifact])
            for attempt in local_pairs:
                if source_attempt is not None and attempt == source_attempt:
                    _update_embedding_metadata(attempt.attempt_root, shared_paths)
                    continue
                for path in _legacy_pair_paths(attempt.attempt_root, artifacts).values():
                    path.unlink()
                    removed_files += 1
                _update_embedding_metadata(attempt.attempt_root, shared_paths)

        reports.append(
            MigrationGroupReport(
                artifact_kind=artifact_kind,
                shared_root=str(shared_root.relative_to(root)),
                source_attempt=(
                    str(source_attempt.attempt_root.relative_to(root))
                    if source_attempt is not None
                    else None
                ),
                matching_attempts=len(local_pairs),
                removed_files=removed_files,
                vector_sha256=representative_vector_sha256,
                ids_sha256=representative_ids_sha256,
                non_identical_vector_attempts=non_identical_vector_attempts,
                max_vector_abs_diff=max_vector_abs_diff,
            )
        )

    if execute:
        report_path = root / 'shared_embeddings_migration.json'
        _write_json_atomic(
            report_path,
            {
                'suite_id': suite_id,
                'migration_version': 1,
                'groups': [asdict(report) for report in reports],
            },
        )
    return reports


def _legacy_pair_paths(
    attempt_root: Path,
    artifacts: tuple[PairArtifactName, PairArtifactName],
) -> dict[PairArtifactName, Path]:
    return {
        artifact: attempt_root / EMBEDDING_ARTIFACT_FILENAMES[artifact]
        for artifact in artifacts
    }


def _safe_suite_path(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'artifact path escapes suite root: {relative_path}')
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(_HASH_BLOCK_BYTES), b''):
            digest.update(block)
    return digest.hexdigest()


def _compare_vector_arrays(left_path: Path, right_path: Path) -> tuple[bool, float]:
    left = np.load(left_path, mmap_mode='r')
    right = np.load(right_path, mmap_mode='r')
    if left.dtype != right.dtype or left.shape != right.shape or left.ndim != 2:
        raise RuntimeError(
            f'incompatible embedding vector arrays: {left_path} {left.shape}/{left.dtype} versus '
            f'{right_path} {right.shape}/{right.dtype}'
        )
    if not np.issubdtype(left.dtype, np.floating):
        raise RuntimeError(f'embedding vectors must be floating point: {left_path}')
    bytes_per_row = max(int(left.shape[1]) * int(left.dtype.itemsize), 1)
    rows_per_block = max(_VECTOR_COMPARE_BLOCK_BYTES // bytes_per_row, 1)
    exact = True
    max_abs_diff = 0.0
    for start in range(0, int(left.shape[0]), rows_per_block):
        left_block = np.asarray(left[start : start + rows_per_block])
        right_block = np.asarray(right[start : start + rows_per_block])
        if not np.isfinite(left_block).all() or not np.isfinite(right_block).all():
            raise RuntimeError(f'non-finite embedding vector encountered: {left_path}, {right_path}')
        if np.array_equal(left_block, right_block):
            continue
        exact = False
        max_abs_diff = max(
            max_abs_diff,
            float(np.max(np.abs(left_block - right_block))),
        )
    return exact, max_abs_diff


def _update_embedding_metadata(
    attempt_root: Path, shared_paths: dict[PairArtifactName, Path]
) -> None:
    metadata_path = attempt_root / EMBEDDING_ARTIFACT_FILENAMES['metadata']
    if not metadata_path.is_file():
        return
    raw = json.loads(metadata_path.read_text())
    if not isinstance(raw, dict):
        raise RuntimeError(f'embedding metadata is not an object: {metadata_path}')
    metadata = cast(dict[str, object], raw)
    for artifact, path in shared_paths.items():
        metadata[f'{artifact}_file'] = str(path)
    _write_json_atomic(metadata_path, metadata)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Move repeated per-run embedding matrices into suite-level shared artifacts.'
    )
    parser.add_argument('--suite', required=True)
    parser.add_argument(
        '--results-dir', type=Path, default=MedicalDatasetGenPaths.results_dir
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Apply the validated moves/deletions; the default is an exact-hash dry run.',
    )
    parser.add_argument(
        '--trust-resolved-surfaces',
        action='store_true',
        help='Skip content scans and trust the suite surface identity keys.',
    )
    args = parser.parse_args(argv)
    reports = migrate_suite_chunk_embeddings(
        results_dir=args.results_dir.expanduser().resolve(),
        suite_id=str(args.suite),
        execute=bool(args.execute),
        verify=not bool(args.trust_resolved_surfaces),
    )
    mode = 'migrated' if args.execute else 'validated'
    print(
        f'{mode} shared embeddings: groups={len(reports)}, '
        f'attempts={sum(report.matching_attempts for report in reports)}, '
        f'removed_files={sum(report.removed_files for report in reports)}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
