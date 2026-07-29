from __future__ import annotations

import argparse
import hashlib
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

import polars as pl

from experiments.medical_dataset_gen.utils.exp_naming import resolve_experiment_name
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths

type RewriteKind = Literal['none', 'defer_template', 'drop_geometry']

_BASE_TABLES = frozenset({'query_plans', 'clinical_facts'})
_CHUNK_TABLES = frozenset({'chunk_documents', 'chunk_memberships', 'qrels'})
_QUERY_TABLES = frozenset({'queries', 'gold_answers'})
_MIGRATED_TABLES = _BASE_TABLES | _CHUNK_TABLES | _QUERY_TABLES
_CHUNK_SUFFIXES: tuple[tuple[str, str], ...] = (
    ('_simple_c', 'simple_c'),
    ('_hardened_c', 'hardened_c'),
)


@dataclass(frozen=True, slots=True)
class MigrationGroup:
    table: str
    target_path: Path
    source_paths: tuple[Path, ...]
    rewrite: RewriteKind
    target_exists: bool


@dataclass(frozen=True, slots=True)
class MigrationConflict:
    table: str
    target_path: Path
    source_paths: tuple[Path, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class RegenerationGroup:
    table: str
    target_path: Path
    source_paths: tuple[Path, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    groups: tuple[MigrationGroup, ...]
    regenerations: tuple[RegenerationGroup, ...]
    conflicts: tuple[MigrationConflict, ...]


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    plan = build_migration_plan(
        parents=_parent_names(args.parents),
        artifact_versions=_artifact_versions(args.artifact_versions),
        verify_content=args.verify_content,
    )
    _print_plan(plan, apply=args.apply, summary_only=args.summary_only)
    if not args.apply:
        print('[shared-layout] dry run only. Re-run with --apply after reviewing conflicts.')
        return
    if plan.conflicts:
        raise SystemExit('[shared-layout] refusing to apply while conflicts remain')

    moved, consolidated, discarded, removed_dirs = apply_migration_plan(plan)
    print(
        '[shared-layout] applied: '
        f'targets_created={moved:,}, redundant_sources_removed={consolidated:,}, '
        f'obsolete_gold_answer_sources_removed={discarded:,}, '
        f'empty_legacy_dirs_removed={removed_dirs:,}'
    )


def build_migration_plan(
    *,
    parents: set[str] | None,
    artifact_versions: set[str] | None,
    verify_content: bool = False,
) -> MigrationPlan:
    sources_by_target: dict[tuple[str, Path, RewriteKind], list[Path]] = defaultdict(list)
    for shared_root in _iter_shared_roots(
        parents=parents,
        artifact_versions=artifact_versions,
    ):
        for legacy_dir in sorted(path for path in shared_root.iterdir() if path.is_dir()):
            mode = _parse_legacy_mode_dir(legacy_dir.name)
            if mode is None:
                continue
            query_key, chunk_key = mode
            for source_path in sorted(legacy_dir.glob('*.parquet')):
                table = source_path.stem
                if table not in _MIGRATED_TABLES:
                    continue
                target_path, rewrite = _target_for(
                    shared_root=shared_root,
                    table=table,
                    query_key=query_key,
                    chunk_key=chunk_key,
                )
                sources_by_target[(table, target_path, rewrite)].append(source_path)

    groups: list[MigrationGroup] = []
    regenerations: list[RegenerationGroup] = []
    conflicts: list[MigrationConflict] = []
    for (table, target_path, rewrite), raw_sources in sorted(
        sources_by_target.items(),
        key=lambda item: str(item[0][1]),
    ):
        source_paths = tuple(sorted(set(raw_sources)))
        if table == 'gold_answers':
            regenerations.append(
                RegenerationGroup(
                    table=table,
                    target_path=target_path,
                    source_paths=source_paths,
                    reason=(
                        'legacy answers depend on chunk memberships; regenerate from migrated '
                        'clinical facts'
                    ),
                )
            )
            continue
        if verify_content:
            candidates = ((target_path,) if target_path.is_file() else ()) + source_paths
            equivalent, reason = _equivalent_candidates(candidates, rewrite=rewrite)
            if not equivalent:
                conflicts.append(
                    MigrationConflict(
                        table=table,
                        target_path=target_path,
                        source_paths=source_paths,
                        reason=reason,
                    )
                )
                continue
        groups.append(
            MigrationGroup(
                table=table,
                target_path=target_path,
                source_paths=source_paths,
                rewrite=rewrite,
                target_exists=target_path.is_file(),
            )
        )
    return MigrationPlan(
        groups=tuple(groups),
        regenerations=tuple(regenerations),
        conflicts=tuple(conflicts),
    )


def apply_migration_plan(plan: MigrationPlan) -> tuple[int, int, int, int]:
    if plan.conflicts:
        raise ValueError('cannot apply a migration plan containing conflicts')

    targets_created = 0
    redundant_sources_removed = 0
    obsolete_sources_removed = 0
    legacy_dirs: set[Path] = set()
    for group in plan.groups:
        legacy_dirs.update(source.parent for source in group.source_paths)
        if not group.target_exists:
            source = group.source_paths[0]
            group.target_path.parent.mkdir(parents=True, exist_ok=True)
            if group.rewrite == 'none':
                source.rename(group.target_path)
            else:
                _write_normalized(source, group.target_path, rewrite=group.rewrite)
                source.unlink()
            targets_created += 1

        for source in group.source_paths:
            if source.exists():
                source.unlink()
                redundant_sources_removed += 1

    for regeneration in plan.regenerations:
        legacy_dirs.update(source.parent for source in regeneration.source_paths)
        for source in regeneration.source_paths:
            if source.exists():
                source.unlink()
                obsolete_sources_removed += 1

    removed_dirs = 0
    for directory in sorted(legacy_dirs, key=lambda path: len(path.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            continue
        removed_dirs += 1
    return targets_created, redundant_sources_removed, obsolete_sources_removed, removed_dirs


def _iter_shared_roots(
    *,
    parents: set[str] | None,
    artifact_versions: set[str] | None,
) -> Iterable[Path]:
    parent_dirs = (
        [MedicalDatasetGenPaths.results_dir / name for name in sorted(parents)]
        if parents is not None
        else sorted(
            path
            for path in MedicalDatasetGenPaths.results_dir.iterdir()
            if path.is_dir() and not path.name.startswith('_')
        )
    )
    for parent_dir in parent_dirs:
        if not parent_dir.is_dir():
            continue
        for shared_root in sorted(parent_dir.glob('_shared_v*')):
            if not shared_root.is_dir():
                continue
            version = shared_root.name.removeprefix('_shared_')
            if artifact_versions is not None and version not in artifact_versions:
                continue
            yield shared_root


def _parse_legacy_mode_dir(name: str) -> tuple[str, str] | None:
    if name in {'base', 'chunks', 'queries'}:
        return None
    for suffix, chunk_key in _CHUNK_SUFFIXES:
        if name.endswith(suffix):
            query_key = name.removesuffix(suffix)
            if query_key:
                return query_key, chunk_key
    return None


def _target_for(
    *,
    shared_root: Path,
    table: str,
    query_key: str,
    chunk_key: str,
) -> tuple[Path, RewriteKind]:
    if table in _BASE_TABLES:
        rewrite: RewriteKind = 'defer_template' if table == 'query_plans' else 'none'
        return shared_root / 'base' / f'{table}.parquet', rewrite
    if table in _CHUNK_TABLES:
        return shared_root / 'chunks' / chunk_key / f'{table}.parquet', 'none'
    if table in _QUERY_TABLES:
        rewrite = 'drop_geometry' if table == 'queries' else 'none'
        return shared_root / 'queries' / query_key / f'{table}.parquet', rewrite
    raise ValueError(f'unsupported migration table: {table}')


def _equivalent_candidates(
    paths: tuple[Path, ...],
    *,
    rewrite: RewriteKind,
) -> tuple[bool, str]:
    if len(paths) < 2:
        return True, ''
    fingerprints = {_artifact_fingerprint(path, rewrite=rewrite) for path in paths}
    if len(fingerprints) == 1:
        return True, ''
    return False, 'candidate artifacts differ after dependency-safe normalization'


def _artifact_fingerprint(path: Path, *, rewrite: RewriteKind) -> str:
    if rewrite == 'none':
        digest = hashlib.sha256()
        with path.open('rb') as file:
            while block := file.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    frame = _normalized_frame(path, rewrite=rewrite)
    buffer = BytesIO()
    frame.write_ipc(buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _normalized_frame(path: Path, *, rewrite: RewriteKind) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    if rewrite == 'defer_template':
        if 'template_id' not in frame.columns:
            raise ValueError(f'query plan table lacks template_id: {path}')
        return frame.with_columns(pl.lit('deferred').alias('template_id'))
    if rewrite == 'drop_geometry':
        return frame.drop('passes_geometry_filter', strict=False)
    return frame


def _write_normalized(source: Path, target: Path, *, rewrite: RewriteKind) -> None:
    frame = _normalized_frame(source, rewrite=rewrite)
    temporary = target.with_suffix('.parquet.tmp')
    frame.write_parquet(temporary)
    os.replace(temporary, target)


def _parent_names(raw_parents: list[str] | None) -> set[str] | None:
    if not raw_parents:
        return None
    resolved: set[str] = set()
    for value in raw_parents:
        name = resolve_experiment_name(value)
        resolved.add(Path(name).parts[0])
    return resolved


def _artifact_versions(raw_versions: list[str] | None) -> set[str] | None:
    if not raw_versions:
        return None
    return {
        version if version.startswith('v') else f'v{version}'
        for version in raw_versions
    }


def _print_plan(plan: MigrationPlan, *, apply: bool, summary_only: bool) -> None:
    source_count = sum(len(group.source_paths) for group in plan.groups)
    obsolete_answer_sources = sum(len(group.source_paths) for group in plan.regenerations)
    print(
        f'[shared-layout] mode={"apply" if apply else "dry-run"}, '
        f'targets={len(plan.groups):,}, sources={source_count:,}, '
        f'deferred_answer_targets={len(plan.regenerations):,}, '
        f'obsolete_answer_sources={obsolete_answer_sources:,}, '
        f'conflicts={len(plan.conflicts):,}'
    )
    if summary_only:
        return
    for group in plan.groups:
        action = 'consolidate' if group.target_exists else 'migrate'
        print(
            f'[{action}] {group.table}: {len(group.source_paths)} source(s) '
            f'-> {group.target_path}'
        )
    for regeneration in plan.regenerations:
        print(
            f'[discard] {regeneration.table}: '
            f'remove {len(regeneration.source_paths)} obsolete source(s); '
            f'pipeline writes {regeneration.target_path} only when answers are required: '
            f'{regeneration.reason}'
        )
    for conflict in plan.conflicts:
        print(
            f'[conflict] {conflict.table}: {len(conflict.source_paths)} source(s) '
            f'-> {conflict.target_path}: {conflict.reason}'
        )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Consolidate legacy combined query/chunk shared-generation directories into '
            'dependency-aware base, chunks, and queries directories. Dry-run by default.'
        )
    )
    parser.add_argument(
        '--parent',
        dest='parents',
        action='append',
        help='Parent experiment to migrate. May be repeated. Defaults to all parents.',
    )
    parser.add_argument(
        '--artifact-version',
        dest='artifact_versions',
        action='append',
        help='Shared artifact version such as v4. May be repeated. Defaults to every version.',
    )
    parser.add_argument('--apply', action='store_true', help='Apply the migration.')
    parser.add_argument(
        '--verify-content',
        action='store_true',
        help=(
            'Hash candidate payloads after dependency-safe normalization and reject divergent '
            'sources. Disabled by default because legacy paths map deterministically.'
        ),
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Print aggregate counts without listing every target.',
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    main()
