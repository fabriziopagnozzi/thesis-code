from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from experiments.medical_dataset_gen.utils.exp_naming import resolve_experiment_name
from experiments.medical_dataset_gen.utils.global_utils import (
    SHARED_GENERATION_TABLES,
    MedicalDatasetGenPaths,
    SharedGenerationTableName,
    SyntheticMedicalDatasetTableName,
    load_config,
    shared_generation_dir_for_config,
)


@dataclass(frozen=True, slots=True)
class LocalSharedArtifact:
    exp_name: str
    source_path: Path
    target_path: Path
    table: SharedGenerationTableName
    size_bytes: int


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    moves: tuple[LocalSharedArtifact, ...]
    duplicate_deletes: tuple[LocalSharedArtifact, ...]
    skipped_disabled: int


def main() -> None:
    args = _parse_args()
    tables = _selected_tables(args.tables)
    parents = _parent_names(args.parents)
    plan = build_migration_plan(
        parents=parents,
        tables=tables,
        delete_local_duplicates=bool(args.delete_local_duplicates),
    )
    _print_plan(plan, apply=bool(args.apply), summary_only=bool(args.summary_only))

    if not args.apply:
        print('[shared-gen] dry run only. Re-run with --apply to move files.')
        return

    moved, deleted, moved_bytes, deleted_bytes = apply_migration_plan(plan)
    print(
        '[shared-gen] applied: '
        f'moved_files={moved:,}, moved={_format_bytes(moved_bytes)}, '
        f'deleted_duplicate_files={deleted:,}, freed={_format_bytes(deleted_bytes)}'
    )


def build_migration_plan(
    *,
    parents: set[str] | None,
    tables: tuple[SharedGenerationTableName, ...],
    delete_local_duplicates: bool,
) -> MigrationPlan:
    grouped: dict[tuple[Path, SharedGenerationTableName], list[LocalSharedArtifact]] = defaultdict(
        list
    )
    skipped_disabled = 0

    for exp_name in _iter_child_experiment_names(parents=parents):
        cfg = load_config(exp_name)
        shared_dir = shared_generation_dir_for_config(cfg)
        if shared_dir is None:
            skipped_disabled += 1
            continue
        local_paths = MedicalDatasetGenPaths(exp_name)
        for table in tables:
            table_name = cast(SyntheticMedicalDatasetTableName, table)
            source_path = local_paths.local_table_path(table_name)
            if not source_path.is_file():
                continue
            target_path = shared_dir / f'{table}.parquet'
            grouped[(target_path, table)].append(
                LocalSharedArtifact(
                    exp_name=exp_name,
                    source_path=source_path,
                    target_path=target_path,
                    table=table,
                    size_bytes=source_path.stat().st_size,
                )
            )

    moves: list[LocalSharedArtifact] = []
    duplicate_deletes: list[LocalSharedArtifact] = []
    for (target_path, _table), artifacts in sorted(grouped.items(), key=lambda item: str(item[0])):
        ordered = sorted(artifacts, key=lambda item: (item.exp_name, str(item.source_path)))
        if target_path.exists():
            if delete_local_duplicates:
                duplicate_deletes.extend(ordered)
            continue

        selected = ordered[0]
        moves.append(selected)
        if delete_local_duplicates:
            duplicate_deletes.extend(artifact for artifact in ordered[1:])

    return MigrationPlan(
        moves=tuple(moves),
        duplicate_deletes=tuple(duplicate_deletes),
        skipped_disabled=skipped_disabled,
    )


def apply_migration_plan(plan: MigrationPlan) -> tuple[int, int, int, int]:
    moved = 0
    deleted = 0
    moved_bytes = 0
    deleted_bytes = 0

    for item in plan.moves:
        if not item.source_path.exists():
            continue
        if item.target_path.exists():
            continue
        item.target_path.parent.mkdir(parents=True, exist_ok=True)
        item.source_path.rename(item.target_path)
        moved += 1
        moved_bytes += item.size_bytes

    for item in plan.duplicate_deletes:
        if item.source_path.exists():
            deleted_bytes += item.source_path.stat().st_size
            item.source_path.unlink()
            deleted += 1

    return moved, deleted, moved_bytes, deleted_bytes


def _iter_child_experiment_names(*, parents: set[str] | None) -> Iterable[str]:
    parent_dirs = (
        [MedicalDatasetGenPaths.results_dir / parent for parent in sorted(parents)]
        if parents is not None
        else sorted(path for path in MedicalDatasetGenPaths.results_dir.iterdir() if path.is_dir())
    )
    for parent_dir in parent_dirs:
        if not parent_dir.is_dir():
            continue
        for subconfig_path in sorted(parent_dir.glob('*/_subconfig.yaml')):
            child_dir = subconfig_path.parent
            try:
                yield str(child_dir.relative_to(MedicalDatasetGenPaths.results_dir))
            except ValueError:
                continue


def _selected_tables(raw_tables: list[str] | None) -> tuple[SharedGenerationTableName, ...]:
    if not raw_tables:
        return SHARED_GENERATION_TABLES

    valid = set[str](SHARED_GENERATION_TABLES)
    invalid = sorted({table for table in raw_tables if table not in valid})
    if invalid:
        raise ValueError(
            'invalid shared table(s): '
            + ', '.join(invalid)
            + '. Valid tables: '
            + ', '.join(SHARED_GENERATION_TABLES)
        )
    return tuple(cast(SharedGenerationTableName, table) for table in raw_tables)


def _parent_names(raw_parents: list[str] | None) -> set[str] | None:
    if not raw_parents:
        return None
    return {resolve_experiment_name(value) for value in raw_parents}


def _print_plan(plan: MigrationPlan, *, apply: bool, summary_only: bool) -> None:
    move_bytes = sum(item.size_bytes for item in plan.moves)
    delete_bytes = sum(item.size_bytes for item in plan.duplicate_deletes)
    print(
        f'[shared-gen] mode={"apply" if apply else "dry-run"}, '
        f'moves={len(plan.moves):,}, move_bytes={_format_bytes(move_bytes)}, '
        f'duplicate_deletes={len(plan.duplicate_deletes):,}, '
        f'duplicate_bytes={_format_bytes(delete_bytes)}, '
        f'skipped_use_shared_false_or_parent={plan.skipped_disabled:,}'
    )
    if summary_only:
        return
    for item in plan.moves:
        print(f'[move] {item.table}: {item.source_path} -> {item.target_path}')
    for item in plan.duplicate_deletes:
        print(f'[delete duplicate] {item.table}: {item.source_path}')


def _format_bytes(value: int) -> str:
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f'{amount:.1f} {unit}' if unit != 'B' else f'{int(amount)} B'
        amount /= 1024.0
    raise RuntimeError('unreachable byte formatter state')


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Move existing generation-stage parquet artifacts into the per-parent '
            'schema-versioned _shared_v<version> mode '
            'directories. The script never copies artifact payloads.'
        )
    )
    parser.add_argument(
        '--parent',
        dest='parents',
        action='append',
        help='Parent experiment to migrate. May be repeated. Defaults to all parents.',
    )
    parser.add_argument(
        '--table',
        dest='tables',
        action='append',
        choices=SHARED_GENERATION_TABLES,
        help='Shared table to migrate. May be repeated. Defaults to all shared tables.',
    )
    parser.add_argument('--apply', action='store_true', help='Actually move/delete files.')
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Print only aggregate counts instead of every planned move/delete.',
    )
    parser.add_argument(
        '--delete-local-duplicates',
        action='store_true',
        help=(
            'After one source has populated each shared target, delete remaining local files for '
            'the same shared table key. This trusts the effective config grouping.'
        ),
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
