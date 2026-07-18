from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ruamel.yaml import YAML

from experiments.medical_dataset_gen.utils.exp_naming import resolve_experiment_name
from experiments.medical_dataset_gen.utils.global_utils import (
    EMBEDDING_ARTIFACT_FILENAMES,
    EmbeddingArtifactName,
    MedicalDatasetGenPaths,
    ResultDirOverrides,
    load_config,
    paths_for,
)

ARTIFACTS: tuple[EmbeddingArtifactName, ...] = (
    'chunk_ids',
    'query_ids',
    'chunk_vectors',
    'query_vectors',
    'metadata',
)


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    exp_name: str
    artifact: EmbeddingArtifactName
    path: Path


@dataclass(frozen=True, slots=True)
class TargetPlan:
    target_path: Path
    sources: tuple[SourceCandidate, ...]

    @property
    def existing_sources(self) -> tuple[SourceCandidate, ...]:
        return tuple(source for source in self.sources if source.path.exists())

    @property
    def target_exists(self) -> bool:
        return self.target_path.exists()


def main() -> None:
    args = _parse_args()
    parents = _parent_names(args.parents)
    artifacts = _selected_artifacts(args.artifacts)
    plans = _build_plans(parents=parents, artifacts=artifacts)

    move_count = sum(
        1 for plan in plans if not plan.target_exists and bool(plan.existing_sources)
    )
    move_bytes = sum(
        plan.existing_sources[0].path.stat().st_size
        for plan in plans
        if not plan.target_exists and bool(plan.existing_sources)
    )
    duplicate_delete_count = sum(
        len(
            _deletable_old_sources(
                plan,
                kept_source=None if plan.target_exists else plan.existing_sources[0],
            )
        )
        for plan in plans
        if bool(plan.existing_sources)
    )
    duplicate_delete_bytes = sum(
        source.path.stat().st_size
        for plan in plans
        if bool(plan.existing_sources)
        for source in _deletable_old_sources(
            plan,
            kept_source=None if plan.target_exists else plan.existing_sources[0],
        )
        if source.path.exists()
    )
    missing_count = sum(
        1 for plan in plans if not plan.target_exists and not plan.existing_sources
    )
    strip_count = _count_embedding_override_strips(parents=parents, artifacts=artifacts)

    mode = 'apply' if args.apply else 'dry-run'
    print(
        '[embeddings-migration] '
        f'mode={mode}, targets={len(plans):,}, moves={move_count:,}, '
        f'move_bytes={_format_bytes(move_bytes)}, '
        f'duplicate_deletes={duplicate_delete_count:,}, '
        f'duplicate_bytes={_format_bytes(duplicate_delete_bytes)}, '
        f'missing_sources={missing_count:,}, '
        f'override_strips={strip_count:,}'
    )

    if not args.summary_only:
        _print_plan(plans)

    if not args.apply:
        print('[embeddings-migration] dry run only. Re-run with --apply to move files.')
        return

    moved, deleted = _apply_plans(
        plans,
        delete_local_duplicates=args.delete_local_duplicates,
    )
    stripped = (
        _strip_embedding_overrides(parents=parents, artifacts=artifacts)
        if args.strip_embedding_overrides
        else 0
    )
    print(
        f'[embeddings-migration] applied: '
        f'moved={moved:,}, deleted={deleted:,}, stripped_subconfigs={stripped:,}'
    )


def _build_plans(
    *,
    parents: set[str] | None,
    artifacts: tuple[EmbeddingArtifactName, ...],
) -> list[TargetPlan]:
    grouped: dict[Path, list[SourceCandidate]] = defaultdict(list)
    for exp_name in _iter_child_experiment_names(parents=parents):
        cfg = load_config(exp_name)
        if not cfg.global_.use_shared:
            continue
        paths = paths_for(cfg)
        exp_dir = MedicalDatasetGenPaths.results_dir / exp_name
        for artifact in artifacts:
            target_path = paths.embeddings_paths(artifact)
            for source_path in _legacy_embedding_source_paths(
                exp_dir=exp_dir,
                result_dir_overrides=cfg.global_.result_dir_overrides,
                artifact=artifact,
            ):
                grouped[target_path].append(
                    SourceCandidate(
                        exp_name=exp_name,
                        artifact=artifact,
                        path=source_path,
                    )
                )

    return [
        TargetPlan(target_path=target_path, sources=tuple(_unique_sources(sources)))
        for target_path, sources in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]


def _legacy_embedding_source_paths(
    *,
    exp_dir: Path,
    result_dir_overrides: ResultDirOverrides,
    artifact: EmbeddingArtifactName,
) -> tuple[Path, ...]:
    filename = EMBEDDING_ARTIFACT_FILENAMES[artifact]
    paths: list[Path] = []
    override = result_dir_overrides.get(artifact)
    if override is not None:
        override_path = Path(override)
        if override_path.is_absolute():
            paths.append(override_path / filename)
        else:
            paths.append(MedicalDatasetGenPaths.results_dir / override_path / filename)
    paths.append(exp_dir / filename)
    return tuple(dict.fromkeys(paths))


def _unique_sources(sources: Sequence[SourceCandidate]) -> Iterable[SourceCandidate]:
    seen: set[Path] = set()
    for source in sorted(sources, key=lambda item: (str(item.path), item.exp_name)):
        if source.path in seen:
            continue
        seen.add(source.path)
        yield source


def _deletable_old_sources(
    plan: TargetPlan,
    *,
    kept_source: SourceCandidate | None,
) -> tuple[SourceCandidate, ...]:
    return tuple(
        source
        for source in plan.existing_sources
        if source.path != plan.target_path
        and (kept_source is None or source.path != kept_source.path)
    )


def _apply_plans(
    plans: Sequence[TargetPlan],
    *,
    delete_local_duplicates: bool,
) -> tuple[int, int]:
    moved = 0
    deleted = 0
    for plan in plans:
        existing_sources = plan.existing_sources
        moved_source: SourceCandidate | None = None
        if not plan.target_exists and existing_sources:
            moved_source = existing_sources[0]
            plan.target_path.parent.mkdir(parents=True, exist_ok=True)
            moved_source.path.rename(plan.target_path)
            moved += 1
        elif existing_sources:
            moved_source = existing_sources[0]

        if not delete_local_duplicates or moved_source is None:
            continue

        kept_source = None if plan.target_exists else moved_source
        for source in _deletable_old_sources(plan, kept_source=kept_source):
            if source.path.exists():
                source.path.unlink()
                deleted += 1
    return moved, deleted


def _print_plan(plans: Sequence[TargetPlan]) -> None:
    for plan in plans:
        existing_sources = plan.existing_sources
        status = 'exists' if plan.target_exists else 'move' if existing_sources else 'missing'
        print(f'[{status}] {plan.target_path}')
        for source in existing_sources[:5]:
            print(f'  <- {source.path} ({source.exp_name})')
        if len(existing_sources) > 5:
            print(f'  ... {len(existing_sources) - 5:,} more source(s)')


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
                yield child_dir.relative_to(MedicalDatasetGenPaths.results_dir).as_posix()
            except ValueError:
                continue


def _count_embedding_override_strips(
    *,
    parents: set[str] | None,
    artifacts: tuple[EmbeddingArtifactName, ...],
) -> int:
    return sum(
        1
        for exp_name in _iter_child_experiment_names(parents=parents)
        if _subconfig_has_embedding_overrides(exp_name, artifacts)
    )


def _subconfig_has_embedding_overrides(
    exp_name: str,
    artifacts: tuple[EmbeddingArtifactName, ...],
) -> bool:
    cfg = load_config(exp_name)
    if not cfg.global_.use_shared:
        return False
    subconfig_path = MedicalDatasetGenPaths.results_dir / exp_name / '_subconfig.yaml'
    raw = _read_yaml(subconfig_path)
    global_section = raw.get('global')
    if not isinstance(global_section, dict):
        return False
    result_dir_overrides = global_section.get('result_dir_overrides')
    if not isinstance(result_dir_overrides, dict):
        return False
    return any(artifact in result_dir_overrides for artifact in artifacts)


def _strip_embedding_overrides(
    *,
    parents: set[str] | None,
    artifacts: tuple[EmbeddingArtifactName, ...],
) -> int:
    changed = 0
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    for exp_name in _iter_child_experiment_names(parents=parents):
        cfg = load_config(exp_name)
        if not cfg.global_.use_shared:
            continue
        subconfig_path = MedicalDatasetGenPaths.results_dir / exp_name / '_subconfig.yaml'
        raw = _read_yaml(subconfig_path)
        global_section = raw.get('global')
        if not isinstance(global_section, dict):
            continue
        result_dir_overrides = global_section.get('result_dir_overrides')
        if not isinstance(result_dir_overrides, dict):
            continue

        removed = False
        for artifact in artifacts:
            if artifact in result_dir_overrides:
                del result_dir_overrides[artifact]
                removed = True
        if not removed:
            continue
        if not result_dir_overrides:
            del global_section['result_dir_overrides']
        with open(subconfig_path, 'w') as file:
            yaml.dump(raw, file)
        changed += 1
    return changed


def _read_yaml(path: Path) -> dict[str, object]:
    yaml = YAML(typ='rt')
    with open(path) as file:
        raw = yaml.load(file)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f'YAML file must contain a mapping: {path}')
    return cast(dict[str, object], raw)


def _selected_artifacts(raw_artifacts: list[str] | None) -> tuple[EmbeddingArtifactName, ...]:
    if not raw_artifacts:
        return ARTIFACTS
    valid = set[str](ARTIFACTS)
    invalid = sorted({artifact for artifact in raw_artifacts if artifact not in valid})
    if invalid:
        raise ValueError(
            'invalid artifact(s): '
            + ', '.join(invalid)
            + '. Valid artifacts: '
            + ', '.join(ARTIFACTS)
        )
    return tuple(cast(EmbeddingArtifactName, artifact) for artifact in raw_artifacts)


def _parent_names(raw_parents: list[str] | None) -> set[str] | None:
    if not raw_parents:
        return None
    return {
        resolve_experiment_name(parent, results_dir=MedicalDatasetGenPaths.results_dir)
        for parent in raw_parents
    }


def _format_bytes(value: int) -> str:
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f'{amount:.1f} {unit}' if unit != 'B' else f'{int(amount)} B'
        amount /= 1024.0
    return f'{value} B'


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Move old local/canonical embedding arrays into parent _embeddings storage.'
    )
    parser.add_argument(
        '--parent',
        dest='parents',
        action='append',
        default=None,
        help='Parent experiment to migrate. May be repeated. Defaults to all parents.',
    )
    parser.add_argument(
        '--artifact',
        dest='artifacts',
        action='append',
        choices=ARTIFACTS,
        default=None,
        help='Embedding artifact to migrate. May be repeated. Defaults to all artifacts.',
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Only print aggregate counts.',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually move files. Defaults to dry-run.',
    )
    parser.add_argument(
        '--delete-local-duplicates',
        action='store_true',
        help='After a target is populated, delete remaining old source files for that target.',
    )
    parser.add_argument(
        '--strip-embedding-overrides',
        action='store_true',
        help='Remove migrated embedding artifact keys from child global.result_dir_overrides.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
