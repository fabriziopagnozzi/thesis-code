from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import yaml

from experiments.medical_dataset_gen.utils.exp_naming import embedding_child_token
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths

type RenameAction = Literal['rename', 'skip']
type QueryScope = Literal['pass_only', 'all_query', 'unknown']

ALIAS_FILENAME = '_experiment_aliases.json'
TEMP_RENAME_PREFIX = '.__embedding_child_rename_tmp__'
TEXT_ARTIFACT_NAMES = ('_subconfig.yaml', 'embeddings_metadata.json')


@dataclass(frozen=True)
class ChildRenamePlanItem:
    parent_name: str
    old_child_name: str
    new_child_name: str
    old_experiment: str
    new_experiment: str
    model_name: str
    model_token: str
    query_scope: QueryScope
    action: RenameAction


class ChildRenamePlanItemJson(TypedDict):
    parent_name: str
    old_child_name: str
    new_child_name: str
    old_experiment: str
    new_experiment: str
    model_name: str
    model_token: str
    query_scope: QueryScope
    action: RenameAction


class RenamePlanJson(TypedDict):
    schema_version: int
    applied: bool
    results_dir: str
    naming_rule: str
    plan: list[ChildRenamePlanItemJson]
    aliases_added_or_updated: dict[str, str]
    warnings: list[str]


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    results_dir = args.results_dir
    plan, aliases, warnings = build_rename_plan(
        results_dir=results_dir,
        include_scrapped=args.include_scrapped,
    )
    validate_rename_plan(results_dir=results_dir, plan=plan, force=args.force)

    if args.apply:
        apply_rename_plan(results_dir=results_dir, plan=plan, aliases=aliases)

    payload = _plan_payload(
        results_dir=results_dir,
        plan=plan,
        aliases=aliases,
        warnings=warnings,
        applied=args.apply,
    )
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if args.plan_output is None:
        print(serialized)
    else:
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(serialized + '\n')
        print(f'wrote rename plan to {args.plan_output}')


@dataclass(frozen=True)
class CliArgs:
    results_dir: Path
    plan_output: Path | None
    apply: bool
    force: bool
    include_scrapped: bool


def build_rename_plan(
    *,
    results_dir: Path,
    include_scrapped: bool,
) -> tuple[list[ChildRenamePlanItem], dict[str, str], list[str]]:
    aliases = _load_aliases(results_dir)
    plan: list[ChildRenamePlanItem] = []
    warnings: list[str] = []

    for subconfig_path in sorted(results_dir.glob('*/*/_subconfig.yaml')):
        parent_path = subconfig_path.parent.parent
        child_path = subconfig_path.parent
        parent_name = parent_path.name
        old_child_name = child_path.name
        if parent_name.startswith('_'):
            continue
        if parent_name == '00_scrapped' and not include_scrapped:
            continue

        child_config = _read_yaml_mapping(subconfig_path)
        parent_config = _read_yaml_mapping(parent_path / '_config.yaml')
        metadata = _read_json_mapping(child_path / 'embeddings_metadata.json')
        model_name = _model_name(child_config, metadata)
        if model_name is None:
            warnings.append(f'{parent_name}/{old_child_name}: missing embeddings.model_name')
            continue

        model_token = embedding_child_token(model_name)
        query_scope = _query_scope(child_config, parent_config)
        new_child_name = model_token if query_scope != 'all_query' else f'{model_token}_allq'
        old_experiment = f'{parent_name}/{old_child_name}'
        new_experiment = f'{parent_name}/{new_child_name}'
        action: RenameAction = 'skip' if old_child_name == new_child_name else 'rename'
        plan.append(
            ChildRenamePlanItem(
                parent_name=parent_name,
                old_child_name=old_child_name,
                new_child_name=new_child_name,
                old_experiment=old_experiment,
                new_experiment=new_experiment,
                model_name=model_name,
                model_token=model_token,
                query_scope=query_scope,
                action=action,
            )
        )

    alias_updates = _alias_updates_for_plan(plan, aliases)
    return sorted(plan, key=lambda item: item.new_experiment), alias_updates, warnings


def validate_rename_plan(
    *,
    results_dir: Path,
    plan: Sequence[ChildRenamePlanItem],
    force: bool,
) -> None:
    targets_by_parent: dict[str, list[str]] = {}
    for item in plan:
        targets_by_parent.setdefault(item.parent_name, []).append(item.new_child_name)
    duplicates = sorted(
        f'{parent}/{target}'
        for parent, targets in targets_by_parent.items()
        for target in set(targets)
        if targets.count(target) > 1
    )
    if duplicates:
        raise ValueError(f'rename plan has duplicate child targets: {duplicates}')

    current_experiments = {item.old_experiment for item in plan}
    for item in plan:
        old_path = results_dir / item.old_experiment
        new_path = results_dir / item.new_experiment
        if not old_path.is_dir():
            raise FileNotFoundError(f'missing source child experiment directory: {old_path}')
        if new_path.exists() and item.new_experiment not in current_experiments and not force:
            raise FileExistsError(
                f'target already exists: {new_path}. Pass --force only after inspecting it.'
            )


def apply_rename_plan(
    *,
    results_dir: Path,
    plan: Sequence[ChildRenamePlanItem],
    aliases: Mapping[str, str],
) -> None:
    rename_items = [item for item in plan if item.action == 'rename']
    replacements = _text_replacements(results_dir=results_dir, plan=plan)
    _rename_child_dirs(results_dir=results_dir, plan=rename_items)
    _rewrite_text_artifacts(results_dir=results_dir, plan=plan, replacements=replacements)
    _write_aliases(results_dir=results_dir, alias_updates=aliases)


def _parse_args(argv: Sequence[str] | None) -> CliArgs:
    parser = argparse.ArgumentParser(
        description='Rename embedding-comparison child experiments to compact model tokens.',
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=MedicalDatasetGenPaths.results_dir,
        help='Results directory containing parent/child experiment folders.',
    )
    parser.add_argument(
        '--plan-output',
        type=Path,
        default=None,
        help='Optional JSON file for the dry-run/apply plan. Prints to stdout when omitted.',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually rename child directories and update aliases/text artifact references.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Allow target paths that already exist outside the current rename plan.',
    )
    parser.add_argument(
        '--include-scrapped',
        action='store_true',
        help='Also process child experiments under 00_scrapped.',
    )
    namespace = parser.parse_args(argv)
    return CliArgs(
        results_dir=namespace.results_dir,
        plan_output=namespace.plan_output,
        apply=namespace.apply,
        force=namespace.force,
        include_scrapped=namespace.include_scrapped,
    )


def _rename_child_dirs(
    *,
    results_dir: Path,
    plan: Sequence[ChildRenamePlanItem],
) -> None:
    temp_moves: list[tuple[Path, Path, Path]] = []
    for index, item in enumerate(plan, start=1):
        old_path = results_dir / item.old_experiment
        temp_path = results_dir / item.parent_name / f'{TEMP_RENAME_PREFIX}{index:03d}'
        new_path = results_dir / item.new_experiment
        if temp_path.exists():
            raise FileExistsError(f'temporary rename path already exists: {temp_path}')
        temp_moves.append((old_path, temp_path, new_path))

    for old_path, temp_path, _new_path in temp_moves:
        old_path.rename(temp_path)

    try:
        for _old_path, temp_path, new_path in temp_moves:
            temp_path.rename(new_path)
    except Exception:
        for old_path, temp_path, new_path in reversed(temp_moves):
            if temp_path.exists() and not old_path.exists():
                temp_path.rename(old_path)
            elif new_path.exists() and not old_path.exists():
                new_path.rename(old_path)
        raise


def _rewrite_text_artifacts(
    *,
    results_dir: Path,
    plan: Sequence[ChildRenamePlanItem],
    replacements: Mapping[str, str],
) -> None:
    for item in plan:
        child_dir = results_dir / item.new_experiment
        for artifact_name in TEXT_ARTIFACT_NAMES:
            path = child_dir / artifact_name
            if not path.is_file():
                continue
            original = path.read_text()
            updated = _replace_all(original, replacements)
            if updated != original:
                path.write_text(updated)


def _text_replacements(
    *,
    results_dir: Path,
    plan: Sequence[ChildRenamePlanItem],
) -> dict[str, str]:
    aliases = _load_aliases(results_dir)
    experiment_replacements: dict[str, str] = {
        item.old_experiment: item.new_experiment for item in plan if item.action == 'rename'
    }
    for alias, target in aliases.items():
        replacement = experiment_replacements.get(target)
        if replacement is not None:
            experiment_replacements[alias] = replacement

    replacements: dict[str, str] = {}
    result_roots = [results_dir, results_dir.resolve()]
    for old_experiment, new_experiment in experiment_replacements.items():
        replacements[old_experiment] = new_experiment
        for result_root in result_roots:
            replacements[str(result_root / old_experiment)] = str(result_root / new_experiment)
    return dict(sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True))


def _replace_all(value: str, replacements: Mapping[str, str]) -> str:
    out = value
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def _alias_updates_for_plan(
    plan: Sequence[ChildRenamePlanItem],
    aliases: Mapping[str, str],
) -> dict[str, str]:
    direct_updates = {
        item.old_experiment: item.new_experiment for item in plan if item.action == 'rename'
    }
    updates = dict(direct_updates)
    for alias, target in aliases.items():
        replacement = direct_updates.get(target)
        if replacement is not None:
            updates[alias] = replacement
    return dict(sorted(updates.items()))


def _write_aliases(
    *,
    results_dir: Path,
    alias_updates: Mapping[str, str],
) -> None:
    aliases = _load_aliases(results_dir)
    aliases.update(alias_updates)
    alias_path = results_dir / ALIAS_FILENAME
    alias_path.write_text(json.dumps(dict(sorted(aliases.items())), indent=2) + '\n')


def _load_aliases(results_dir: Path) -> dict[str, str]:
    alias_path = results_dir / ALIAS_FILENAME
    if not alias_path.is_file():
        return {}
    raw = json.loads(alias_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f'experiment aliases file must contain an object: {alias_path}')
    return {str(key): str(value) for key, value in raw.items()}


def _read_yaml_mapping(path: Path) -> Mapping[str, object]:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f'expected YAML mapping at {path}')
    return {str(key): value for key, value in raw.items()}


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): value for key, value in raw.items()}


def _model_name(
    child_config: Mapping[str, object],
    metadata: Mapping[str, object],
) -> str | None:
    configured = _nested_string(child_config, 'embeddings', 'model_name')
    if configured is not None:
        return configured
    raw_metadata_model = metadata.get('model_name')
    return raw_metadata_model if isinstance(raw_metadata_model, str) else None


def _query_scope(
    child_config: Mapping[str, object],
    parent_config: Mapping[str, object],
) -> QueryScope:
    raw_retrieval = child_config.get('retrieval')
    parent_retrieval = parent_config.get('retrieval')
    if not isinstance(raw_retrieval, Mapping):
        raw_retrieval = parent_retrieval
    if not isinstance(raw_retrieval, Mapping):
        return 'unknown'
    only_pass_geometry = raw_retrieval.get('only_pass_geometry')
    if only_pass_geometry is None and isinstance(parent_retrieval, Mapping):
        only_pass_geometry = parent_retrieval.get('only_pass_geometry')
    if only_pass_geometry is True:
        return 'pass_only'
    if only_pass_geometry is False:
        return 'all_query'
    return 'unknown'


def _nested_string(mapping: Mapping[str, object], section: str, field: str) -> str | None:
    raw_section = mapping.get(section)
    if not isinstance(raw_section, Mapping):
        return None
    raw_value = raw_section.get(field)
    return raw_value if isinstance(raw_value, str) else None


def _plan_payload(
    *,
    results_dir: Path,
    plan: Sequence[ChildRenamePlanItem],
    aliases: Mapping[str, str],
    warnings: Sequence[str],
    applied: bool,
) -> RenamePlanJson:
    return {
        'schema_version': 1,
        'applied': applied,
        'results_dir': str(results_dir),
        'naming_rule': 'pass-only children use the compact embedding token; all-query children append _allq',
        'plan': [_plan_item_payload(item) for item in plan],
        'aliases_added_or_updated': dict(sorted(aliases.items())),
        'warnings': list(warnings),
    }


def _plan_item_payload(item: ChildRenamePlanItem) -> ChildRenamePlanItemJson:
    return {
        'parent_name': item.parent_name,
        'old_child_name': item.old_child_name,
        'new_child_name': item.new_child_name,
        'old_experiment': item.old_experiment,
        'new_experiment': item.new_experiment,
        'model_name': item.model_name,
        'model_token': item.model_token,
        'query_scope': item.query_scope,
        'action': item.action,
    }


if __name__ == '__main__':
    main()
