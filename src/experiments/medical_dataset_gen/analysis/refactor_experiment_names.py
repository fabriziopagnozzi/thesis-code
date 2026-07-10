from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

import yaml

from experiments.medical_dataset_gen.analysis.analysis_constants import (
    EXPERIMENT_FAMILIES,
    ExperimentFamilyId,
)
from experiments.medical_dataset_gen.analysis.profile_dataset_sizes import (
    DatasetSizeMarker,
)
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths

type RefactorAction = Literal['rename', 'skip']

DEFAULT_SIZE_MAPPING_RELATIVE_PATH = Path('_reports') / 'experiment_size_mapping.json'
DEFAULT_PLAN_RELATIVE_PATH = Path('_reports') / 'experiment_rename_plan.json'
ALIAS_FILENAME = '_experiment_aliases.json'
TEMP_PREFIX = '.__experiment_rename_tmp__'

FAMILY_PREFIXES: dict[ExperimentFamilyId, str] = {
    'balanced_clean': 'BAL',
    'dominance': 'DOM',
    'sparse_niche': 'NIC',
    'near_miss_heavy': 'MIS',
    'background_variant': 'BG',
    'budget_sweep': 'BUD',
    'embedding_comparison': 'EMB',
    'unknown': 'UNK',
}

SIZE_MARKERS: frozenset[DatasetSizeMarker] = frozenset({'S', 'M', 'L'})


@dataclass(frozen=True)
class ParentInput:
    old_name: str
    size_marker: DatasetSizeMarker
    family_id: ExperimentFamilyId
    total_chunks_per_query: int
    children: tuple[str, ...]


@dataclass(frozen=True)
class RenamePlanItem:
    old_name: str
    new_name: str
    action: RefactorAction
    family_id: ExperimentFamilyId
    family_prefix: str
    size_marker: DatasetSizeMarker
    group_number: int
    slug: str
    total_chunks_per_query: int
    children: tuple[str, ...]


class RenamePlanItemJson(TypedDict):
    old_name: str
    new_name: str
    action: RefactorAction
    family_id: str
    family_prefix: str
    size_marker: DatasetSizeMarker
    group_number: int
    slug: str
    total_chunks_per_query: int
    children: list[str]


class RenamePlanJson(TypedDict):
    schema_version: int
    applied: bool
    size_mapping_path: str
    results_dir: str
    naming_pattern: str
    plan: list[RenamePlanItemJson]
    aliases: dict[str, str]
    warnings: list[str]


class CliArgs(TypedDict):
    results_dir: Path
    size_mapping: Path
    plan_output: Path | None
    apply: bool
    force: bool


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    plan, aliases, warnings = build_rename_plan(
        results_dir=args['results_dir'],
        size_mapping_path=args['size_mapping'],
    )
    validate_rename_plan(
        results_dir=args['results_dir'],
        plan=plan,
        force=args['force'],
    )

    if args['apply']:
        apply_rename_plan(
            results_dir=args['results_dir'],
            plan=plan,
            aliases=aliases,
        )

    payload = _plan_payload(
        results_dir=args['results_dir'],
        size_mapping_path=args['size_mapping'],
        plan=plan,
        aliases=aliases,
        warnings=warnings,
        applied=args['apply'],
    )
    _write_or_print_plan(payload, args['plan_output'])


def build_rename_plan(
    *,
    results_dir: Path,
    size_mapping_path: Path,
) -> tuple[list[RenamePlanItem], dict[str, str], list[str]]:
    size_profile = _load_size_profile(size_mapping_path)
    parent_inputs, warnings = _load_parent_inputs(results_dir, size_profile)
    numbered_inputs = _number_parent_inputs(parent_inputs)

    plan: list[RenamePlanItem] = []
    aliases: dict[str, str] = {}
    for parent_input, group_number in numbered_inputs:
        family_prefix = FAMILY_PREFIXES[parent_input.family_id]
        slug = _slug_from_old_name(parent_input.old_name)
        new_name = f'{family_prefix}_{parent_input.size_marker}{group_number:02d}_{slug}'
        action: RefactorAction = 'skip' if new_name == parent_input.old_name else 'rename'
        item = RenamePlanItem(
            old_name=parent_input.old_name,
            new_name=new_name,
            action=action,
            family_id=parent_input.family_id,
            family_prefix=family_prefix,
            size_marker=parent_input.size_marker,
            group_number=group_number,
            slug=slug,
            total_chunks_per_query=parent_input.total_chunks_per_query,
            children=parent_input.children,
        )
        plan.append(item)
        if action == 'rename':
            aliases[parent_input.old_name] = new_name
            for child in parent_input.children:
                child_name = Path(child).name
                aliases[child] = f'{new_name}/{child_name}'

    return sorted(plan, key=lambda item: item.new_name), dict(sorted(aliases.items())), warnings


def validate_rename_plan(
    *,
    results_dir: Path,
    plan: Sequence[RenamePlanItem],
    force: bool,
) -> None:
    old_names = {item.old_name for item in plan}
    new_names = [item.new_name for item in plan]
    duplicates = sorted({name for name in new_names if new_names.count(name) > 1})
    if duplicates:
        raise ValueError(f'rename plan has duplicate target names: {duplicates}')

    for item in plan:
        old_path = results_dir / item.old_name
        new_path = results_dir / item.new_name
        if not old_path.is_dir():
            raise FileNotFoundError(f'missing source experiment directory: {old_path}')
        if new_path.exists() and item.new_name not in old_names and not force:
            raise FileExistsError(
                f'target already exists: {new_path}. Pass --force only after inspecting it.'
            )


def apply_rename_plan(
    *,
    results_dir: Path,
    plan: Sequence[RenamePlanItem],
    aliases: Mapping[str, str],
) -> None:
    rename_items = [item for item in plan if item.action == 'rename']
    temp_moves: list[tuple[Path, Path, Path]] = []
    for index, item in enumerate(rename_items, start=1):
        old_path = results_dir / item.old_name
        temp_path = results_dir / f'{TEMP_PREFIX}{index:03d}_{item.old_name}'
        new_path = results_dir / item.new_name
        if temp_path.exists():
            raise FileExistsError(f'temporary rename path already exists: {temp_path}')
        temp_moves.append((old_path, temp_path, new_path))

    for old_path, temp_path, _new_path in temp_moves:
        old_path.rename(temp_path)

    try:
        for _old_path, temp_path, new_path in temp_moves:
            temp_path.rename(new_path)
    except Exception:
        # Best effort rollback while source paths are still free.
        for old_path, temp_path, new_path in reversed(temp_moves):
            if temp_path.exists() and not old_path.exists():
                temp_path.rename(old_path)
            elif new_path.exists() and not old_path.exists():
                new_path.rename(old_path)
        raise

    _write_aliases(results_dir, aliases)


def _load_parent_inputs(
    results_dir: Path,
    size_profile: Mapping[str, object],
) -> tuple[list[ParentInput], list[str]]:
    parent_profiles = size_profile.get('parent_profiles')
    if not isinstance(parent_profiles, Mapping):
        raise ValueError('size mapping JSON must contain a parent_profiles object')

    parent_inputs: list[ParentInput] = []
    warnings: list[str] = []
    for raw_name, raw_profile in parent_profiles.items():
        if not isinstance(raw_name, str) or not isinstance(raw_profile, Mapping):
            warnings.append(f'skipped malformed parent profile entry: {raw_name!r}')
            continue
        if raw_name.startswith('_') or raw_name.startswith('00_scrapped/'):
            continue

        size_marker = _read_size_marker(raw_profile, raw_name)
        pool_mass = raw_profile.get('pool_mass')
        if not isinstance(pool_mass, Mapping):
            raise ValueError(f'{raw_name}: parent profile is missing pool_mass')
        total_chunks = pool_mass.get('total_chunks_per_query')
        if not isinstance(total_chunks, int):
            raise ValueError(f'{raw_name}: pool_mass.total_chunks_per_query must be an int')

        children = _read_children(raw_profile, raw_name)
        family_id = _load_family_id(results_dir / raw_name / '_exp_family.yaml', raw_name)
        parent_inputs.append(
            ParentInput(
                old_name=raw_name,
                size_marker=size_marker,
                family_id=family_id,
                total_chunks_per_query=total_chunks,
                children=children,
            )
        )
    return parent_inputs, warnings


def _number_parent_inputs(parent_inputs: Sequence[ParentInput]) -> list[tuple[ParentInput, int]]:
    grouped: dict[tuple[ExperimentFamilyId, DatasetSizeMarker], list[ParentInput]] = {}
    for parent_input in parent_inputs:
        grouped.setdefault((parent_input.family_id, parent_input.size_marker), []).append(
            parent_input
        )

    numbered: list[tuple[ParentInput, int]] = []
    for group_items in grouped.values():
        ordered = sorted(
            group_items,
            key=lambda item: (
                item.total_chunks_per_query,
                _slug_from_old_name(item.old_name),
                item.old_name,
            ),
        )
        numbered.extend((item, index) for index, item in enumerate(ordered, start=1))
    return numbered


def _slug_from_old_name(old_name: str) -> str:
    base_name = Path(old_name).name
    family_prefix_pattern = '|'.join(sorted(FAMILY_PREFIXES.values(), key=len, reverse=True))
    slug = re.sub(rf'^(?:{family_prefix_pattern})_[SML]\d+_', '', base_name)
    if slug != base_name:
        slug = re.sub(r'[^A-Za-z0-9]+', '_', slug).strip('_').lower()
        slug = re.sub(r'_+', '_', slug)
        if not slug:
            raise ValueError(f'could not derive slug from experiment name {old_name!r}')
        return slug

    slug = re.sub(r'^(?:[SM]\d+[a-z]?|\d+P?)_', '', base_name)
    slug = re.sub(r'^(?:small|medium|large)_', '', slug)
    slug = re.sub(r'[^A-Za-z0-9]+', '_', slug).strip('_').lower()
    slug = re.sub(r'_+', '_', slug)
    if not slug:
        raise ValueError(f'could not derive slug from experiment name {old_name!r}')
    return slug


def _load_family_id(path: Path, experiment_name: str) -> ExperimentFamilyId:
    if not path.is_file():
        raise FileNotFoundError(f'{experiment_name}: missing family metadata {path}')
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f'{experiment_name}: family metadata must be a YAML mapping')
    family_id = raw.get('family_id')
    if not isinstance(family_id, str) or family_id not in EXPERIMENT_FAMILIES:
        raise ValueError(f'{experiment_name}: invalid family_id {family_id!r}')
    return cast(ExperimentFamilyId, family_id)


def _read_size_marker(raw_profile: Mapping[str, object], experiment_name: str) -> DatasetSizeMarker:
    raw_size_marker = raw_profile.get('size_marker')
    if not isinstance(raw_size_marker, str) or raw_size_marker not in SIZE_MARKERS:
        raise ValueError(f'{experiment_name}: invalid size marker {raw_size_marker!r}')
    return cast(DatasetSizeMarker, raw_size_marker)


def _read_children(raw_profile: Mapping[str, object], experiment_name: str) -> tuple[str, ...]:
    raw_children = raw_profile.get('children')
    if not isinstance(raw_children, list) or not all(
        isinstance(child, str) for child in raw_children
    ):
        raise ValueError(f'{experiment_name}: children must be a list of strings')
    return tuple(cast(list[str], raw_children))


def _load_size_profile(path: Path) -> Mapping[str, object]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, Mapping):
        raise ValueError(f'size mapping JSON must be an object: {path}')
    return raw


def _write_aliases(results_dir: Path, aliases: Mapping[str, str]) -> None:
    aliases_path = results_dir / ALIAS_FILENAME
    existing: dict[str, str] = {}
    if aliases_path.is_file():
        raw = json.loads(aliases_path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f'existing aliases file must contain a JSON object: {aliases_path}')
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError(f'existing aliases must map strings to strings: {aliases_path}')
            existing[key] = value

    merged = {**existing, **aliases}
    text = json.dumps(dict(sorted(merged.items())), indent=2, sort_keys=True)
    aliases_path.write_text(f'{text}\n')


def _plan_payload(
    *,
    results_dir: Path,
    size_mapping_path: Path,
    plan: Sequence[RenamePlanItem],
    aliases: Mapping[str, str],
    warnings: Sequence[str],
    applied: bool,
) -> RenamePlanJson:
    return {
        'schema_version': 1,
        'applied': applied,
        'size_mapping_path': str(size_mapping_path),
        'results_dir': str(results_dir),
        'naming_pattern': '<family-prefix>_<size-marker><number>_<slug>',
        'plan': [_plan_item_json(item) for item in plan],
        'aliases': dict(sorted(aliases.items())),
        'warnings': list(warnings),
    }


def _plan_item_json(item: RenamePlanItem) -> RenamePlanItemJson:
    return {
        'old_name': item.old_name,
        'new_name': item.new_name,
        'action': item.action,
        'family_id': item.family_id,
        'family_prefix': item.family_prefix,
        'size_marker': item.size_marker,
        'group_number': item.group_number,
        'slug': item.slug,
        'total_chunks_per_query': item.total_chunks_per_query,
        'children': list(item.children),
    }


def _write_or_print_plan(payload: RenamePlanJson, output_path: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output_path is None:
        print(text)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f'{text}\n')
    print(output_path)


def _parse_args(argv: Sequence[str] | None) -> CliArgs:
    parser = argparse.ArgumentParser(
        description='Rename parent experiment result directories using family and profiled size.'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=MedicalDatasetGenPaths.results_dir,
        help='Root _results directory containing experiment folders.',
    )
    parser.add_argument(
        '--size-mapping',
        type=Path,
        default=None,
        help='JSON produced by analysis.profile_dataset_sizes. Defaults to <results-dir>/_reports/experiment_size_mapping.json.',
    )
    parser.add_argument(
        '--plan-output',
        type=Path,
        default=None,
        help='Where to write the JSON rename plan. Defaults to <results-dir>/_reports/experiment_rename_plan.json. Use "-" for stdout.',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Apply the rename plan. Without this flag, only writes the plan.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Allow pre-existing target paths after manual inspection.',
    )
    namespace = parser.parse_args(argv)
    results_dir = cast(Path, namespace.results_dir)
    raw_size_mapping = cast(Path | None, namespace.size_mapping)
    raw_plan_output = cast(Path | None, namespace.plan_output)
    size_mapping = (
        results_dir / DEFAULT_SIZE_MAPPING_RELATIVE_PATH
        if raw_size_mapping is None
        else raw_size_mapping
    )
    plan_output = (
        results_dir / DEFAULT_PLAN_RELATIVE_PATH if raw_plan_output is None else raw_plan_output
    )
    return {
        'results_dir': results_dir,
        'size_mapping': size_mapping,
        'plan_output': None if str(plan_output) == '-' else plan_output,
        'apply': bool(namespace.apply),
        'force': bool(namespace.force),
    }


if __name__ == '__main__':
    main()
