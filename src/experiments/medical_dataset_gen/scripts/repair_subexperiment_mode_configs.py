from __future__ import annotations

import argparse
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from ruamel.yaml import YAML

from experiments.medical_dataset_gen.schemas.generation_schemas import (
    ChunkTextStyle,
    QueryFocusMode,
    QueryStructure,
)
from experiments.medical_dataset_gen.utils.exp_naming import resolve_experiment_name
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    YamlMapping,
)

type QueryModeToken = Literal['biased', 'unbiased']
type ChunkModeToken = Literal['simple', 'hardened']

_CHILD_NAME_RE = re.compile(
    r'^(?P<query_mode>biased|unbiased)_q_'
    r'(?P<focus_mode>list|natural)_f_'
    r'(?P<chunk_mode>simple|hardened)_c(?:_.+)?$'
)
_QUERY_STRUCTURE_BY_QUERY_MODE: dict[QueryModeToken, QueryStructure] = {
    'biased': 'unbalanced',
    'unbiased': 'balanced',
}
_CHUNK_TEXT_STYLE_BY_CHUNK_MODE: dict[ChunkModeToken, ChunkTextStyle] = {
    'simple': 'ontology_explicit',
    'hardened': 'semantic_hardened',
}


@dataclass(frozen=True, slots=True)
class ExpectedModes:
    query_mode: QueryModeToken
    focus_mode: QueryFocusMode
    chunk_mode: ChunkModeToken
    query_structure: QueryStructure
    chunk_text_style: ChunkTextStyle


@dataclass(frozen=True, slots=True)
class FieldRepair:
    field: str
    current: object
    expected: str


@dataclass(frozen=True, slots=True)
class SubconfigRepairPlan:
    subconfig_path: Path
    exp_name: str
    expected: ExpectedModes
    repairs: tuple[FieldRepair, ...]


def main() -> None:
    args = _parse_args()
    parents = _parent_names(args.parents)
    plans, skipped = _build_repair_plans(parents=parents)
    _print_plans(plans, skipped=skipped, apply=bool(args.apply))
    if not args.apply:
        print('[mode-repair] dry run only. Re-run with --apply to update _subconfig.yaml files.')
        return

    changed = _apply_repair_plans(plans)
    print(f'[mode-repair] applied: updated_subconfigs={changed:,}')


def _build_repair_plans(
    *,
    parents: set[str] | None,
) -> tuple[list[SubconfigRepairPlan], int]:
    plans: list[SubconfigRepairPlan] = []
    skipped = 0
    yaml = _yaml()
    for subconfig_path in _iter_subconfig_paths(parents=parents):
        child_name = subconfig_path.parent.name
        expected = _expected_modes_from_child_name(child_name)
        if expected is None:
            skipped += 1
            continue
        raw = _read_yaml(subconfig_path, yaml)
        generation = raw.get('generation')
        generation_map = cast(YamlMapping, generation) if isinstance(generation, dict) else {}
        repairs = _field_repairs(generation_map, expected)
        if not repairs:
            continue
        try:
            exp_name = str(subconfig_path.parent.relative_to(MedicalDatasetGenPaths.results_dir))
        except ValueError:
            exp_name = str(subconfig_path.parent)
        plans.append(
            SubconfigRepairPlan(
                subconfig_path=subconfig_path,
                exp_name=exp_name,
                expected=expected,
                repairs=tuple(repairs),
            )
        )
    return sorted(plans, key=lambda plan: plan.exp_name), skipped


def _expected_modes_from_child_name(child_name: str) -> ExpectedModes | None:
    match = _CHILD_NAME_RE.fullmatch(child_name)
    if match is None:
        return None
    query_mode = cast(QueryModeToken, match.group('query_mode'))
    focus_mode = cast(QueryFocusMode, match.group('focus_mode'))
    chunk_mode = cast(ChunkModeToken, match.group('chunk_mode'))
    return ExpectedModes(
        query_mode=query_mode,
        focus_mode=focus_mode,
        chunk_mode=chunk_mode,
        query_structure=_QUERY_STRUCTURE_BY_QUERY_MODE[query_mode],
        chunk_text_style=_CHUNK_TEXT_STYLE_BY_CHUNK_MODE[chunk_mode],
    )


def _field_repairs(
    generation: YamlMapping,
    expected: ExpectedModes,
) -> list[FieldRepair]:
    expected_fields = {
        'focus_mode': expected.focus_mode,
        'query_structure': expected.query_structure,
        'chunk_text_style': expected.chunk_text_style,
    }
    repairs: list[FieldRepair] = []
    for field, expected_value in expected_fields.items():
        current = generation.get(field)
        if current != expected_value:
            repairs.append(FieldRepair(field=field, current=current, expected=expected_value))
    return repairs


def _apply_repair_plans(plans: list[SubconfigRepairPlan]) -> int:
    yaml = _yaml()
    changed = 0
    for plan in plans:
        raw = _read_yaml(plan.subconfig_path, yaml)
        generation = _ensure_mapping(raw, 'generation')
        generation['focus_mode'] = plan.expected.focus_mode
        generation['query_structure'] = plan.expected.query_structure
        generation['chunk_text_style'] = plan.expected.chunk_text_style
        _write_yaml_atomic(plan.subconfig_path, raw, yaml)
        changed += 1
    return changed


def _print_plans(
    plans: list[SubconfigRepairPlan],
    *,
    skipped: int,
    apply: bool,
) -> None:
    mode = 'apply' if apply else 'dry-run'
    print(
        f'[mode-repair] mode={mode}, mismatched_subconfigs={len(plans):,}, '
        f'skipped_nonmatching_names={skipped:,}'
    )
    for plan in plans:
        print(
            f'[{plan.exp_name}] '
            f'query_mode={plan.expected.query_mode}, focus_mode={plan.expected.focus_mode}, '
            f'chunk_mode={plan.expected.chunk_mode}'
        )
        for repair in plan.repairs:
            print(f'  {repair.field}: {repair.current!r} -> {repair.expected!r}')


def _iter_subconfig_paths(*, parents: set[str] | None) -> Iterable[Path]:
    if parents is not None:
        for parent in sorted(parents):
            parent_dir = MedicalDatasetGenPaths.results_dir / parent
            yield from sorted(parent_dir.glob('*/_subconfig.yaml'))
        return

    for parent_dir in sorted(
        path for path in MedicalDatasetGenPaths.results_dir.iterdir() if path.is_dir()
    ):
        yield from sorted(parent_dir.glob('*/_subconfig.yaml'))


def _parent_names(raw_parents: list[str] | None) -> set[str] | None:
    if not raw_parents:
        return None
    return {resolve_experiment_name(value) for value in raw_parents}


def _yaml() -> YAML:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


def _read_yaml(path: Path, yaml: YAML) -> YamlMapping:
    with open(path) as file:
        raw = yaml.load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f'subconfig must be a YAML mapping: {path}')
    return cast(YamlMapping, raw)


def _ensure_mapping(parent: YamlMapping, key: str) -> YamlMapping:
    raw = parent.get(key)
    if raw is None:
        child: YamlMapping = {}
        parent[key] = child
        return child
    if not isinstance(raw, dict):
        raise ValueError(f'expected YAML mapping at key {key!r}')
    return cast(YamlMapping, raw)


def _write_yaml_atomic(path: Path, payload: YamlMapping, yaml: YAML) -> None:
    tmp_path = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with open(tmp_path, 'w') as file:
        yaml.dump(payload, file)
    os.replace(tmp_path, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Verify and repair subexperiment generation modes encoded by names like '
            'biased_q_list_f_simple_c_bge_m3.'
        )
    )
    parser.add_argument(
        '--parent',
        dest='parents',
        action='append',
        help='Parent experiment to repair. May be repeated. Defaults to all parents.',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually update _subconfig.yaml files. Defaults to dry-run.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
