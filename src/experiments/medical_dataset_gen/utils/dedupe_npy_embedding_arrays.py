from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

import numpy as np
from ruamel.yaml import YAML

from experiments.medical_dataset_gen.dataset_generation.deterministic_caches import (
    chunk_embedding_signature,
)
from experiments.medical_dataset_gen.schemas.global_config_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.exp_naming import resolve_experiment_name
from experiments.medical_dataset_gen.utils.global_utils import (
    EmbeddingArtifactName,
    MedicalDatasetGenPaths,
    YamlMapping,
    load_config,
)
from helpers.embedder import MODEL_PROFILES

type CompactScope = Literal['all', 'chunk', 'query']
type ArtifactGroupKind = Literal['chunk', 'query']
type QuerySurfaceGroup = Literal['biased', 'unbiased']
type QueryFocusToken = Literal['list', 'natural']
type ChunkModeToken = Literal['simple', 'hardened']

QUERY_EMBEDDING_SIGNATURE_VERSION = 1
MODEL_TOKEN_PREFERENCE = {
    'bge_m3': 0,
    'qwen3_06': 1,
    'qwen3_06B': 2,
    'multi_mpnet': 3,
    'medembed_L': 4,
}
_CHILD_NAME_RE = re.compile(
    r'^(?P<query_surface>biased|unbiased)_q_'
    r'(?P<focus_mode>list|natural)_f_'
    r'(?P<chunk_mode>simple|hardened)_c_'
    r'(?P<model_token>.+)$'
)

CHUNK_ARTIFACTS: tuple[EmbeddingArtifactName, ...] = ('chunk_vectors', 'chunk_ids')
QUERY_ARTIFACTS: tuple[EmbeddingArtifactName, ...] = ('query_vectors', 'query_ids')
ARTIFACT_FILENAMES: dict[EmbeddingArtifactName, str] = {
    'chunk_vectors': 'embeddings_chunk_vectors.npy',
    'query_vectors': 'embeddings_query_vectors.npy',
    'chunk_ids': 'embeddings_chunk_ids.npy',
    'query_ids': 'embeddings_query_ids.npy',
    'metadata': 'embeddings_metadata.json',
}


class QueryEmbeddingSignaturePayload(TypedDict):
    signature_version: int
    model_name: str
    profile_mode: str
    query_prompt: str | None
    query_prompt_name: str | None
    normalize: bool


@dataclass(frozen=True, slots=True)
class ChildNameParts:
    query_surface: QuerySurfaceGroup
    focus_mode: QueryFocusToken
    chunk_mode: ChunkModeToken
    model_token: str


@dataclass(frozen=True, slots=True)
class NpyFileInfo:
    path: Path
    exists: bool
    size_bytes: int
    dtype: str | None
    shape: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class ChildEmbeddingArtifacts:
    exp_name: str
    parent_name: str
    child_name: str
    experiment_dir: Path
    subconfig_path: Path
    cfg: ExperimentCfg
    files: dict[EmbeddingArtifactName, NpyFileInfo]


@dataclass(frozen=True, slots=True)
class ArtifactGroupPlan:
    kind: ArtifactGroupKind
    artifacts: tuple[EmbeddingArtifactName, ...]
    canonical: ChildEmbeddingArtifacts
    duplicates: tuple[ChildEmbeddingArtifacts, ...]
    key: tuple[object, ...]

    @property
    def deletable_bytes(self) -> int:
        total = 0
        for duplicate in self.duplicates:
            for artifact in self.artifacts:
                info = duplicate.files[artifact]
                if info.exists:
                    total += info.size_bytes
        return total


def main() -> None:
    args = _parse_args()
    parents = _parent_names(args.parents)
    children = _load_children(parents=parents)
    plans = _build_plans(
        children,
        scope=cast(CompactScope, args.scope),
        include_llm=bool(args.include_llm),
    )
    verify_byte_identical = bool(args.require_byte_identical) or (
        bool(args.apply) and not bool(args.trust_config)
    )
    if verify_byte_identical:
        print('[compact] verifying byte-identical artifact files before rewiring/deletion')
        plans = _filter_byte_identical_plans(plans)

    if args.canonical_report is not None:
        report_path = Path(str(args.canonical_report))
        _write_canonical_report(plans, report_path)
        print(f'[compact] canonical report -> {report_path}')

    _print_plan(
        plans,
        apply=bool(args.apply),
        no_delete=bool(args.no_delete),
        summary_only=bool(args.summary_only),
    )
    if not args.apply:
        print('[compact] dry run only. Re-run with --apply to update subconfigs and delete files.')
        return

    repaired_subconfigs, removed_invalid_overrides = _remove_invalid_existing_overrides(
        children,
        scope=cast(CompactScope, args.scope),
        include_llm=bool(args.include_llm),
    )
    changed_subconfigs = repaired_subconfigs | _apply_overrides(plans)
    deleted_files, deleted_bytes = (0, 0) if args.no_delete else _delete_duplicate_files(plans)
    print(
        '[compact] applied: '
        f'updated_subconfigs={len(changed_subconfigs):,}, '
        f'removed_invalid_overrides={removed_invalid_overrides:,}, '
        f'deleted_files={deleted_files:,}, freed={_format_bytes(deleted_bytes)}'
    )


def _build_plans(
    children: list[ChildEmbeddingArtifacts],
    *,
    scope: CompactScope,
    include_llm: bool,
) -> list[ArtifactGroupPlan]:
    plans: list[ArtifactGroupPlan] = []
    if scope in {'all', 'chunk'}:
        plans.extend(
            _plans_for_kind(
                children,
                kind='chunk',
                artifacts=CHUNK_ARTIFACTS,
                include_llm=include_llm,
            )
        )
    if scope in {'all', 'query'}:
        plans.extend(
            _plans_for_kind(
                children,
                kind='query',
                artifacts=QUERY_ARTIFACTS,
                include_llm=include_llm,
            )
        )
    return plans


def _plans_for_kind(
    children: list[ChildEmbeddingArtifacts],
    *,
    kind: ArtifactGroupKind,
    artifacts: tuple[EmbeddingArtifactName, ...],
    include_llm: bool,
) -> list[ArtifactGroupPlan]:
    groups: dict[tuple[object, ...], list[ChildEmbeddingArtifacts]] = defaultdict(list)
    for child in children:
        if kind == 'chunk' and _skip_llm_chunks(child.cfg, include_llm=include_llm):
            continue
        if kind == 'query' and _skip_llm_queries(child.cfg, include_llm=include_llm):
            continue
        key = _group_key(child, kind=kind)
        groups[key].append(child)

    plans: list[ArtifactGroupPlan] = []
    for key, group_children in sorted(groups.items(), key=lambda item: str(item[0])):
        compatible = sorted(group_children, key=lambda item: item.exp_name)
        canonical_candidates = [
            child for child in compatible if _has_all_local_files(child, artifacts)
        ]
        if not canonical_candidates:
            continue
        canonical = min(
            canonical_candidates,
            key=lambda child: _canonical_sort_key(child, kind=kind),
        )
        duplicates = tuple(
            child
            for child in compatible
            if child != canonical and _files_compatible_with_canonical(child, canonical, artifacts)
        )
        if not duplicates:
            continue
        plans.append(
            ArtifactGroupPlan(
                kind=kind,
                artifacts=artifacts,
                canonical=canonical,
                duplicates=duplicates,
                key=key,
            )
        )
    return plans


def _group_key(
    child: ChildEmbeddingArtifacts,
    *,
    kind: ArtifactGroupKind,
) -> tuple[object, ...]:
    cfg = child.cfg
    if kind == 'chunk':
        return (
            child.parent_name,
            cfg.generation.chunk_text_style,
            chunk_embedding_signature(cfg),
        )
    return (
        child.parent_name,
        _query_surface_group(child),
        cfg.generation.focus_mode,
        cfg.generation.query_structure,
        _query_embedding_signature(cfg),
    )


def _query_embedding_signature(cfg: ExperimentCfg) -> str:
    return _hash_json(_query_embedding_signature_payload(cfg))


def _query_embedding_signature_payload(cfg: ExperimentCfg) -> QueryEmbeddingSignaturePayload:
    profile = MODEL_PROFILES[cfg.embeddings.model_name]
    return {
        'signature_version': QUERY_EMBEDDING_SIGNATURE_VERSION,
        'model_name': cfg.embeddings.model_name,
        'profile_mode': profile.mode,
        'query_prompt': (
            cfg.embeddings.query_prompt
            if cfg.embeddings.query_prompt is not None
            else profile.query_prompt
        ),
        'query_prompt_name': profile.query_prompt_name
        if cfg.embeddings.query_prompt is None
        else None,
        'normalize': cfg.embeddings.normalize,
    }


def _hash_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _shape_key(info: NpyFileInfo) -> tuple[str | None, tuple[int, ...] | None]:
    return info.dtype, info.shape


def _query_surface_group(child: ChildEmbeddingArtifacts) -> QuerySurfaceGroup | str:
    parts = _child_name_parts(child.child_name)
    if parts is not None:
        return parts.query_surface
    if child.child_name.startswith('biased_q_'):
        return 'biased'
    if child.child_name.startswith('unbiased_q_'):
        return 'unbiased'
    return f'child:{child.child_name}'


def _canonical_sort_key(
    child: ChildEmbeddingArtifacts,
    *,
    kind: ArtifactGroupKind,
) -> tuple[int, int, int, int, str]:
    parts = _child_name_parts(child.child_name)
    if parts is None:
        return (1, 1, 1, 999, child.exp_name)

    if kind == 'chunk':
        expected_chunk_mode = (
            'simple' if child.cfg.generation.chunk_text_style == 'ontology_explicit' else 'hardened'
        )
        return (
            0,
            0 if parts.query_surface == 'biased' else 1,
            _focus_preference(parts.focus_mode),
            0 if parts.chunk_mode == expected_chunk_mode else 1,
            f'{_model_token_preference(parts.model_token):03d}:{child.exp_name}',
        )

    # Query embeddings do not depend on chunk text style. Prefer the simple
    # chunk-mode subexperiment as the query-vector source for each query surface/focus.
    return (
        0,
        0 if parts.chunk_mode == 'simple' else 1,
        _model_token_preference(parts.model_token),
        0,
        child.exp_name,
    )


def _child_name_parts(child_name: str) -> ChildNameParts | None:
    match = _CHILD_NAME_RE.fullmatch(child_name)
    if match is None:
        return None
    return ChildNameParts(
        query_surface=cast(QuerySurfaceGroup, match.group('query_surface')),
        focus_mode=cast(QueryFocusToken, match.group('focus_mode')),
        chunk_mode=cast(ChunkModeToken, match.group('chunk_mode')),
        model_token=match.group('model_token'),
    )


def _focus_preference(focus_mode: QueryFocusToken) -> int:
    return 0 if focus_mode == 'list' else 1


def _model_token_preference(model_token: str) -> int:
    return MODEL_TOKEN_PREFERENCE.get(model_token, 100)


def _files_compatible_with_canonical(
    child: ChildEmbeddingArtifacts,
    canonical: ChildEmbeddingArtifacts,
    artifacts: tuple[EmbeddingArtifactName, ...],
) -> bool:
    for artifact in artifacts:
        info = child.files[artifact]
        if not info.exists:
            continue
        if _shape_key(info) != _shape_key(canonical.files[artifact]):
            return False
    return True


def _has_all_local_files(
    child: ChildEmbeddingArtifacts,
    artifacts: tuple[EmbeddingArtifactName, ...],
) -> bool:
    return all(child.files[artifact].exists for artifact in artifacts)


def _skip_llm_chunks(cfg: ExperimentCfg, *, include_llm: bool) -> bool:
    return not include_llm and (
        cfg.generation.llm_config.use_llm_chunk_generation
        or cfg.generation.llm_config.use_llm_chunk_rewriting
    )


def _skip_llm_queries(cfg: ExperimentCfg, *, include_llm: bool) -> bool:
    return not include_llm and cfg.generation.llm_config.use_llm_query_paraphrase


def _filter_byte_identical_plans(plans: list[ArtifactGroupPlan]) -> list[ArtifactGroupPlan]:
    digest_cache: dict[Path, str] = {}
    filtered: list[ArtifactGroupPlan] = []
    for plan in plans:
        canonical_hashes: dict[EmbeddingArtifactName, str] = {}
        matching_duplicates: list[ChildEmbeddingArtifacts] = []
        for duplicate in plan.duplicates:
            if _duplicate_files_byte_identical_or_absent(
                duplicate,
                canonical=plan.canonical,
                canonical_hashes=canonical_hashes,
                artifacts=plan.artifacts,
                digest_cache=digest_cache,
            ):
                matching_duplicates.append(duplicate)
        if matching_duplicates:
            filtered.append(
                ArtifactGroupPlan(
                    kind=plan.kind,
                    artifacts=plan.artifacts,
                    canonical=plan.canonical,
                    duplicates=tuple(matching_duplicates),
                    key=plan.key,
                )
            )
    return filtered


def _duplicate_files_byte_identical_or_absent(
    duplicate: ChildEmbeddingArtifacts,
    *,
    canonical: ChildEmbeddingArtifacts,
    canonical_hashes: dict[EmbeddingArtifactName, str],
    artifacts: tuple[EmbeddingArtifactName, ...],
    digest_cache: dict[Path, str],
) -> bool:
    for artifact in artifacts:
        info = duplicate.files[artifact]
        if not info.exists:
            continue
        canonical_hash = canonical_hashes.setdefault(
            artifact,
            _file_sha256(canonical.files[artifact].path, digest_cache),
        )
        if _file_sha256(info.path, digest_cache) != canonical_hash:
            return False
    return True


def _apply_overrides(plans: list[ArtifactGroupPlan]) -> set[Path]:
    overrides_by_subconfig: dict[Path, dict[EmbeddingArtifactName, str]] = defaultdict(dict)
    for plan in plans:
        canonical_dir = str(plan.canonical.experiment_dir)
        for duplicate in plan.duplicates:
            for artifact in plan.artifacts:
                overrides_by_subconfig[duplicate.subconfig_path][artifact] = canonical_dir

    changed: set[Path] = set()
    yaml = _yaml()
    for subconfig_path, overrides in sorted(overrides_by_subconfig.items()):
        raw = _read_yaml(subconfig_path, yaml)
        global_section = _ensure_mapping(raw, 'global')
        result_dir_overrides = _ensure_mapping(global_section, 'result_dir_overrides')
        did_change = False
        for artifact, target_dir in sorted(overrides.items()):
            if result_dir_overrides.get(artifact) != target_dir:
                result_dir_overrides[artifact] = target_dir
                did_change = True
        if did_change:
            _write_yaml_atomic(subconfig_path, raw, yaml)
            changed.add(subconfig_path)
    return changed


def _remove_invalid_existing_overrides(
    children: list[ChildEmbeddingArtifacts],
    *,
    scope: CompactScope,
    include_llm: bool,
) -> tuple[set[Path], int]:
    children_by_dir = {_normalized_path(child.experiment_dir): child for child in children}
    changed: set[Path] = set()
    removed = 0
    yaml = _yaml()
    for child in children:
        raw = _read_yaml(child.subconfig_path, yaml)
        global_section = raw.get('global')
        if not isinstance(global_section, dict):
            continue
        result_dir_overrides = global_section.get('result_dir_overrides')
        if not isinstance(result_dir_overrides, dict):
            continue
        result_overrides = cast(YamlMapping, result_dir_overrides)
        did_change = False
        for kind, artifacts in _scoped_artifact_groups(scope):
            if kind == 'chunk' and _skip_llm_chunks(child.cfg, include_llm=include_llm):
                continue
            if kind == 'query' and _skip_llm_queries(child.cfg, include_llm=include_llm):
                continue
            for artifact in artifacts:
                target_raw = result_overrides.get(artifact)
                if target_raw is None:
                    continue
                target_child = children_by_dir.get(_normalized_path(Path(str(target_raw))))
                if target_child is None or _group_key(child, kind=kind) != _group_key(
                    target_child,
                    kind=kind,
                ):
                    del result_overrides[artifact]
                    removed += 1
                    did_change = True
        if did_change and not result_overrides:
            del global_section['result_dir_overrides']
        if did_change:
            _write_yaml_atomic(child.subconfig_path, raw, yaml)
            changed.add(child.subconfig_path)
    return changed, removed


def _scoped_artifact_groups(
    scope: CompactScope,
) -> Iterable[tuple[ArtifactGroupKind, tuple[EmbeddingArtifactName, ...]]]:
    if scope in {'all', 'chunk'}:
        yield 'chunk', CHUNK_ARTIFACTS
    if scope in {'all', 'query'}:
        yield 'query', QUERY_ARTIFACTS


def _normalized_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _delete_duplicate_files(plans: list[ArtifactGroupPlan]) -> tuple[int, int]:
    deleted_files = 0
    deleted_bytes = 0
    for plan in plans:
        canonical_paths = {plan.canonical.files[artifact].path for artifact in plan.artifacts}
        for duplicate in plan.duplicates:
            for artifact in plan.artifacts:
                info = duplicate.files[artifact]
                if not info.exists or info.path in canonical_paths:
                    continue
                deleted_bytes += info.size_bytes
                info.path.unlink()
                deleted_files += 1
    return deleted_files, deleted_bytes


def _load_children(*, parents: set[str] | None) -> list[ChildEmbeddingArtifacts]:
    result: list[ChildEmbeddingArtifacts] = []
    for subconfig_path in _iter_subconfig_paths(parents=parents):
        child_dir = subconfig_path.parent
        parent_dir = child_dir.parent
        try:
            exp_name = str(child_dir.relative_to(MedicalDatasetGenPaths.results_dir))
        except ValueError:
            continue
        cfg = load_config(exp_name)
        result.append(
            ChildEmbeddingArtifacts(
                exp_name=exp_name,
                parent_name=parent_dir.name,
                child_name=child_dir.name,
                experiment_dir=child_dir,
                subconfig_path=subconfig_path,
                cfg=cfg,
                files=_local_embedding_files(child_dir),
            )
        )
    return sorted(result, key=lambda item: item.exp_name)


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


def _local_embedding_files(child_dir: Path) -> dict[EmbeddingArtifactName, NpyFileInfo]:
    return {
        artifact: _npy_info(child_dir / filename)
        for artifact, filename in ARTIFACT_FILENAMES.items()
        if artifact != 'metadata'
    }


def _npy_info(path: Path) -> NpyFileInfo:
    if not path.exists():
        return NpyFileInfo(
            path=path,
            exists=False,
            size_bytes=0,
            dtype=None,
            shape=None,
        )
    array = np.load(path, mmap_mode='r')
    return NpyFileInfo(
        path=path,
        exists=True,
        size_bytes=path.stat().st_size,
        dtype=str(array.dtype),
        shape=tuple(int(value) for value in array.shape),
    )


def _print_plan(
    plans: list[ArtifactGroupPlan],
    *,
    apply: bool,
    no_delete: bool,
    summary_only: bool,
) -> None:
    total_bytes = sum(plan.deletable_bytes for plan in plans)
    mode = 'apply' if apply else 'dry-run'
    delete_note = 'rewire only' if no_delete else 'rewire and delete duplicates'
    print(
        f'[compact] mode={mode}, action={delete_note}, '
        f'groups={len(plans):,}, potential_free={_format_bytes(total_bytes)}'
    )
    if summary_only:
        return
    for plan in plans:
        print(
            f'[{plan.kind}] canonical={plan.canonical.exp_name} '
            f'duplicates={len(plan.duplicates):,} free={_format_bytes(plan.deletable_bytes)} '
            f'key={_plan_key_label(plan)}'
        )
        for duplicate in plan.duplicates:
            print(f'  -> {duplicate.exp_name}')


def _write_canonical_report(plans: list[ArtifactGroupPlan], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        'kind\tcanonical\tduplicates\tdeletable_bytes\tkey',
        *[
            (
                f'{plan.kind}\t{plan.canonical.exp_name}\t{len(plan.duplicates)}\t'
                f'{plan.deletable_bytes}\t{_plan_key_label(plan)}'
            )
            for plan in plans
        ],
    ]
    tmp_path = report_path.with_name(f'.{report_path.name}.{os.getpid()}.tmp')
    tmp_path.write_text('\n'.join(lines) + '\n')
    os.replace(tmp_path, report_path)


def _plan_key_label(plan: ArtifactGroupPlan) -> str:
    if plan.kind == 'chunk':
        parent, chunk_text_style, signature = plan.key
        return (
            f'parent={parent}, chunk_text_style={chunk_text_style}, signature={str(signature)[:8]}'
        )
    parent, query_surface_group, focus_mode, query_structure, signature = plan.key
    return (
        f'parent={parent}, query_surface={query_surface_group}, focus_mode={focus_mode}, '
        f'query_structure={query_structure}, '
        f'signature={str(signature)[:8]}'
    )


def _parent_names(raw_parents: list[str] | None) -> set[str] | None:
    if not raw_parents:
        return None
    return {resolve_experiment_name(value) for value in raw_parents}


def _file_sha256(path: Path, digest_cache: dict[Path, str]) -> str:
    cached = digest_cache.get(path)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        while chunk := file.read(16 * 1024 * 1024):
            digest.update(chunk)
    value = digest.hexdigest()
    digest_cache[path] = value
    return value


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
            'Rewrite legacy subexperiment embedding .npy artifacts to shared canonical '
            'result_dir_overrides and optionally delete duplicate local arrays.'
        )
    )
    parser.add_argument(
        '--parent',
        dest='parents',
        action='append',
        help='Parent experiment to compact. May be repeated. Defaults to all parents.',
    )
    parser.add_argument('--scope', choices=['all', 'chunk', 'query'], default='all')
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually update _subconfig.yaml files and delete duplicate arrays.',
    )
    parser.add_argument(
        '--no-delete',
        action='store_true',
        help='Only write result_dir_overrides; keep duplicate local .npy files.',
    )
    parser.add_argument(
        '--include-llm',
        action='store_true',
        help='Allow grouping LLM-generated/re-written/paraphrased artifacts by config.',
    )
    parser.add_argument(
        '--require-byte-identical',
        action='store_true',
        help='In dry-run mode, hash duplicate .npy files and only report byte-identical copies.',
    )
    parser.add_argument(
        '--trust-config',
        action='store_true',
        help=(
            'With --apply, skip byte-identity checks and trust the effective config grouping. '
            'By default, --apply verifies byte identity before rewiring/deleting.'
        ),
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Print only the aggregate plan line instead of every canonical/duplicate group.',
    )
    parser.add_argument(
        '--canonical-report',
        help='Write a tab-separated canonical group report to this path.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
