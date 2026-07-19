from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, TypedDict

import yaml

from experiments.medical_dataset_gen.utils.exp_naming import embedding_child_token
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths

type YamlMap = dict[str, object]

DEFAULT_EMBEDDING_MODEL = 'Qwen/Qwen3-Embedding-0.6B'
DEFAULT_BATCH_SIZE = 256


@dataclass(frozen=True)
class AddEmbeddingPlanItem:
    parent_name: str
    child_name: str
    experiment_name: str
    model_name: str
    child_dir: Path


@dataclass(frozen=True)
class Skip:
    parent_name: str
    reason: str


class AddEmbeddingPlanItemJson(TypedDict):
    parent_name: str
    child_name: str
    experiment_name: str
    model_name: str
    child_dir: str


class AddEmbeddingPlanJson(TypedDict):
    schema_version: int
    applied: bool
    results_dir: str
    model_name: str
    child_name: str
    planned: list[AddEmbeddingPlanItemJson]
    skipped: dict[str, str]


@dataclass(frozen=True)
class CliArgs:
    results_dir: Path
    experiments: tuple[str, ...]
    name: str | None
    emb_model: str
    child_name: str | None
    batch_size: int
    device: str
    include_scrapped: bool
    apply: bool
    dry_run: bool
    force: bool
    plan_output: Path | None


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    stdin_text = sys.stdin.read()
    if args.name is not None or stdin_text.strip():
        _run_stdin_subconfig_mode(args=args, subconfig_text=stdin_text)
        return

    child_name = args.child_name or embedding_child_token(args.emb_model)
    plan, skips = build_plan(
        results_dir=args.results_dir,
        requested_experiments=args.experiments,
        include_scrapped=args.include_scrapped,
        child_name=child_name,
        emb_model=args.emb_model,
    )
    if args.apply:
        apply_plan(
            plan,
            emb_model=args.emb_model,
            batch_size=args.batch_size,
            device=args.device,
        )

    payload = _plan_payload(
        results_dir=args.results_dir,
        child_name=child_name,
        emb_model=args.emb_model,
        plan=plan,
        skips=skips,
        applied=args.apply,
    )
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if args.plan_output is None:
        print(serialized)
    else:
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(serialized + '\n')
        print(f'wrote plan to {args.plan_output}')


def build_plan(
    *,
    results_dir: Path,
    requested_experiments: Sequence[str],
    include_scrapped: bool,
    child_name: str,
    emb_model: str,
) -> tuple[list[AddEmbeddingPlanItem], list[Skip]]:
    plan: list[AddEmbeddingPlanItem] = []
    skips: list[Skip] = []

    for parent_dir in _parent_dirs(
        results_dir=results_dir,
        requested_experiments=requested_experiments,
        include_scrapped=include_scrapped,
    ):
        parent_name = parent_dir.relative_to(results_dir).as_posix()
        if not (parent_dir / '_config.yaml').is_file():
            skips.append(Skip(parent_name=parent_name, reason='missing _config.yaml'))
            continue
        if _has_embedding_child(parent_dir=parent_dir, emb_model=emb_model):
            skips.append(
                Skip(
                    parent_name=parent_name,
                    reason=f'already has child for embedding model {emb_model!r}',
                )
            )
            continue
        child_dir = parent_dir / child_name
        if child_dir.exists():
            skips.append(Skip(parent_name=parent_name, reason=f'{child_name} already exists'))
            continue
        plan.append(
            AddEmbeddingPlanItem(
                parent_name=parent_name,
                child_name=child_name,
                experiment_name=f'{parent_name}/{child_name}',
                model_name=emb_model,
                child_dir=child_dir,
            )
        )

    return plan, skips


def apply_plan(
    plan: Sequence[AddEmbeddingPlanItem],
    *,
    emb_model: str,
    batch_size: int,
    device: str,
) -> None:
    for item in plan:
        item.child_dir.mkdir(parents=False, exist_ok=False)
        (item.child_dir / '_subconfig.yaml').write_text(
            yaml.safe_dump(
                _subconfig_payload(
                    emb_model=emb_model,
                    batch_size=batch_size,
                    device=device,
                ),
                sort_keys=False,
                allow_unicode=False,
            )
        )


def _parse_args(argv: Sequence[str] | None) -> CliArgs:
    parser = argparse.ArgumentParser(
        description='Add embedding or stdin-defined subexperiments to parent experiments.'
    )
    parser.add_argument(
        '--name',
        default=None,
        help='Generic mode: child directory name. Reads _subconfig.yaml content from stdin.',
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=MedicalDatasetGenPaths.results_dir,
        help='Results directory containing parent experiment folders.',
    )
    parser.add_argument(
        '--experiments',
        nargs='*',
        default=(),
        help='Optional parent experiment names to process. Defaults to all non-scrapped parents.',
    )
    parser.add_argument(
        '--emb-model',
        default=DEFAULT_EMBEDDING_MODEL,
        help='Embedding model name for the new child. Default: Qwen/Qwen3-Embedding-0.6B.',
    )
    parser.add_argument(
        '--child-name',
        default=None,
        help='Optional child directory name. Defaults to the compact token for --emb-model.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Embedding batch size written to _subconfig.yaml. Default: {DEFAULT_BATCH_SIZE}.',
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help='Embedding device written to _subconfig.yaml. Default: cuda.',
    )
    parser.add_argument(
        '--include-scrapped',
        action='store_true',
        help='Also process experiments under 00_scrapped.',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Create the planned child directories. Without this flag the script is a dry-run.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print planned writes without creating or modifying files.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Generic mode: overwrite existing <parent>/<name>/_subconfig.yaml files.',
    )
    parser.add_argument(
        '--plan-output',
        type=Path,
        default=None,
        help='Optional JSON path for the plan. Prints JSON to stdout when omitted.',
    )
    namespace = parser.parse_args(argv)
    if namespace.apply and namespace.dry_run:
        parser.error('--apply and --dry-run cannot be used together')
    return CliArgs(
        results_dir=namespace.results_dir,
        experiments=tuple(str(experiment) for experiment in namespace.experiments),
        name=str(namespace.name) if namespace.name is not None else None,
        emb_model=str(namespace.emb_model),
        child_name=str(namespace.child_name) if namespace.child_name is not None else None,
        batch_size=max(1, int(namespace.batch_size)),
        device=str(namespace.device),
        include_scrapped=bool(namespace.include_scrapped),
        apply=bool(namespace.apply),
        dry_run=bool(namespace.dry_run),
        force=bool(namespace.force),
        plan_output=namespace.plan_output,
    )


def _run_stdin_subconfig_mode(*, args: CliArgs, subconfig_text: str) -> None:
    if args.name is None:
        _die('Generic stdin mode requires --name.')
    _validate_child_name(args.name)
    subconfig = _parse_subconfig_from_stdin(subconfig_text)
    subconfig = _with_default_embedding_config(
        subconfig,
        emb_model=args.emb_model,
        batch_size=args.batch_size,
        device=args.device,
    )

    plan: list[AddEmbeddingPlanItem] = []
    skips: list[Skip] = []
    for parent_dir in _parent_dirs(
        results_dir=args.results_dir,
        requested_experiments=args.experiments,
        include_scrapped=args.include_scrapped,
    ):
        parent_name = parent_dir.relative_to(args.results_dir).as_posix()
        if not (parent_dir / '_config.yaml').is_file():
            skips.append(Skip(parent_name=parent_name, reason='missing _config.yaml'))
            continue
        child_dir = parent_dir / args.name
        subconfig_path = child_dir / '_subconfig.yaml'
        if subconfig_path.exists() and not args.force:
            skips.append(
                Skip(parent_name=parent_name, reason=f'{args.name}/_subconfig.yaml exists')
            )
            continue
        if child_dir.exists() and not child_dir.is_dir():
            skips.append(
                Skip(parent_name=parent_name, reason=f'{args.name} exists and is not a directory')
            )
            continue
        plan.append(
            AddEmbeddingPlanItem(
                parent_name=parent_name,
                child_name=args.name,
                experiment_name=f'{parent_name}/{args.name}',
                model_name='stdin',
                child_dir=child_dir,
            )
        )

    if args.apply:
        text_to_write = yaml.safe_dump(subconfig, sort_keys=False, allow_unicode=False)
        for item in plan:
            item.child_dir.mkdir(parents=True, exist_ok=True)
            (item.child_dir / '_subconfig.yaml').write_text(text_to_write)

    payload = {
        'schema_version': 1,
        'mode': 'stdin_subconfig',
        'applied': args.apply,
        'results_dir': str(args.results_dir),
        'child_name': args.name,
        'subconfig_keys': sorted(subconfig),
        'planned': [
            {
                'parent_name': item.parent_name,
                'child_name': item.child_name,
                'experiment_name': item.experiment_name,
                'child_dir': str(item.child_dir),
            }
            for item in plan
        ],
        'skipped': {skip.parent_name: skip.reason for skip in skips},
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if args.plan_output is None:
        print(serialized)
    else:
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(serialized + '\n')
        print(f'wrote plan to {args.plan_output}')


def _validate_child_name(name: str) -> None:
    path = Path(name)
    if not name or path.is_absolute() or len(path.parts) != 1 or name in {'.', '..'}:
        _die(f'--name must be a single child directory name, got {name!r}.')
    if name.startswith('_'):
        _die(
            f'--name must not start with "_"; reserved result directories use that prefix: {name!r}.'
        )


def _parse_subconfig_from_stdin(text: str) -> YamlMap:
    if not text.strip():
        _die('No subconfig YAML received on stdin.')
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        _die(f'Invalid YAML received on stdin: {exc}')
    if not isinstance(loaded, Mapping):
        _die('Subconfig YAML must be a mapping at the top level.')
    subconfig = {str(key): value for key, value in loaded.items()}
    generation = _mapping(subconfig.get('generation'))
    if generation is not None and 'chunk_pools' in generation:
        _die(
            'Subconfigs must not override generation.chunk_pools; child experiments must keep '
            'the parent generated dataset distribution unchanged.'
        )
    return subconfig


def _with_default_embedding_config(
    subconfig: YamlMap,
    *,
    emb_model: str,
    batch_size: int,
    device: str,
) -> YamlMap:
    default_embeddings = _subconfig_payload(
        emb_model=emb_model,
        batch_size=batch_size,
        device=device,
    )['embeddings']
    embeddings = _mapping(subconfig.get('embeddings'))
    merged = dict(subconfig)
    if embeddings is None:
        merged['embeddings'] = default_embeddings
        return merged
    merged['embeddings'] = {
        **default_embeddings,  # type: ignore
        **embeddings,
    }
    return merged


def _parent_dirs(
    *,
    results_dir: Path,
    requested_experiments: Sequence[str],
    include_scrapped: bool,
) -> list[Path]:
    if requested_experiments:
        return [results_dir / experiment for experiment in requested_experiments]

    parent_dirs = [
        path
        for path in sorted(results_dir.iterdir())
        if path.is_dir() and not path.name.startswith('_')
    ]
    if include_scrapped:
        scrapped_dir = results_dir / '00_scrapped'
        if scrapped_dir.is_dir():
            parent_dirs.extend(
                path
                for path in sorted(scrapped_dir.iterdir())
                if path.is_dir() and not path.name.startswith('_')
            )
        return parent_dirs

    return [path for path in parent_dirs if path.name != '00_scrapped']


def _has_embedding_child(*, parent_dir: Path, emb_model: str) -> bool:
    for subconfig_path in sorted(parent_dir.glob('*/_subconfig.yaml')):
        raw = _read_yaml_mapping(subconfig_path)
        embeddings = _mapping(raw.get('embeddings'))
        if embeddings is None:
            continue
        if embeddings.get('model_name') == emb_model:
            return True
    return False


def _subconfig_payload(
    *,
    emb_model: str,
    batch_size: int,
    device: str,
) -> YamlMap:
    return {
        'embeddings': {
            'model_name': emb_model,
            'batch_size': batch_size,
            'device': device,
            'query_prompt': None,
            'normalize': True,
        }
    }


def _read_yaml_mapping(path: Path) -> YamlMap:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f'expected YAML mapping at {path}')
    return {str(key): value for key, value in raw.items()}


def _mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}


def _plan_payload(
    *,
    results_dir: Path,
    child_name: str,
    emb_model: str,
    plan: Sequence[AddEmbeddingPlanItem],
    skips: Sequence[Skip],
    applied: bool,
) -> AddEmbeddingPlanJson:
    return {
        'schema_version': 1,
        'applied': applied,
        'results_dir': str(results_dir),
        'model_name': emb_model,
        'child_name': child_name,
        'planned': [
            {
                'parent_name': item.parent_name,
                'child_name': item.child_name,
                'experiment_name': item.experiment_name,
                'model_name': item.model_name,
                'child_dir': str(item.child_dir),
            }
            for item in plan
        ],
        'skipped': {skip.parent_name: skip.reason for skip in skips},
    }


def _die(message: str) -> NoReturn:
    print(f'error: {message}', file=sys.stderr)
    raise SystemExit(1)


if __name__ == '__main__':
    main()
