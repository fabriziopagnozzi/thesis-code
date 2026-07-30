from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal, cast

from experiments.medical_dataset_gen.utils.global_utils import get_literals

type EmbeddingModelName = Literal[
    'BAAI/bge-m3',
    'Qwen/Qwen3-Embedding-0.6B',
    'Qwen/Qwen3-Embedding-4B',
    'Qwen/Qwen3-Embedding-8B',
    'jinaai/jina-embeddings-v5-text-small',
    'multi-qa-mpnet-base-cos-v1',
    'abhinand/MedEmbed-large-v0.1',
    'ncbi/MedCPT',
]

EMBEDDING_MODEL_NAMES = tuple[EmbeddingModelName, ...](get_literals(EmbeddingModelName))
EMBEDDING_CHILD_TOKENS: dict[EmbeddingModelName, str] = {
    'BAAI/bge-m3': 'bge_m3',
    'Qwen/Qwen3-Embedding-0.6B': 'qwen3_06',
    'Qwen/Qwen3-Embedding-4B': 'qwen3_4b',
    'Qwen/Qwen3-Embedding-8B': 'qwen3_8b',
    'jinaai/jina-embeddings-v5-text-small': 'jinaai',
    'multi-qa-mpnet-base-cos-v1': 'multi_mpnet',
    'abhinand/MedEmbed-large-v0.1': 'medembed_L',
    'ncbi/MedCPT': 'medcpt',
}
COMPACT_EMBEDDING_CHILD_TOKENS = frozenset(EMBEDDING_CHILD_TOKENS.values())


def embedding_child_token(model_name: str) -> str:
    if model_name in EMBEDDING_MODEL_NAMES:
        return EMBEDDING_CHILD_TOKENS[cast(EmbeddingModelName, model_name)]
    return _fallback_model_token(model_name)


def is_compact_embedding_child_token(value: str) -> bool:
    base_value = value.removesuffix('_allq')
    return base_value in COMPACT_EMBEDDING_CHILD_TOKENS


def _fallback_model_token(model_name: str) -> str:
    token = model_name.rsplit('/', 1)[-1]
    token = token.removesuffix('-v0.1').removesuffix('-base-cos-v1')
    token = token.replace('Embedding-', '')
    token = re.sub(r'[^A-Za-z0-9]+', '_', token).strip('_')
    token = re.sub(r'_+', '_', token)
    return token[:32] or 'embedding'


def resolve_experiment_dir_prefix(prefix: str, parent_dir: Path) -> Path:
    exact_dir = parent_dir / prefix
    if exact_dir.is_dir():
        return exact_dir

    matches = sorted(path for path in parent_dir.glob(f'{prefix}*') if path.is_dir())
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f'{prefix!r} is an ambiguous prefix in {parent_dir}, '
            f'found many matches: {[path.name for path in matches]}'
        )
    raise FileNotFoundError(f'no experiment directory prefixed {prefix!r} in {parent_dir}. ')


def load_experiment_aliases(results_dir: Path) -> dict[str, str]:
    aliases_path = results_dir / '_experiment_aliases.json'
    if not aliases_path.is_file():
        return {}
    raw = json.loads(aliases_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f'experiment aliases file must contain a JSON object: {aliases_path}')

    aliases: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f'experiment aliases must map strings to strings: {aliases_path}')
        aliases[key] = value
    return aliases


def resolve_experiment_alias(exp_name: str, results_dir: Path) -> str | None:
    alias_target = load_experiment_aliases(results_dir).get(exp_name)
    if alias_target is None:
        return None

    target_path = Path(alias_target)
    if target_path.is_absolute() or len(target_path.parts) > 2:
        raise ValueError(
            f'experiment alias target must be relative with at most one child: {alias_target!r}'
        )
    if not (results_dir / target_path).is_dir():
        raise FileNotFoundError(
            f'experiment alias {exp_name!r} points to missing directory {alias_target!r}'
        )
    return alias_target


def resolve_experiment_name(exp_name: str, results_dir: Path | None = None) -> str:
    from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths

    results_dir = results_dir or MedicalDatasetGenPaths.results_dir
    exp_path = Path(exp_name)
    if not exp_path.parts:
        raise ValueError('experiment name cannot be empty')
    if exp_path.is_absolute() or len(exp_path.parts) > 2:
        raise ValueError(
            f'experiment names are relative and support at most one child level: {exp_name!r}'
        )

    dir = results_dir / exp_path
    if dir.is_dir():
        return exp_name

    alias_target = resolve_experiment_alias(exp_name, results_dir)
    if alias_target is not None:
        return alias_target

    if len(exp_path.parts) == 2:
        parent_prefix, child_prefix = exp_path.parts
        parent_name = resolve_experiment_alias(parent_prefix, results_dir)
        if parent_name is None:
            parent_name = resolve_experiment_dir_prefix(parent_prefix, results_dir).name
        child_alias = resolve_experiment_alias(f'{parent_name}/{child_prefix}', results_dir)
        if child_alias is not None:
            return child_alias
        child_dir = resolve_experiment_dir_prefix(child_prefix, results_dir / parent_name)
        return f'{parent_name}/{child_dir.name}'

    matches = sorted(path for path in results_dir.glob(f'{exp_name}*') if path.is_dir())
    if len(matches) == 1:
        return matches[0].name
    elif len(matches) > 1:
        raise RuntimeError(
            f'{exp_name!r} is an ambiguous prefix, found many matches: {[m.name for m in matches]}'
        )
    else:
        raise FileNotFoundError(f'no experiment directory prefixed {exp_name!r} in {results_dir}. ')


def child_experiment_names(
    parent_exp_name: str,
    results_dir: Path | None = None,
) -> list[str]:
    from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths

    results_dir = results_dir or MedicalDatasetGenPaths.results_dir
    parent_name = resolve_experiment_name(parent_exp_name, results_dir=results_dir)
    parent_path = results_dir / parent_name
    if len(Path(parent_name).parts) != 1:
        return []
    return [
        f'{parent_name}/{child_path.name}'
        for child_path in sorted(parent_path.iterdir())
        if child_path.is_dir() and (child_path / '_subconfig.yaml').is_file()
    ]
