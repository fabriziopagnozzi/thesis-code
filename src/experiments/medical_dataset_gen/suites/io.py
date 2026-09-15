"""Deterministic and traversal-safe suite I/O helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

import yaml


def read_yaml_mapping(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f'missing YAML file: {path}')
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f'YAML document must be a mapping: {path}')
    return cast(dict[str, object], raw)


def safe_relative(root: Path, raw_path: str) -> Path:
    path = (root / raw_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f'path escapes suite root: {raw_path!r}') from exc
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + '\n'


def write_json(path: Path, value: Mapping[str, object]) -> None:
    write_text_atomically(path, json_text(value))


def write_text_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=path.parent,
        prefix=f'.{path.name}.',
        suffix='.tmp',
        delete=False,
    ) as temporary:
        temporary.write(text)
        temporary_path = Path(temporary.name)
    try:
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
