from __future__ import annotations

import hashlib


def stable_int(*parts: object) -> int:
    raw = '|'.join(str(part) for part in parts)
    return int(hashlib.sha256(raw.encode()).hexdigest()[:16], 16)


def stable_id(prefix: str, *parts: object) -> str:
    raw = '|'.join(str(part) for part in parts)
    return f'{prefix}_{hashlib.sha256(raw.encode()).hexdigest()[:16]}'


def stable_seed(*parts: object) -> int:
    return stable_int(*parts)


def stable_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
