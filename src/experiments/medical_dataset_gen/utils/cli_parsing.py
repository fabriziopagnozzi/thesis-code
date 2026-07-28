from __future__ import annotations

from collections.abc import Collection, Sequence


def parse_comma_separated_names(
    *,
    raw_value: str | None,
    valid_names: Collection[str],
    option_name: str,
) -> list[str] | None:
    if raw_value is None:
        return None

    names = [part.strip() for part in raw_value.split(',')]
    if not names or any(not name for name in names):
        raise ValueError(f'{option_name} must be a comma-separated list of names')

    invalid_names = [name for name in names if name not in valid_names]
    if invalid_names:
        raise ValueError(
            f'{option_name} contains invalid name(s): {", ".join(invalid_names)}. '
            f'Valid names: {", ".join(sorted(valid_names))}'
        )

    duplicates = duplicate_names(names)
    if duplicates:
        raise ValueError(f'{option_name} contains duplicate name(s): {", ".join(duplicates)}')

    return names


def duplicate_names(names: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    return duplicates
