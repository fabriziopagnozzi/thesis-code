import argparse
import csv
import json
import sys
from decimal import Decimal
from pathlib import Path

import ijson


def read_csv_head(path: Path, n: int) -> list[dict]:
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            rows.append(dict(row))
    return rows


def read_jsonl_head(path: Path, n: int) -> list[dict]:
    rows = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json_array_head(path: Path, n: int) -> list[dict]:
    def _find_array_prefix(path: Path) -> str:
        with open(path, 'rb') as f:
            for prefix, event, _ in ijson.parse(f):
                if event == 'start_array':
                    return f'{prefix}.item' if prefix else 'item'
        raise ValueError('No array found in JSON file')

    prefix = _find_array_prefix(path)
    rows = []
    with open(path, 'rb') as f:
        for i, item in enumerate(ijson.items(f, prefix)):
            if i >= n:
                break
            rows.append(item)
    return rows


def read_head(path: Path, n: int = 100) -> tuple[list[dict], str]:
    suffix = path.suffix.lower()
    if suffix == '.jsonl':
        return read_jsonl_head(path, n), 'json'
    elif suffix == '.json':
        return read_json_array_head(path, n), 'json'
    elif suffix == '.csv':
        return read_csv_head(path, n), 'csv'
    else:
        raise ValueError(
            f'Unsupported file type: {suffix!r} (expected .csv, .json, or .jsonl)'
        )


def _json_default(obj: object) -> object:
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def print_rows(rows: list[dict], fmt: str) -> None:
    if not rows:
        return
    if fmt == 'csv':
        writer = csv.DictWriter(sys.stdout, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    else:  # json
        sys.stdout.write(json.dumps(rows, ensure_ascii=True, indent=2, default=_json_default) + '\n')


def interactive_mode() -> None:
    print('Interactive mode. Syntax: <path> [rows]  |  q to quit', file=sys.stderr)
    while True:
        try:
            line = input('> ').strip()
        except EOFError, KeyboardInterrupt:
            print(file=sys.stderr)
            break
        if not line or line.startswith('#'):
            continue
        if line in ('q', 'quit', 'exit'):
            break
        parts = line.split()
        path = Path(parts[0]).expanduser().resolve()
        n = 100
        if len(parts) >= 2:
            try:
                n = int(parts[1])
            except ValueError:
                print(f'Error: invalid row count {parts[1]!r}', file=sys.stderr)
                continue
        try:
            rows, fmt = read_head(path, n)
            print_rows(rows, fmt)
        except Exception as e:
            print(f'Error: {e}', file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preview the first N rows of a CSV, JSON, or JSONL file.'
    )
    parser.add_argument(
        '-i', '--interactive', action='store_true', help='Interactive mode'
    )
    parser.add_argument('-p', '--path', type=Path, help='Path to the file')
    parser.add_argument(
        '-r', '--rows', type=int, default=100, help='Number of rows (default: 100)'
    )
    args = parser.parse_args()

    if args.interactive:
        if args.path is not None or args.rows != 100:
            parser.error(
                '--interactive (-i) is standalone and cannot'
                ' be combined with --path or --rows'
            )
        interactive_mode()
    else:
        if args.path is None:
            parser.error('--path is required')
        rows, fmt = read_head(args.path, args.rows)
        print_rows(rows, fmt)
