import argparse
import difflib
import io
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

MAX_ITEMS = 10
SCALAR_ONLY = True
EXCLUDE_FLOW_KEYS = {'distractors'}


def is_dir_to_include(path: Path) -> bool:
    return str(path).find('00_scrapped') == -1


def is_scalar(value) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def convert_short_lists(obj, parent_key=None):
    if isinstance(obj, CommentedSeq):
        for item in obj:
            convert_short_lists(item)

        if parent_key in EXCLUDE_FLOW_KEYS:
            return

        if len(obj) <= MAX_ITEMS and (not SCALAR_ONLY or all(is_scalar(item) for item in obj)):
            obj.fa.set_flow_style()

    elif isinstance(obj, dict):
        for key, value in obj.items():
            convert_short_lists(value, parent_key=key)


def make_yaml() -> YAML:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=4, sequence=6, offset=4)

    # Force explicit `null` instead of blank null values.
    def represent_none(representer, data):
        return representer.represent_scalar('tag:yaml.org,2002:null', 'null')

    yaml.representer.add_representer(type(None), represent_none)

    return yaml


def render_yaml(data) -> str:
    yaml = make_yaml()
    buffer = io.StringIO()
    yaml.dump(data, buffer)
    return buffer.getvalue()


def process_file(path: Path, dry_run: bool, make_backups: bool):
    original = path.read_text(encoding='utf-8')

    yaml = make_yaml()
    data = yaml.load(original)

    convert_short_lists(data)

    updated = render_yaml(data)

    if original == updated:
        return False

    if dry_run:
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=str(path),
            tofile=f'{path} converted',
        )
        print(''.join(diff))
    else:
        if make_backups:
            backup = path.with_suffix(path.suffix + '.bak')
            backup.write_text(original, encoding='utf-8')

        path.write_text(updated, encoding='utf-8')
        print(f'Updated {path}')

    return True


def find_config_files(root: Path):
    for subdir in root.iterdir():
        if subdir.is_dir() and is_dir_to_include(subdir):
            yield from subdir.rglob('*_config.yaml')


def main():
    parser = argparse.ArgumentParser(description='Convert short YAML block lists to inline arrays.')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print diffs instead of modifying files.',
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create .bak files when writing changes.',
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=Path.cwd(),
        help='Root directory to scan. Defaults to current directory.',
    )

    args = parser.parse_args()

    changed_count = 0
    checked_count = 0

    for config_file in find_config_files(args.root):
        checked_count += 1
        changed = process_file(
            config_file,
            dry_run=args.dry_run,
            make_backups=not args.no_backup,
        )
        if changed:
            changed_count += 1

    mode = 'Would update' if args.dry_run else 'Updated'
    print(f'{mode} {changed_count} of {checked_count} files.')


if __name__ == '__main__':
    main()
