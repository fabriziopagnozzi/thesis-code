"""Selectively rebuild Results suite figures from an existing report."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from experiments.medical_dataset_gen.reports.suite_analysis import (
    RESULTS_SUITE_FIGURE_STEMS,
    write_suite_factor_figures,
)

_COMBINED_INTERACTION_STEM = 'stressor_interactions_by_objective'
_OBSOLETE_INTERACTION_STEMS = (
    'dominance_background_interaction_by_objective',
    'sparse_near_miss_interaction_by_objective',
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Rebuild the six-metric suite figures directly from '
            'data/suite_matched_contrasts.csv in an existing report.'
        )
    )
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument(
        '--stem',
        action='append',
        choices=RESULTS_SUITE_FIGURE_STEMS,
        help='Figure stem to rebuild; repeat as needed (defaults to every Results figure).',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_dir = args.report_dir.expanduser().resolve()
    contrast_path = report_dir / 'data' / 'suite_matched_contrasts.csv'
    if not contrast_path.is_file():
        raise FileNotFoundError(f'missing report input: {contrast_path}')

    with contrast_path.open(newline='', encoding='utf-8') as handle:
        contrast_rows = cast(list[dict[str, object]], list(csv.DictReader(handle)))

    requested_stems = tuple(args.stem or RESULTS_SUITE_FIGURE_STEMS)
    written = write_suite_factor_figures(
        output_dir=report_dir,
        contrast_rows=contrast_rows,
        stems=requested_stems,
    )
    written_stems = {path.stem for path in written}
    missing = sorted(set(requested_stems) - written_stems)
    if missing:
        raise RuntimeError(f'no matching report rows for figure(s): {", ".join(missing)}')

    _refresh_combined_interaction_references(report_dir=report_dir, written=written)

    for path in written:
        print(f'[suite_figures] wrote {path}')
    return 0


def _refresh_combined_interaction_references(
    *, report_dir: Path, written: Sequence[Path]
) -> None:
    """Replace obsolete interaction paths in report indexes after consolidation."""
    combined_paths = [
        path.relative_to(report_dir).as_posix()
        for path in written
        if path.stem == _COMBINED_INTERACTION_STEM
    ]
    if not combined_paths:
        return
    obsolete_paths = {
        f'figures/suite/{stem}.{suffix}'
        for stem in _OBSOLETE_INTERACTION_STEMS
        for suffix in ('png', 'pdf')
    }

    manifest_path = report_dir / 'manifest.json'
    if manifest_path.is_file():
        try:
            manifest: object = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = None
        if isinstance(manifest, dict):
            for key in ('figures', 'files'):
                existing = manifest.get(key)
                if not isinstance(existing, list):
                    continue
                retained = {
                    value
                    for value in existing
                    if isinstance(value, str)
                    and value not in obsolete_paths
                    and value not in combined_paths
                }
                manifest[key] = sorted([*retained, *combined_paths])
            manifest['figures_refreshed_at_utc'] = datetime.now(UTC).isoformat()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')

    markdown_path = report_dir / 'txt_report.md'
    if markdown_path.is_file():
        old_bullets = {f'- `{path}`' for path in obsolete_paths}
        new_bullets = [f'- `{path}`' for path in combined_paths]
        output_lines: list[str] = []
        inserted = False
        for line in markdown_path.read_text().splitlines():
            if line in old_bullets or line in new_bullets:
                if not inserted:
                    output_lines.extend(new_bullets)
                    inserted = True
                continue
            output_lines.append(line)
        markdown_path.write_text('\n'.join(output_lines) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
