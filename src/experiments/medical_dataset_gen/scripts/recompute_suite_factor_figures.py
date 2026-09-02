"""Selectively rebuild Results suite figures from an existing report."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from experiments.medical_dataset_gen.reports.suite_analysis import (
    RESULTS_SUITE_FIGURE_STEMS,
    write_suite_factor_figures,
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

    for path in written:
        print(f'[suite_figures] wrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
