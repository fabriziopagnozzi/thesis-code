"""Regenerate only the two embedding-model overview figures.

The script deliberately reads the existing report CSVs.  It never loads suite
manifests or evaluation artifacts, so it is safe for palette/layout-only
refreshes of an already-computed report.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from experiments.medical_dataset_gen.reports.plot_models import (
    _embedding_geometry_overview,
    _embedding_model_fcp_overview,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Regenerate embedding_model_fcp_overview and '
            'embedding_geometry_overview from an existing report.'
        )
    )
    parser.add_argument(
        '--report-dir',
        type=Path,
        default=Path('/home/fab/Projects/thesis-writing/reports/current'),
        help='Existing report directory containing data/*.csv and figures/.',
    )
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f'missing report CSV: {path}')
    with path.open(newline='') as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def main() -> None:
    args = _parse_args()
    report_dir = args.report_dir.resolve()
    output_dir = report_dir / 'figures' / 'aggregates' / 'embedding_models'
    written = []
    written.extend(
        _embedding_model_fcp_overview(
            plt=plt,
            rows=_read_rows(report_dir / 'data' / 'embedding_model_metric_summary.csv'),
            output_dir=output_dir,
        )
    )
    written.extend(
        _embedding_geometry_overview(
            plt=plt,
            rows=_read_rows(report_dir / 'data' / 'embedding_geometry_summary.csv'),
            output_dir=output_dir,
        )
    )
    if not written:
        raise RuntimeError('no embedding-model overview figures were generated')
    for path in written:
        print(path)


if __name__ == '__main__':
    main()
