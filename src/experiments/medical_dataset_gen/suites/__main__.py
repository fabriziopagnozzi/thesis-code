"""CLI for strict v5 suite validation and materialization."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from experiments.medical_dataset_gen.suites.core import (
    load_suite_spec,
    materialize_suite,
    validate_suite,
)
from experiments.medical_dataset_gen.suites.geometry import freeze_separability_strata
from experiments.medical_dataset_gen.suites.runtime import validate_materialized_suite
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Manage frozen schema-v5 experiment suites.')
    subparsers = parser.add_subparsers(dest='command', required=True)
    for command in ('validate', 'materialize', 'freeze-geometry'):
        subparser = subparsers.add_parser(command)
        subparser.add_argument('--suite', required=True, help='Suite ID or YAML spec path.')
        subparser.add_argument(
            '--results-dir', type=Path, default=MedicalDatasetGenPaths.results_dir
        )
        if command == 'validate':
            subparser.add_argument(
                '--check-artifacts',
                action='store_true',
                help='Also validate the materialized manifest, configs, and nested qrels.',
            )
    args = parser.parse_args(argv)
    results_dir = args.results_dir.expanduser().resolve()
    if args.command == 'freeze-geometry':
        path = freeze_separability_strata(
            results_dir=results_dir,
            suite_id=str(args.suite),
        )
        print(f'frozen geometry strata: {path}')
        return 0
    spec = load_suite_spec(args.suite)
    validation = validate_suite(spec)
    if spec.origin == 'native':
        cell_count = len(validation.resolved_configs)
    else:
        assert spec.source is not None
        cell_count = len(spec.source.distribution_ids) * 4 * len(spec.source.embedding_models)
    print(f'validated suite={spec.suite_id} cells={cell_count}')
    if args.command == 'materialize':
        manifest = materialize_suite(spec, results_dir=results_dir)
        print(f'materialized {len(manifest.cells)} cells')
    elif args.check_artifacts:
        result = validate_materialized_suite(
            results_dir=results_dir,
            suite_id=spec.suite_id,
        )
        if result.errors:
            raise ValueError('materialized suite validation failed:\n' + '\n'.join(result.errors))
        print(f'validated materialized artifacts cells={result.checked_cells}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
