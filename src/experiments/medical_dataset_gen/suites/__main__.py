"""CLI for validating and materializing declarative experiment suites."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from experiments.medical_dataset_gen.suites.core import (
    load_suite_spec,
    materialize_suite,
    suite_spec_root,
    validate_suite,
)
from experiments.medical_dataset_gen.suites.geometry import freeze_separability_strata
from experiments.medical_dataset_gen.suites.runtime import validate_materialized_suite
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Validate or materialize a v5 experiment suite.')
    subparsers = parser.add_subparsers(dest='command', required=True)
    for command in ('validate', 'materialize', 'freeze-geometry'):
        subparser = subparsers.add_parser(command)
        subparser.add_argument('--suite', required=True, help='Suite ID or path to its YAML spec.')
        subparser.add_argument(
            '--results-dir', type=Path, default=MedicalDatasetGenPaths.results_dir
        )
        subparser.add_argument(
            '--check-artifacts',
            action='store_true',
            help='Also validate a materialized manifest, factor drift, and completed nested qrels.',
        )
        subparser.add_argument(
            '--verify-hashes',
            action='store_true',
            help='With --check-artifacts, stream and verify immutable artifact hashes.',
        )
        if command == 'materialize':
            replacement_mode = subparser.add_mutually_exclusive_group()
            replacement_mode.add_argument(
                '--replace-planned',
                action='store_true',
                help='Replace only a metadata-only native suite whose every cell is still planned.',
            )
            replacement_mode.add_argument(
                '--refresh-planned-execution',
                action='store_true',
                help=(
                    'Refresh run-profile/config snapshots only when every cell remains planned '
                    'and the generated-dataset hashes are unchanged.'
                ),
            )
            replacement_mode.add_argument(
                '--prune-planned',
                action='store_true',
                help=(
                    'Remove only obsolete metadata-only planned distributions while preserving '
                    'unchanged generated smoke data.'
                ),
            )
        if command == 'freeze-geometry':
            subparser.add_argument('--replace', action='store_true')
    args = parser.parse_args(argv)
    results_dir = args.results_dir.expanduser().resolve()
    spec_path = Path(args.suite)
    if not spec_path.suffix:
        spec_path = suite_spec_root() / f'{args.suite}.yaml'
    spec = load_suite_spec(args.suite) if spec_path.is_file() else None
    if args.command == 'freeze-geometry':
        path = freeze_separability_strata(
            results_dir=results_dir,
            suite_id=str(args.suite),
            replace=bool(args.replace),
        )
        print(f'frozen geometry strata: {path}')
        return 0
    if spec is not None:
        validation = validate_suite(spec)
        print(f'validated suite={spec.suite_id} cells={len(validation.resolved_configs)}')
        for warning in validation.warnings:
            print(f'warning: {warning}')
    elif args.command == 'materialize':
        raise FileNotFoundError(
            f'{args.suite}: no declarative suite specification at {spec_path}; '
            'migrated suites are archival and cannot be materialized'
        )
    else:
        print(f'validated archived manifest suite={args.suite}')
    if args.command == 'materialize':
        assert spec is not None
        manifest = materialize_suite(
            spec,
            results_dir=results_dir,
            replace_planned=bool(args.replace_planned),
            refresh_planned_execution=bool(args.refresh_planned_execution),
            prune_planned=bool(args.prune_planned),
        )
        print(f'materialized {len(manifest.cells)} cells')
    if args.check_artifacts or spec is None:
        result = validate_materialized_suite(
            results_dir=results_dir,
            suite_id=spec.suite_id if spec is not None else str(args.suite),
            verify_hashes=bool(args.verify_hashes),
        )
        if result.errors:
            raise ValueError('materialized suite validation failed:\n' + '\n'.join(result.errors))
        print(f'validated materialized artifacts cells={result.checked_cells}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
