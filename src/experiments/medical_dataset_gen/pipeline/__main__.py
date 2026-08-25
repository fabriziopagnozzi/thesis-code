"""Command-line entrypoint for the synthetic medical benchmark pipeline."""

from __future__ import annotations

from experiments.medical_dataset_gen.pipeline.cli import build_parser
from experiments.medical_dataset_gen.pipeline.normal import run_normal_mode
from experiments.medical_dataset_gen.pipeline.standalone import (
    parse_run_specs,
    validate_run_mode_args,
)
from experiments.medical_dataset_gen.pipeline.suite import run_suite_mode
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths


def main() -> None:
    parser = build_parser()
    args, unknown_args = parser.parse_known_args()
    if args.results_dir is not None:
        MedicalDatasetGenPaths.results_dir = args.results_dir.expanduser().resolve()

    if args.exp is None and args.suite is None:
        parser.error('missing experiment; pass --exp or --suite with --cell/--where')
    if args.exp is not None and args.suite is not None:
        parser.error('--exp and --suite are mutually exclusive')
    if args.suite is not None and (args.cell is None) == (args.where is None):
        parser.error('--suite requires exactly one of --cell or --where')

    # Standalone scripts own their secondary CLI arguments; the orchestrator
    # validates the outer command before forwarding the quoted arguments.
    run_specs = parse_run_specs(parser, args.run) if args.run is not None else None
    if run_specs is not None:
        validate_run_mode_args(parser, args, unknown_args)
    elif unknown_args:
        parser.error('unknown argument(s): ' + ' '.join(unknown_args))

    if args.suite is not None:
        run_suite_mode(parser=parser, args=args, run_specs=run_specs)
        return

    run_normal_mode(parser=parser, args=args, run_specs=run_specs)


if __name__ == '__main__':
    main()
