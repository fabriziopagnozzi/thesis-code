"""Execution of standalone pipeline scripts requested with ``--run``."""

from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass

from experiments.medical_dataset_gen.pipeline.stages import (
    STANDALONE_SCRIPT_BY_NAME,
    STANDALONE_SCRIPT_SET,
    StandalonePipelineScript,
    standalone_script,
)
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths
from experiments.medical_dataset_gen.utils.logging_utils import colorprint, setup_logging


@dataclass(frozen=True)
class StandaloneRunSpec:
    script: StandalonePipelineScript
    script_args: list[str]


def parse_run_specs(
    parser: argparse.ArgumentParser,
    raw_runs: list[str],
) -> list[StandaloneRunSpec]:
    run_specs: list[StandaloneRunSpec] = []
    for raw_run in raw_runs:
        parts = shlex.split(raw_run)
        if not parts:
            parser.error('--run value cannot be empty')

        script_name = parts[0]
        if script_name not in STANDALONE_SCRIPT_SET:
            parser.error(
                f'unknown standalone script in --run: {script_name}. '
                + 'Valid scripts: '
                + ', '.join(sorted(STANDALONE_SCRIPT_SET))
            )
        run_specs.append(
            StandaloneRunSpec(
                script=standalone_script(script_name),
                script_args=parts[1:],
            )
        )
    return run_specs


def validate_run_mode_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    unknown_args: list[str],
) -> None:
    if args.from_stage or args.to_stage or args.stages or args.exclude:
        parser.error('--run cannot be combined with --from, --to, --stages, or --exclude')
    if unknown_args:
        parser.error(
            'unknown argument(s) outside --run: '
            + ' '.join(unknown_args)
            + '. Put standalone script arguments inside the quoted --run value.'
        )


def run_standalone_script_sequence(
    *,
    run_specs: list[StandaloneRunSpec],
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    no_log_tee: bool,
) -> None:
    if not no_log_tee:
        setup_logging(paths)

    colorprint(
        'bright_blue',
        f'[pipeline] running standalone scripts: {[spec.script for spec in run_specs]}',
    )
    print(f'[pipeline] experiment={paths.exp_name} dir={paths.experiment_dir}')

    for run_spec in run_specs:
        script_spec = STANDALONE_SCRIPT_BY_NAME[run_spec.script]

        print(f'\n=== Script: {run_spec.script} ===')
        colorprint(
            'bright_blue',
            f'[pipeline] running standalone script: {run_spec.script} experiment={paths.exp_name}',
        )
        script_spec.run(cfg, paths, run_spec.script_args)
