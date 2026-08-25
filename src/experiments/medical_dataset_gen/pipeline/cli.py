"""Argument parsing and selection helpers for the pipeline CLI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import cast

from experiments.medical_dataset_gen.pipeline.stages import (
    EXPLICIT_ONLY_STAGE_SET,
    PIPELINE_STAGE_SET,
    STAGE_SPECS,
    PipelineStage,
    pipeline_stage,
    stage_index,
)
from experiments.medical_dataset_gen.utils.cli_parsing import parse_comma_separated_names
from experiments.medical_dataset_gen.utils.global_schemas import (
    DATASET_SCHEMA_VERSION_LIST,
    DatasetSchemaVersion,
    ExperimentCfg,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default=os.getenv('EXP') or os.getenv('EXP_NAME'))
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=None,
        help='Override the results root for this invocation; useful for an isolated suite.',
    )
    parser.add_argument(
        '--suite',
        type=str,
        default=None,
        help='Materialized v5 suite ID. Requires --cell and bypasses legacy directory discovery.',
    )
    parser.add_argument(
        '--cell', type=str, default=None, help='Explicit materialized suite cell ID.'
    )
    parser.add_argument(
        '--where',
        type=str,
        default=None,
        help='Batch-select materialized suite cells, for example tag=smoke|core or analysis_tier=scale.',
    )
    parser.add_argument(
        '--attempt',
        type=str,
        default=None,
        help='New immutable evaluation attempt ID for a suite cell.',
    )
    parser.add_argument(
        '--version',
        choices=[f'v{version}' for version in DATASET_SCHEMA_VERSION_LIST],
        default=None,
        help=(
            'Override dataset_schema_version from the resolved experiment config for this run, '
            'for example --version v4.'
        ),
    )

    # Normal pipeline selection.
    parser.add_argument('--from', dest='from_stage', choices=PIPELINE_STAGE_SET, default=None)
    parser.add_argument('--to', dest='to_stage', choices=PIPELINE_STAGE_SET, default=None)
    parser.add_argument('--stages', default=None)
    parser.add_argument(
        '--exclude',
        action='append',
        nargs='+',
        choices=PIPELINE_STAGE_SET,
        default=None,
        help='Exclude one or more stages from a normal pipeline run. May be repeated.',
    )

    # Secondary stage entrypoints and execution controls.
    parser.add_argument(
        '--run',
        action='append',
        default=None,
        help=(
            'Run a standalone pipeline script through the orchestrator entrypoint. '
            'May be repeated. Format: --run "eval --steps evaluation_stats".'
        ),
    )
    parser.add_argument('--no-log-tee', action='store_true')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Resolve suite selection and nested dependencies without writing or running stages.',
    )
    parser.add_argument(
        '--queries-only',
        action='store_true',
        help=(
            'At the embed stage, require and reuse existing chunk embeddings and write only '
            'query IDs and query vectors.'
        ),
    )
    parser.add_argument(
        '--parent',
        action='store_true',
        help='Run the parent experiment itself even if child subexperiments exist.',
    )
    return parser


def with_dataset_schema_version(
    cfg: ExperimentCfg,
    raw_version: str | None,
) -> ExperimentCfg:
    if raw_version is None:
        return cfg

    version = cast(DatasetSchemaVersion, int(raw_version.removeprefix('v')))
    if cfg.dataset_schema_version != version:
        print(
            '[pipeline] overriding dataset_schema_version: '
            f'v{cfg.dataset_schema_version} -> v{version}'
        )
    raw_cfg = cfg.model_dump(mode='python', by_alias=True)
    raw_cfg['dataset_schema_version'] = version
    return ExperimentCfg.model_validate(raw_cfg)


def selected_stage_names(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> list[PipelineStage]:
    if args.stages is not None:
        if args.from_stage or args.to_stage:
            parser.error('--stages cannot be combined with --from or --to')
        try:
            parsed = parse_comma_separated_names(
                raw_value=args.stages,
                valid_names=PIPELINE_STAGE_SET,
                option_name='--stages',
            )
        except ValueError as exc:
            parser.error(str(exc))
        selected = [pipeline_stage(name) for name in parsed or []]
    else:
        start_idx = stage_index(pipeline_stage(args.from_stage)) if args.from_stage else 0
        stop_idx = (
            stage_index(pipeline_stage(args.to_stage)) if args.to_stage else len(STAGE_SPECS) - 1
        )
        if stop_idx < start_idx:
            parser.error('--to must be the same as or later than --from')
        selected = [spec.name for spec in STAGE_SPECS[start_idx : stop_idx + 1]]

        # Destructive maintenance stages must be named through a range bound.
        # A default run, or an open-ended --from run, must never imply cleanup.
        explicitly_named_bounds = {args.from_stage, args.to_stage}
        selected = [
            stage
            for stage in selected
            if stage not in EXPLICIT_ONLY_STAGE_SET or stage in explicitly_named_bounds
        ]

    excluded = {pipeline_stage(stage) for group in args.exclude or [] for stage in group}
    selected_set = set(selected) - excluded

    return [spec.name for spec in STAGE_SPECS if spec.name in selected_set]
