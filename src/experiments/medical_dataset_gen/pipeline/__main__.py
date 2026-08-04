from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import cast

from experiments.medical_dataset_gen.embedding.stage import run_embed
from experiments.medical_dataset_gen.pipeline.stages import (
    PIPELINE_STAGE_SET,
    STAGE_BY_NAME,
    STAGE_SPECS,
    STANDALONE_SCRIPT_BY_NAME,
    STANDALONE_SCRIPT_SET,
    PipelineStage,
    StageSpec,
    StandalonePipelineScript,
    pipeline_stage,
    stage_index,
    standalone_script,
)
from experiments.medical_dataset_gen.utils.cli_parsing import parse_comma_separated_names
from experiments.medical_dataset_gen.utils.exp_naming import child_experiment_names
from experiments.medical_dataset_gen.utils.global_schemas import (
    DATASET_SCHEMA_VERSION_LIST,
    DatasetSchemaVersion,
    ExperimentCfg,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    load_config,
    paths_for,
)
from experiments.medical_dataset_gen.utils.logging_utils import colorprint, setup_logging


@dataclass(frozen=True)
class StandaloneRunSpec:
    script: StandalonePipelineScript
    script_args: list[str]


def main() -> None:
    parser = _build_parser()
    args, unknown_args = parser.parse_known_args()

    if args.exp is None:
        parser.error('missing experiment name; pass --exp or set EXP/EXP_NAME')

    run_specs = _parse_run_specs(parser, args.run) if args.run is not None else None
    if run_specs is not None:
        _validate_run_mode_args(parser, args, unknown_args)
    elif unknown_args:
        parser.error('unknown argument(s): ' + ' '.join(unknown_args))

    children = [] if args.parent else child_experiment_names(args.exp)
    if children:
        _run_child_experiments(children=children, parent_exp=args.exp)
        return

    cfg = _with_dataset_schema_version(load_config(args.exp), args.version)
    paths = paths_for(cfg)

    if run_specs is not None:
        _run_standalone_script_sequence(
            run_specs=run_specs,
            cfg=cfg,
            paths=paths,
            no_log_tee=args.no_log_tee,
        )
        return

    stages_to_run = _selected_stage_names(parser, args)
    _run_pipeline_stages(
        cfg=cfg,
        paths=paths,
        stage_names=stages_to_run,
        queries_only=args.queries_only,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default=os.getenv('EXP') or os.getenv('EXP_NAME'))
    parser.add_argument(
        '--version',
        choices=[f'v{version}' for version in DATASET_SCHEMA_VERSION_LIST],
        default=None,
        help=(
            'Override dataset_schema_version from the resolved experiment config for this run, '
            'for example --version v4.'
        ),
    )
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


def _with_dataset_schema_version(
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


def _validate_run_mode_args(
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


def _run_child_experiments(*, children: list[str], parent_exp: str) -> None:
    print(f'[pipeline] experiment={parent_exp} has children: {children}')
    for child_exp in children:
        print(f'\n[pipeline] launching child experiment: {child_exp}')
        subprocess.run(
            [
                sys.executable,
                '-m',
                'experiments.medical_dataset_gen.pipeline',
                *sys.argv[1:],
                '--exp',
                child_exp,
            ],
            check=True,
        )


def _parse_run_specs(
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


def _run_standalone_script_sequence(
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


def _selected_stage_names(
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

    excluded = {pipeline_stage(stage) for group in args.exclude or [] for stage in group}
    selected_set = set(selected) - excluded
    return [spec.name for spec in STAGE_SPECS if spec.name in selected_set]


def _run_pipeline_stages(
    *,
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    stage_names: list[PipelineStage],
    queries_only: bool = False,
) -> None:
    colorprint('bright_blue', f'\n[pipeline] running experiment: {paths.exp_name}')

    for stage_name in stage_names:
        spec = STAGE_BY_NAME[stage_name]
        if _should_skip_shared_stage(cfg, paths, spec):
            continue
        if spec.ready is not None and spec.ready(paths):
            print(f'[pipeline] skipping {stage_name}; artifacts already exist')
            continue
        colorprint('bright_green', f'\n{"=" * 3} Stage: {stage_name} {"=" * 3}')
        if stage_name == 'embed':
            run_embed(cfg, paths, queries_only=queries_only)
        else:
            spec.run(cfg, paths)


def _should_skip_shared_stage(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    spec: StageSpec,
) -> bool:
    if not spec.shared_outputs or not paths.uses_shared_generation():
        return False

    required_outputs = spec.shared_outputs
    if spec.name == 'queries_answers' and not cfg.retrieval.compute_answer_rouge:
        # Retrieval-only runs consume query text but never load answer references.
        required_outputs = ('queries',)

    missing = [table for table in required_outputs if not paths.table_path(table).exists()]
    if missing:
        return False

    print(
        f'[pipeline] skipping shared stage {spec.name}; existing outputs in '
        f'{paths.table_path(required_outputs[0]).parent}'
    )
    return True


if __name__ == '__main__':
    main()
