from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast, get_args

from experiments.medical_dataset_gen.schemas.global_config_schemas import (
    ExperimentCfg,
)
from experiments.medical_dataset_gen.utils.exp_naming import child_experiment_names
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    load_config,
    paths_for,
)

# from experiments.medical_dataset_gen.utils.provenance import PipelineProvenance
from experiments.medical_dataset_gen.utils.logging import colorprint, setup_logging
from helpers.ollama_client import stop_model

from .p01_plans import run_make_query_plans
from .p02_calibrate_plans import run_calibrate_query_plans
from .p03_facts import run_make_facts
from .p04_chunks import run_make_chunks
from .p05_queries_answers import run_make_queries_answers
from .p06_qrels import run_make_qrels
from .p07_embed import run_embed
from .p08_filter_queries import run_filter_queries
from .p09_eval import parse_evaluate_cli_args, run_evaluate
from .p10_eval_plots import parse_plots_cli_args, run_eval_plots
from .p11_geom_plots import parse_geom_plots_cli_args, run_query_geom_plots

type PipelineStage = Literal[
    'plans',
    'calibrate_plans',
    'facts',
    'chunks',
    'queries_answers',
    'qrels',
    'embed',
    'filter_queries',
    'eval',
    'eval_plots',
    'geom_plots',
]
PIPELINE_STAGES_SET = set[PipelineStage](get_args(PipelineStage.__value__))
type PipelineStageFn = Callable[[ExperimentCfg, MedicalDatasetGenPaths], object]
type StandalonePipelineScript = Literal['eval', 'geom_plots', 'eval_plots']
STANDALONE_PIPELINE_SCRIPTS = set[StandalonePipelineScript](
    get_args(StandalonePipelineScript.__value__)
)


@dataclass(frozen=True)
class StandaloneRunSpec:
    script: StandalonePipelineScript
    script_args: list[str]


STAGES_TO_FNS_SORTED: list[tuple[PipelineStage, PipelineStageFn]] = [
    ('plans', run_make_query_plans),
    ('calibrate_plans', run_calibrate_query_plans),
    ('facts', run_make_facts),
    ('chunks', run_make_chunks),
    ('queries_answers', run_make_queries_answers),
    ('qrels', run_make_qrels),
    ('embed', run_embed),
    ('filter_queries', run_filter_queries),
    ('eval', run_evaluate),
    ('eval_plots', run_eval_plots),
    ('geom_plots', run_query_geom_plots),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default=os.getenv('EXP') or os.getenv('EXP_NAME'))
    parser.add_argument('--from', dest='from_stage', choices=PIPELINE_STAGES_SET, default=None)
    parser.add_argument('--to', dest='to_stage', choices=PIPELINE_STAGES_SET, default=None)
    parser.add_argument('--stages', default=None)
    parser.add_argument(
        '--exclude',
        action='append',
        nargs='+',
        choices=PIPELINE_STAGES_SET,
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
    parser.add_argument('--release-llm', type=bool, default=None)
    parser.add_argument('--no-log-tee', action='store_true')
    parser.add_argument(
        '--parent',
        action='store_true',
        help='Run the parent experiment itself even if child subexperiments exist.',
    )
    args, unknown_args = parser.parse_known_args()

    if args.exp is None:
        parser.error('missing experiment name; pass --exp or set EXP/EXP_NAME')

    if args.run is not None:
        _validate_run_mode_args(parser, args, unknown_args)
        run_specs = _parse_run_specs(parser=parser, raw_runs=args.run)
    else:
        run_specs = None

    children = [] if args.parent else child_experiment_names(args.exp)
    if children:
        print(f'[pipeline] experiment={args.exp} has children: {children}')
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
        return

    if run_specs is not None:
        _run_standalone_script_sequence(
            run_specs=run_specs,
            exp=args.exp,
            no_log_tee=args.no_log_tee,
        )
        return

    cfg = load_config(args.exp)

    paths = paths_for(cfg)

    if args.stages is not None:
        if args.from_stage or args.to_stage:
            parser.error('--stages cannot be combined with --from or --to')
        selected_stages = _parse_stages_arg(parser, args.stages)
    else:
        start_idx = _stage_index(args.from_stage) if args.from_stage else 0
        stop_idx = _stage_index(args.to_stage) if args.to_stage else len(STAGES_TO_FNS_SORTED) - 1
        if stop_idx < start_idx:
            raise ValueError('--to must be the same as or later than --from')
        selected_stages = [name for name, _ in STAGES_TO_FNS_SORTED[start_idx : stop_idx + 1]]

    excluded_stages = {stage for excluded_group in args.exclude or [] for stage in excluded_group}
    selected_stage_set = set(selected_stages) - excluded_stages
    stages_to_run = [(name, fn) for name, fn in STAGES_TO_FNS_SORTED if name in selected_stage_set]
    selected_stages = [name for name, _ in stages_to_run]

    # provenance = PipelineProvenance(cfg=cfg, paths=paths, stages=selected_stages)
    # if not args.no_log_tee:
    # setup_logging(paths, provenance.run_id)

    colorprint('teal', f'\n[pipeline] running experiment: {paths.exp_name}')
    # colorprint('bright_cyan', f'[pipeline] dir={paths.experiment_dir}')
    # print(f'[pipeline] run_id={provenance.run_id} running stages: {selected_stages}')

    for name, fn in stages_to_run:
        if name == 'embed' and args.release_llm:
            _release_ollama(cfg)
        colorprint('bright_green', f'\n{"=" * 3} Stage: {name} {"=" * 3}')
        # input_fingerprints = provenance.before_stage(name)
        fn(cfg, paths)
        # provenance.after_stage(name, input_fingerprints)
    # provenance.finish()


def _validate_run_mode_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    unknown_args: list[str],
) -> None:
    if args.from_stage or args.to_stage or args.stages or args.exclude:
        parser.error('--run cannot be combined with --from, --to, --stages, or --exclude')
    if args.release_llm is not None:
        parser.error('--release-llm is only supported for normal stage runs')
    if unknown_args:
        parser.error(
            'unknown argument(s) outside --run: '
            + ' '.join(unknown_args)
            + '. Put standalone script arguments inside the quoted --run value.'
        )


def _parse_run_specs(
    *,
    parser: argparse.ArgumentParser,
    raw_runs: list[str],
) -> list[StandaloneRunSpec]:
    run_specs: list[StandaloneRunSpec] = []
    for raw_run in raw_runs:
        parts = shlex.split(raw_run)
        if not parts:
            parser.error('--run value cannot be empty')

        script_name = parts[0]
        if script_name not in STANDALONE_PIPELINE_SCRIPTS:
            parser.error(
                f'unknown standalone script in --run: {script_name}. '
                + 'Valid scripts: '
                + ', '.join(sorted(STANDALONE_PIPELINE_SCRIPTS))
            )
        run_specs.append(
            StandaloneRunSpec(
                script=cast(StandalonePipelineScript, script_name),
                script_args=parts[1:],
            )
        )

    return run_specs


def _run_standalone_script_sequence(
    *,
    run_specs: list[StandaloneRunSpec],
    exp: str,
    no_log_tee: bool,
) -> None:
    cfg = load_config(exp)
    paths = paths_for(cfg)
    if not no_log_tee:
        setup_logging(paths)

    colorprint(
        'aqua', f'[pipeline] running standalone scripts: {[spec.script for spec in run_specs]})'
    )
    print(f'[pipeline] experiment={paths.exp_name} dir={paths.experiment_dir}')

    for run_spec in run_specs:
        print(f'\n=== Script: {run_spec.script} ===')
        _run_standalone_script(run_spec=run_spec, exp=exp)


def _run_standalone_script(*, run_spec: StandaloneRunSpec, exp: str) -> None:
    script_argv = [
        *run_spec.script_args,
        '--exp',
        exp,
    ]

    if run_spec.script == 'eval':
        cfg, selected_steps = parse_evaluate_cli_args(script_argv)
        paths = paths_for(cfg)
        colorprint(
            'teal',
            f'[pipeline] running standalone script: {run_spec.script} experiment={paths.exp_name}',
        )
        run_evaluate(cfg, paths, selected_steps=selected_steps)
    elif run_spec.script == 'geom_plots':
        cfg, selected_plots = parse_geom_plots_cli_args(script_argv)
        paths = paths_for(cfg)
        colorprint(
            'teal',
            f'[pipeline] running standalone script: {run_spec.script} experiment={paths.exp_name}',
        )
        run_query_geom_plots(cfg, paths, selected_plots=selected_plots)
    elif run_spec.script == 'eval_plots':
        cfg, selected_plots = parse_plots_cli_args(script_argv)
        paths = paths_for(cfg)
        colorprint(
            'teal',
            f'[pipeline] running standalone script: {run_spec.script} experiment={paths.exp_name}',
        )
        run_eval_plots(cfg, paths, selected_plots=selected_plots)
    else:
        raise KeyError(run_spec.script)


def _stage_index(name: str) -> int:
    for i, (stage_name, _) in enumerate(STAGES_TO_FNS_SORTED):
        if stage_name == name:
            return i
    raise KeyError(name)


def _parse_stages_arg(parser: argparse.ArgumentParser, raw_stages: str) -> list[PipelineStage]:
    stages = [stage.strip() for stage in raw_stages.split(',')]
    if not stages or any(not stage for stage in stages):
        parser.error('--stages must be a comma-separated list of stage names')

    invalid_stages = [stage for stage in stages if stage not in PIPELINE_STAGES_SET]
    if invalid_stages:
        parser.error(
            '--stages contains invalid stage name(s): '
            + ', '.join(invalid_stages)
            + '. Valid stages: '
            + ', '.join(name for name, _ in STAGES_TO_FNS_SORTED)
        )

    seen: set[str] = set()
    duplicate_stages: list[str] = []
    for stage in stages:
        if stage in seen and stage not in duplicate_stages:
            duplicate_stages.append(stage)
        seen.add(stage)
    if duplicate_stages:
        parser.error('--stages contains duplicate stage name(s): ' + ', '.join(duplicate_stages))

    return [cast(PipelineStage, stage) for stage in stages]


def _release_ollama(cfg: ExperimentCfg) -> None:
    llm_used = (
        cfg.generation.llm_config.use_llm_chunk_generation
        or cfg.generation.llm_config.use_llm_query_paraphrase
    )
    if not llm_used:
        return

    print(
        f'[pipeline] releasing Ollama model before embeddings: {cfg.generation.llm_config.model_name}'
    )
    stop_model(cfg.generation.llm_config.model_name)


if __name__ == '__main__':
    main()
