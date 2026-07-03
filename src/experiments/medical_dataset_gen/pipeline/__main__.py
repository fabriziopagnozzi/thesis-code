from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable
from typing import Literal, cast, get_args

from experiments.medical_dataset_gen.schemas.global_config_schemas import (
    ExperimentCfg,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    child_experiment_names,
    load_config,
    paths_for,
    setup_logging,
)
from experiments.medical_dataset_gen.utils.provenance import PipelineProvenance
from helpers.ollama_client import stop_model

from .p01_plans import run_make_query_plans
from .p02_plans_calibration import run_calibrate_query_plans
from .p03_structured_facts import run_make_facts
from .p04_chunks import run_make_chunks
from .p05_queries_answers import run_make_queries_answers
from .p06_qrels import run_make_qrels
from .p07_embed import run_embed
from .p08_filter_queries import run_filter_queries
from .p09_evaluate import run_evaluate
from .p10_query_geom_plots import run_query_geom_plots
from .p11_eval_plots import run_eval_plots

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
    'geom_plots',
    'eval_plots',
]
PIPELINE_STAGES_SET = set[PipelineStage](get_args(PipelineStage.__value__))
type PipelineStageFn = Callable[[ExperimentCfg, MedicalDatasetGenPaths], object]

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
    ('geom_plots', run_query_geom_plots),
    ('eval_plots', run_eval_plots),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default=os.getenv('EXP') or os.getenv('EXP_NAME'))
    parser.add_argument('--from', dest='from_stage', choices=PIPELINE_STAGES_SET, default=None)
    parser.add_argument('--to', dest='to_stage', choices=PIPELINE_STAGES_SET, default=None)
    parser.add_argument('--stages', default=None)
    parser.add_argument('--release-llm', type=bool, default=None)
    parser.add_argument('--no-log-tee', action='store_true')
    parser.add_argument(
        '--parent',
        action='store_true',
        help='Run the parent experiment itself even if child subexperiments exist.',
    )
    args, _ = parser.parse_known_args()

    if args.exp is None:
        parser.error('missing experiment name; pass --exp or set EXP/EXP_NAME')

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
    selected_stage_set = set(selected_stages)
    stages_to_run = [(name, fn) for name, fn in STAGES_TO_FNS_SORTED if name in selected_stage_set]
    selected_stages = [name for name, _ in stages_to_run]

    provenance = PipelineProvenance(cfg=cfg, paths=paths, stages=selected_stages)
    if not args.no_log_tee:
        setup_logging(paths, provenance.run_id)

    print(f'[pipeline] running experiment: {paths.exp_name}')
    print(f'[pipeline] experiment={paths.exp_name} dir={paths.experiment_dir}')
    print(f'[pipeline] run_id={provenance.run_id} running stages: {selected_stages}')

    for name, fn in stages_to_run:
        if name == 'embed' and args.release_llm:
            _release_ollama(cfg)
        print(f'\n=== Stage: {name} ===')
        input_fingerprints = provenance.before_stage(name)
        fn(cfg, paths)
        provenance.after_stage(name, input_fingerprints)
    provenance.finish()


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
