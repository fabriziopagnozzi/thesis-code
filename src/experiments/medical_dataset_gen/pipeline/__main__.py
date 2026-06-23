from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Literal, get_args

from experiments.medical_dataset_gen.global_config import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
    load_config_from_cli,
    paths_for,
    setup_logging,
)
from experiments.medical_dataset_gen.pipeline.p01_plans import (
    run_make_query_plans,
)
from experiments.medical_dataset_gen.pipeline.p02_plans_calibration import (
    run_calibrate_query_plans,
)
from experiments.medical_dataset_gen.pipeline.p03_structured_facts import run_make_facts
from experiments.medical_dataset_gen.pipeline.p04_chunks import run_make_chunks
from experiments.medical_dataset_gen.pipeline.p05_queries_answers import (
    run_make_queries_answers,
)
from experiments.medical_dataset_gen.pipeline.p06_qrels import (
    run_make_qrels,
)
from experiments.medical_dataset_gen.pipeline.p07_embed import run_embed
from experiments.medical_dataset_gen.pipeline.p08_filter_queries import run_filter_queries
from experiments.medical_dataset_gen.pipeline.p09_evaluate import run_evaluate
from experiments.medical_dataset_gen.pipeline.p10_query_geom_plots import (
    run_query_geom_plots,
)
from experiments.medical_dataset_gen.pipeline.p11_eval_plots import run_eval_plots
from experiments.medical_dataset_gen.utils.provenance import PipelineProvenance
from helpers.ollama_client import stop_model

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
    parser.add_argument('--from', dest='from_stage', choices=PIPELINE_STAGES_SET, default=None)
    parser.add_argument('--to', dest='to_stage', choices=PIPELINE_STAGES_SET, default=None)
    parser.add_argument('--only', choices=PIPELINE_STAGES_SET, default=None)
    parser.add_argument('--release-llm', type=bool, default=None)
    parser.add_argument('--no-log-tee', action='store_true')
    args, _ = parser.parse_known_args()

    cfg = load_config_from_cli()

    paths = paths_for(cfg)

    if args.only:
        start_idx = _stage_index(args.only)
        stop_idx = _stage_index(args.only)
    else:
        start_idx = _stage_index(args.from_stage) if args.from_stage else 0
        stop_idx = _stage_index(args.to_stage) if args.to_stage else len(STAGES_TO_FNS_SORTED) - 1

    if stop_idx < start_idx:
        raise ValueError('--to must be the same as or later than --from')

    selected_stages = [name for name, _ in STAGES_TO_FNS_SORTED[start_idx : stop_idx + 1]]
    provenance = PipelineProvenance(cfg=cfg, paths=paths, stages=selected_stages)
    if not args.no_log_tee:
        setup_logging(paths, provenance.run_id)

    print(f'[pipeline] experiment={paths.exp_name} dir={paths.experiment_dir}')
    print(f'[pipeline] run_id={provenance.run_id} running stages: {selected_stages}')

    for name, fn in STAGES_TO_FNS_SORTED[start_idx : stop_idx + 1]:
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
