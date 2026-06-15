import argparse
from collections.abc import Callable

from helpers.ollama_client import stop_model

from .embedding_geometry.run import run_embedding_geometry
from .evaluation.evaluate import run_evaluate
from .evaluation.plots import store_eval_figures
from .generation.chunks import run_make_chunks
from .generation.facts import run_make_facts
from .generation.qrels import run_make_qrels
from .generation.queries_answers import run_make_queries_answers
from .generation.query_plans import run_make_query_plans
from .global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
    dump_effective_config,
    load_config_from_cli,
    paths_for,
    setup_logging,
)
from .retrieval.embed import run_embed
from .retrieval.filter_geometry import run_filter_geometry

type StageFn = Callable[[ExperimentCfg, MedicalDatasetGenPaths], object]


STAGES: list[tuple[str, StageFn]] = [
    ('plans', run_make_query_plans),
    ('facts', run_make_facts),
    ('chunks', run_make_chunks),
    ('queries_answers', run_make_queries_answers),
    ('qrels', run_make_qrels),
    ('embed', run_embed),
    ('geom_filter', run_filter_geometry),
    ('eval', run_evaluate),
    ('geom_plots', run_embedding_geometry),
    ('eval_plots', store_eval_figures),
]


def main() -> None:
    stage_parser = argparse.ArgumentParser()
    stage_parser.add_argument(
        '--from', dest='from_stage', choices=[name for name, _ in STAGES], default=None
    )
    stage_parser.add_argument(
        '--to', dest='to_stage', choices=[name for name, _ in STAGES], default=None
    )
    stage_parser.add_argument('--only', choices=[name for name, _ in STAGES], default=None)
    stage_parser.add_argument('--release-llm', type=bool, default=None)
    stage_parser.add_argument('--no-log-tee', action='store_true')
    args, _ = stage_parser.parse_known_args()

    cfg = load_config_from_cli()

    paths = paths_for(cfg)
    if not args.no_log_tee:
        setup_logging(paths)
    dump_effective_config(cfg, paths)

    if args.only:
        start_idx = _stage_index(args.only)
        stop_idx = _stage_index(args.only)
    else:
        start_idx = _stage_index(args.from_stage) if args.from_stage else 0
        stop_idx = _stage_index(args.to_stage) if args.to_stage else len(STAGES) - 1

    if stop_idx < start_idx:
        raise ValueError('--to must be the same as or later than --from')

    print(f'[pipeline] experiment={paths.exp_name} dir={paths.experiment_dir}')
    print(f'[pipeline] running stages: {[name for name, _ in STAGES[start_idx : stop_idx + 1]]}')
    for name, fn in STAGES[start_idx : stop_idx + 1]:
        if name == 'embed' and args.release_llm:
            _release_ollama(cfg)
        print(f'\n=== Stage: {name} ===')
        fn(cfg, paths)


def _stage_index(name: str) -> int:
    for i, (stage_name, _) in enumerate(STAGES):
        if stage_name == name:
            return i
    raise KeyError(name)


def _release_ollama(cfg: ExperimentCfg) -> None:
    llm_used = cfg.generation.use_llm_chunk_generation or cfg.generation.use_llm_query_paraphrase
    if not llm_used:
        return

    print(f'[pipeline] releasing Ollama model before embeddings: {cfg.generation.llm_name}')
    stop_model(cfg.generation.llm_name)


if __name__ == '__main__':
    main()
