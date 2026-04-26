import argparse

import duckdb

from experiments.mimic.configs import (
    BuildQueryPromptsCfg,
    FilterQueriesCfg,
    GenQueriesCfg,
    GoldAnnotationCfg,
    setup_logging,
)
from experiments.mimic.utils.constants import MimicPaths
from experiments.mimic.utils.duck_db_init import connect_mimic_duckdb

from .a_build_query_prompts import run_build_query_prompts
from .b_gen_queries_llm import run_gen_queries_llm
from .c_filter_queries import run_filter_queries
from .d_gold_annotation import run_gold_annotation

STEPS = (1, 2, 3, 4)


def run_queries_subpipeline(
    con: duckdb.DuckDBPyConnection | None = None,
    build_query_prompts_cfg: BuildQueryPromptsCfg | None = None,
    gen_queries_cfg: GenQueriesCfg | None = None,
    filter_queries_cfg: FilterQueriesCfg | None = None,
    gold_annotation_cfg: GoldAnnotationCfg | None = None,
    steps: list[int] | None = None,
):
    if steps is None:
        steps = list(STEPS)

    if con is None:
        con = connect_mimic_duckdb()

    if 1 in steps:
        print('\n> Step 3.1: Building grounded query prompts')
        run_build_query_prompts(con, cfg=build_query_prompts_cfg)

    if 2 in steps:
        print('\n> Step 3.2: Generating clinical questions via LLM')
        run_gen_queries_llm(cfg=gen_queries_cfg)

    if 3 in steps:
        print('\n> Step 3.3: Divergence pre-filter (facility-location vs top-k)')
        run_filter_queries(con, cfg=filter_queries_cfg)

    if 4 in steps:
        print('\n> Step 3.4: Gold facet annotation (map-reduce LLM)')
        run_gold_annotation(con, cfg=gold_annotation_cfg)

    print(f'\n\nPhase 3 complete. Outputs in {MimicPaths.experiment}:\n')


if __name__ == '__main__':
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--steps',
        type=int,
        nargs='*',
        default=list(STEPS),
        choices=STEPS,
        help='Subphase(s) to run (default: all)',
    )
    args = parser.parse_args()
    run_queries_subpipeline(steps=sorted(args.steps))
