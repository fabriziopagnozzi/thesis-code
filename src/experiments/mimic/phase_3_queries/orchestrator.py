import argparse

import duckdb

from experiments.mimic.configs import (
    MIMIC_RESULTS_DIR,
    BuildQueryPromptsCfg,
    FilterQueriesCfg,
    GenQueriesCfg,
    GoldAnnotationCfg,
)
from experiments.mimic.duck_db_init import connect_mimic_duckdb

from .a_build_query_prompts import run_build_query_prompts
from .b_gen_queries_llm import run_gen_queries_llm
from .c_filter_queries import run_filter_queries
from .d_gold_annotation import run_gold_annotation


def run_phase_3(
    con: duckdb.DuckDBPyConnection | None = None,
    build_query_prompts_cfg: BuildQueryPromptsCfg | None = None,
    gen_queries_cfg: GenQueriesCfg | None = None,
    filter_queries_cfg: FilterQueriesCfg | None = None,
    gold_annotation_cfg: GoldAnnotationCfg | None = None,
    from_step: int = 1,
):
    if con is None:
        con = connect_mimic_duckdb()

    if from_step <= 1:
        print('\n> Step 3.1: Building grounded query prompts')
        run_build_query_prompts(con, cfg=build_query_prompts_cfg)

    if from_step <= 2:
        print('\n> Step 3.2: Generating clinical questions via LLM')
        run_gen_queries_llm(cfg=gen_queries_cfg)

    if from_step <= 3:
        print('\n> Step 3.3: Divergence pre-filter (facility-location vs top-k)')
        run_filter_queries(con, cfg=filter_queries_cfg)

    if from_step <= 4:
        print('\n> Step 3.4: Gold facet annotation (map-reduce LLM)')
        run_gold_annotation(con, cfg=gold_annotation_cfg)

    print(f'\n\nPhase 3 complete. Outputs in {MIMIC_RESULTS_DIR}:\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--from',
        dest='from_step',
        type=int,
        default=1,
        metavar='STEP',
        help='Resume from this step number (1-4)',
    )
    args = parser.parse_args()
    run_phase_3(from_step=args.from_step)
