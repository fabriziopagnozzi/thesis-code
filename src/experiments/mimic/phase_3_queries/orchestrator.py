import duckdb
import polars as pl

from experiments.mimic.configs import (
    BuildQueryPromptsCfg,
    FilterQueriesCfg,
    GenQueriesCfg,
    GoldAnnotationCfg,
)
from experiments.mimic.duck_db_init import MIMIC_RESULTS_DIR, connect_mimic_duckdb

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
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if con is None:
        con = connect_mimic_duckdb()

    print('\n> Step 3.1: Building grounded query prompts')
    prompts_df = run_build_query_prompts(con, cfg=build_query_prompts_cfg)

    print('\n> Step 3.2: Generating clinical questions via LLM')
    queries_df = run_gen_queries_llm(cfg=gen_queries_cfg)

    print('\n> Step 3.3: Divergence pre-filter (facility-location vs top-k)')
    divergence_df = run_filter_queries(con, cfg=filter_queries_cfg)
    n_pass = divergence_df.filter(pl.col('passes_filter')).height

    print('\n> Step 3.4: Gold facet annotation (map-reduce LLM)')
    gold_df = run_gold_annotation(con, cfg=gold_annotation_cfg)

    print(
        f'\n\nPhase 3 complete. Outputs in {MIMIC_RESULTS_DIR}:\n'
        f'  queries_prompts.parquet:  {len(prompts_df):>10,} prompts\n'
        f'  queries.parquet:          {len(queries_df):>10,} queries\n'
        f'  divergence_stats.parquet: {len(divergence_df):>10,} queries ({n_pass:,} passing)\n'
        f'  gold_annotations.parquet: {len(gold_df):>10,} annotations\n'
        f'  Avg facets per query:     {gold_df["n_facets"].mean():.1f}'
    )
    return prompts_df, queries_df, divergence_df, gold_df


if __name__ == '__main__':
    run_phase_3()
