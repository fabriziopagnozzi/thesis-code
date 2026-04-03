import polars as pl

from experiments.mimic.duck_db_init import (
    MIMIC_RESULTS_DIR,
    connect_mimic_duckdb,
)
from experiments.mimic.phase_4_evaluation.candidate_pool import CandidatePoolBuilder

from .a_build_query_prompts import build_query_prompts
from .b_gen_queries_llm import generate_queries
from .c_filter_queries import filter_queries
from .d_gold_annotation import run_gold_annotation

if __name__ == '__main__':
    con = connect_mimic_duckdb()

    conditions = pl.read_parquet(MIMIC_RESULTS_DIR / 'conditions_stats.parquet')
    chunks = pl.read_parquet(MIMIC_RESULTS_DIR / 'chunks.parquet')
    metadata = pl.read_parquet(MIMIC_RESULTS_DIR / 'admissions_metadata.parquet')
    print(
        f'Loaded {len(conditions):,} conditions, {len(chunks):,} chunks, '
        f'{len(metadata):,} admissions'
    )

    print('\n> Step 3.1: Building grounded query prompts')
    prompts_df = build_query_prompts(conditions, chunks, metadata, con)
    prompts_df.write_parquet(MIMIC_RESULTS_DIR / 'queries_prompts.parquet')
    input('\nDone. Press enter to continue...\n')

    print('\n> Step 3.2: Generating clinical questions via LLM')
    queries_df = generate_queries(prompts_df)
    queries_df.write_parquet(MIMIC_RESULTS_DIR / 'queries.parquet')
    input('\nDone. Press enter to continue...\n')

    print('\n> Step 3.3: Divergence pre-filter (facility-location vs top-k)')
    builder = CandidatePoolBuilder(con, device='cuda')
    divergence_df = filter_queries(queries_df, builder)
    divergence_df.write_parquet(MIMIC_RESULTS_DIR / 'divergence_stats.parquet')
    filtered_df = divergence_df.filter(pl.col('passes_filter'))
    n_pass = len(filtered_df)
    print(f'  {n_pass:,} / {len(divergence_df):,} queries pass filter')
    input('\nDone. Press enter to continue...\n')

    print('\n> Step 3.4: Gold facet annotation (map-reduce LLM)')
    gold_df = run_gold_annotation(filtered_df, builder)
    gold_df.write_parquet(MIMIC_RESULTS_DIR / 'gold_annotations.parquet')

    print(f'\n\nPhase 3 complete. Outputs in {MIMIC_RESULTS_DIR}:')
    print(f'  queries_prompts.parquet:  {len(prompts_df):>10,} prompts')
    print(f'  queries.parquet:          {len(queries_df):>10,} queries')
    print(f'  divergence_stats.parquet: {len(divergence_df):>10,} queries ({n_pass:,} passing)')
    print(f'  gold_annotations.parquet: {len(gold_df):>10,} annotations')
    print(f'  Avg facets per query:     {gold_df["n_facets"].mean():.1f}')
