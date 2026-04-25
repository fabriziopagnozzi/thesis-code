"""
Step 3.2: Generate actual clinical questions from prompts via ollama.

Reads queries_prompts.parquet and calls a local LLM to produce the clinical question for each row.

Output: queries.parquet - all original columns + query_text.
If queries.parquet is already existing, it appends the new queries there.
"""

import sys
from typing import cast

import polars as pl
from tqdm import tqdm

from experiments.mimic.configs import GenQueriesCfg, get_table_path, setup_logging
from experiments.mimic.schemas import QueryPromptRow, QueryRow
from helpers.ollama_client import generate

gen_queries_cfg = GenQueriesCfg.load()


def run_gen_queries_llm(cfg: GenQueriesCfg | None = None) -> pl.DataFrame:
    global gen_queries_cfg
    if cfg is not None:
        gen_queries_cfg = cfg

    prompts_df = pl.read_parquet(get_table_path('queries_prompts'))

    out_path = get_table_path('queries')
    resume_df = None
    if out_path.exists():
        resume_df = pl.read_parquet(out_path)
        print(f'Resuming: {len(resume_df):,} queries already generated')

    if resume_df is not None and 'stratum' not in resume_df.columns:
        resume_df = resume_df.join(
            prompts_df.select('icd10_3char', 'modifiers_json', 'stratum'),
            on=['icd10_3char', 'modifiers_json'],
            how='left',
        )

    results: list[dict] = resume_df.to_dicts() if resume_df is not None else []

    for new_count, curr_query in enumerate(query_generator(prompts_df, resume_df), start=1):
        results.append(curr_query)
        if new_count % gen_queries_cfg.save_every == 0:
            pl.DataFrame(results).write_parquet(out_path)

    df = pl.DataFrame(results)
    df.write_parquet(out_path)

    return df


def query_generator(
    prompts_df: pl.DataFrame,
    resume_df: pl.DataFrame | None = None,
):
    already_done: set[tuple[str, str]] = set()
    if resume_df is not None:
        for done_row in resume_df.iter_rows(named=True):
            done_row = cast(QueryRow, done_row)
            already_done.add((done_row['icd10_3char'], done_row['modifiers_json']))

    for i, row in enumerate(
        tqdm(
            prompts_df.iter_rows(named=True),
            total=len(prompts_df),
            desc='Generating queries',
            file=sys.stderr,
        )
    ):
        row = cast(QueryPromptRow, row)
        key = (row['icd10_3char'], row['modifiers_json'])
        if key in already_done:
            continue

        try:
            query_text = generate(
                row['full_prompt'],
                model=gen_queries_cfg.model or None,
                temperature=gen_queries_cfg.temperature,
                top_p=gen_queries_cfg.top_p,
                top_k=gen_queries_cfg.top_k,
                think=gen_queries_cfg.think,
                stream=True,
            ).strip()
        except Exception as e:
            print(f'  Error on row {i}: {e}')
            continue

        if not query_text:
            continue

        out = {k: v for k, v in row.items() if k != 'full_prompt'}
        out['query_text'] = query_text
        yield out


if __name__ == '__main__':
    setup_logging()
    from experiments.mimic.configs import load_config_from_main

    raw = load_config_from_main(key='queries')
    run_gen_queries_llm(cfg=GenQueriesCfg(**raw['gen_queries_llm']))
