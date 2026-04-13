"""
Step 3.2: Generate actual clinical questions from prompts via ollama.

Reads queries_prompts.parquet and calls a local LLM to produce the clinical question for each row.

Output: queries.parquet - all original columns + query_text.
If queries.parquet is already existing, it appends the new queries there.
"""

import sys

import polars as pl
from tqdm import tqdm

from experiments.mimic.configs import GenQueriesCfg, get_parquet_path
from helpers.ollama_client import generate

gen_queries_cfg = GenQueriesCfg.load()


def run_gen_queries_llm(cfg: GenQueriesCfg | None = None) -> pl.DataFrame:
    global gen_queries_cfg
    if cfg is not None:
        gen_queries_cfg = cfg

    prompts_df = pl.read_parquet(get_parquet_path('queries_prompts'))

    out_path = get_parquet_path('queries')
    resume_df = None
    if out_path.exists():
        resume_df = pl.read_parquet(out_path)
        print(f'Resuming: {len(resume_df):,} queries already generated')

    results: list[dict] = resume_df.to_dicts() if resume_df is not None else []

    for curr_query in query_generator(prompts_df, resume_df):
        results.append(curr_query)
        if len(results) % gen_queries_cfg.save_every == 0:
            pl.DataFrame(results).write_parquet(out_path)

    df = pl.DataFrame(results)
    df.write_parquet(out_path)

    return df


def query_generator(
    prompts_df: pl.DataFrame,
    resume_df: pl.DataFrame | None = None,
):
    already_done: set[tuple] = set()
    if resume_df is not None:
        for row in resume_df.iter_rows(named=True):
            already_done.add((row['icd10_3char'], row['modifier_text']))

    for i, row in enumerate(
        tqdm(
            prompts_df.iter_rows(named=True),
            total=len(prompts_df),
            desc='Generating queries',
            file=sys.stderr,
        )
    ):
        key = (row['icd10_3char'], row['modifier_text'])
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

        result = {k: v for k, v in row.items() if k != 'full_prompt'}
        result['query_text'] = query_text
        yield result


if __name__ == '__main__':
    from experiments.mimic.configs import load_config_from_main

    raw = load_config_from_main(phase=3)
    run_gen_queries_llm(cfg=GenQueriesCfg(**raw['gen_queries_llm']))
