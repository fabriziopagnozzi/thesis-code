"""
Step 3.2: Generate actual clinical questions from prompts via ollama.

Reads queries_prompts.parquet (which contains LLM prompts but not the generated
query text) and calls a local LLM to produce the clinical question for each row.

Output: queries.parquet - all original columns + query_text.
If queries.parquet is already existing, it appends the new queries there.
"""

import sys

import polars as pl
from tqdm import tqdm

from experiments.mimic.config_loader import load_config
from experiments.mimic.duck_db_init import MIMIC_RESULTS_DIR
from helpers.ollama_client import generate

_cfg = load_config(3)['gen_queries_llm']
SAVE_EVERY: int = _cfg['save_every']


def run_gen_queries(cfg: dict | None = None) -> pl.DataFrame:
    global _cfg, SAVE_EVERY
    if cfg is not None:
        _cfg = cfg
        SAVE_EVERY = _cfg['save_every']

    prompts_df = pl.read_parquet(MIMIC_RESULTS_DIR / 'queries_prompts.parquet')
    print(f'Loaded {len(prompts_df):,} prompts')

    out_path = MIMIC_RESULTS_DIR / 'queries.parquet'
    resume_df = None
    if out_path.exists():
        resume_df = pl.read_parquet(out_path)
        print(f'Resuming: {len(resume_df):,} queries already generated')

    results: list[dict] = resume_df.to_dicts() if resume_df is not None else []
    for new_result in generate_queries(prompts_df, resume_df):
        results.append(new_result)
        if len(results) % SAVE_EVERY == 0:
            pl.DataFrame(results).write_parquet(out_path)
            print(f'  Checkpoint: {len(results)} queries saved')

    df = pl.DataFrame(results)
    df.write_parquet(out_path)

    print(
        f'\nSaved {len(df):,} queries to {out_path}\n'
        f'  Conditions: {df["icd10_3char"].n_unique()}\n'
        f'\n--- Sample query ---\n'
        f'{df["query_text"][0]}'
    )
    return df


def generate_queries(
    prompts_df: pl.DataFrame,
    resume_df: pl.DataFrame | None = None,
):
    """Yield one result dict per prompt row, skipping already-done rows.

    Already-done rows are determined by (icd10_3char, modifier_text, persona).
    Does NOT include resume_df rows in the output — caller handles that.
    """
    already_done: set[tuple] = set()
    if resume_df is not None:
        for row in resume_df.iter_rows(named=True):
            already_done.add((row['icd10_3char'], row['modifier_text'], row['persona']))

    for i, row in enumerate(
        tqdm(
            prompts_df.iter_rows(named=True),
            total=len(prompts_df),
            desc='Generating queries',
            file=sys.stderr,
        )
    ):
        key = (row['icd10_3char'], row['modifier_text'], row['persona'])
        if key in already_done:
            continue

        try:
            query_text = generate(
                row['full_prompt'],
                model=_cfg.get('model') or None,
                temperature=_cfg['temperature'],
                top_p=_cfg.get('top_p') or None,
                top_k=_cfg.get('top_k') or None,
                think=_cfg.get('think', False),
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
    from experiments.mimic.config_loader import load_config_from_main

    run_gen_queries(cfg=load_config_from_main(phase=3)['gen_queries_llm'])
