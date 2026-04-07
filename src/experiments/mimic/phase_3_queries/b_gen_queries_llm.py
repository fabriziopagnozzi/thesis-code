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

from experiments.mimic.config_loader import load_phase_config
from experiments.mimic.duck_db_init import MIMIC_RESULTS_DIR
from helpers.ollama_client import generate

_cfg = load_phase_config(3)['gen_queries_llm']
SAVE_EVERY: int = _cfg['save_every']


def main():
    prompts_df = pl.read_parquet(MIMIC_RESULTS_DIR / 'queries_prompts.parquet')
    print(f'Loaded {len(prompts_df):,} prompts')

    # Resume from previous run if available
    out_path = MIMIC_RESULTS_DIR / 'queries.parquet'
    resume_df = None
    if out_path.exists():
        resume_df = pl.read_parquet(out_path)
        print(f'Resuming: {len(resume_df):,} queries already generated')

    df = generate_queries(prompts_df, resume_df)

    print(f'\nSaved {len(df):,} queries to {out_path}')
    print(f'  Conditions: {df["icd10_3char"].n_unique()}')
    print('\n--- Sample query ---')
    print(df['query_text'][0])


def generate_queries(
    prompts_df: pl.DataFrame, resume_df: pl.DataFrame | None = None
) -> pl.DataFrame:
    """For each prompt row, call ollama to generate the clinical question.

    If resume_df is provided, skip rows already present (by matching on
    icd10_3char + modifier_text + persona).
    """
    already_done: set[tuple] = set()
    if resume_df is not None:
        for row in resume_df.iter_rows(named=True):
            already_done.add((row['icd10_3char'], row['modifier_text'], row['persona']))

    results: list[dict] = []
    if resume_df is not None:
        results = resume_df.to_dicts()

    out_path = MIMIC_RESULTS_DIR / 'queries.parquet'

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
                temperature=_cfg['temperature'],
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
        results.append(result)

        if len(results) % SAVE_EVERY == 0:
            pl.DataFrame(results).write_parquet(out_path)
            print(f'  Checkpoint: {len(results)} queries saved')

    df = pl.DataFrame(results)
    df.write_parquet(out_path)
    return df


if __name__ == '__main__':
    import argparse

    from experiments.mimic.config_loader import parse_config_arg

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.parse_args()

    _run_cfg = parse_config_arg(3)['gen_queries_llm']
    SAVE_EVERY = _run_cfg['save_every']

    main()
