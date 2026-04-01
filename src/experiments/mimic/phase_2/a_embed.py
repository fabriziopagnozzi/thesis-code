"""Step 2.1: Build contextual embedding prefixes and generate embeddings."""

import polars as pl

from experiments.mimic.duck_db_init import MIMIC_RESULTS_DIR
from helpers.embedder import Embedder

MODEL_NAME = 'multi-qa-mpnet-base-cos-v1'
BATCH_SIZE = 256
COMMIT_EVERY = 16384


def _age_group(age: float | None) -> str:
    if age is None:
        return 'unknown age'
    if age < 30:
        return 'young adult'
    if age < 50:
        return 'middle-aged'
    if age < 65:
        return 'older adult'
    if age < 80:
        return 'elderly'
    return 'very elderly'


def build_contextual_prefix(meta_row: dict) -> str:
    age_grp = _age_group(meta_row.get('age'))
    gender = meta_row.get('gender', 'unknown')
    gender_label = 'female' if gender == 'F' else 'male' if gender == 'M' else 'unknown gender'
    race = meta_row.get('race', 'unknown')
    primary_dx = meta_row.get('primary_icd_description', 'unknown condition')
    chief_complaint = meta_row.get('chief_complaint')
    top_icds = meta_row.get('top_icd_descriptions', '')

    prefix = f'{age_grp.capitalize()} {gender_label} ({race}) admitted for {primary_dx}.'
    if chief_complaint:
        prefix += f'\nChief complaint: {chief_complaint}.'
    if top_icds:
        prefix += f'\nComorbidities: {top_icds}.'
    return prefix


def prepare_texts(
    chunks: pl.DataFrame,
    metadata: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str]]:
    meta_cols = [
        'hadm_id',
        'age',
        'gender',
        'race',
        'primary_icd_description',
        'top_icd_descriptions',
    ]
    meta_subset = metadata.select(meta_cols).unique(subset=['hadm_id'])
    joined = chunks.join(meta_subset, on='hadm_id', how='left')

    texts: list[str] = []
    for row in joined.iter_rows(named=True):
        prefix = build_contextual_prefix(row)
        section = row['section_name']
        subsection = row.get('subsection_name')
        section_label = f'{section} > {subsection}' if subsection else section
        texts.append(f'{prefix}\nSection: {section_label}.\n{row["text"]}')

    return joined, texts


def main(device: str = 'cpu'):
    import lancedb

    MIMIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    chunks = pl.read_parquet(MIMIC_RESULTS_DIR / 'chunks.parquet')
    metadata = pl.read_parquet(MIMIC_RESULTS_DIR / 'admissions_metadata.parquet')

    joined, texts = prepare_texts(chunks, metadata)
    print(f'{len(texts):,} texts prepared. Sample:\n  {texts[0][:200]}...\n')

    print(f'Embedding with {MODEL_NAME} (committing every {COMMIT_EVERY:,} chunks)...')
    embedder = Embedder(MODEL_NAME, device=device, batch_size=BATCH_SIZE)
    db = lancedb.connect(MIMIC_RESULTS_DIR / '_lancedb')
    table = None
    total = len(texts)

    for start in range(0, total, COMMIT_EVERY):
        end = min(start + COMMIT_EVERY, total)
        batch_texts = texts[start:end]
        batch_df = joined.slice(start, end - start)

        embeddings = embedder.embed_corpus(batch_texts)
        batch_df = batch_df.with_columns(pl.Series('vector', embeddings.tolist()))

        if table is None:
            table = db.create_table('chunks', data=batch_df.to_arrow(), mode='overwrite')
        else:
            table.add(batch_df.to_arrow())

        print(f'  Committed {end:,}/{total:,} chunks')

    print(f'Saved {total:,} rows to {MIMIC_RESULTS_DIR}/_lancedb/chunks')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', help='torch device (e.g. cuda, cpu)')
    args = parser.parse_args()
    main(device=args.device)
