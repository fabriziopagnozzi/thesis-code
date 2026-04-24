from typing import cast

import polars as pl
from lancedb import Table, connect
from pyarrow import FixedSizeListArray

from experiments.mimic.configs import (
    VECTOR_DB_DIR,
    EmbedCfg,
    get_parquet_path,
    global_cfg,
    setup_logging,
)
from experiments.mimic.schemas import EmbedJoinedRow
from experiments.mimic.utils import CHARLSON_LABELS, get_age_group, get_charlson_conditions
from helpers.embedder import Embedder

embed_cfg = EmbedCfg.load()


def run_embed(cfg: EmbedCfg | None = None) -> None:
    global embed_cfg
    if cfg is not None:
        embed_cfg = cfg

    chunks = pl.read_parquet(get_parquet_path('chunks'))
    metadata = pl.read_parquet(get_parquet_path('admissions_metadata'))
    emb_model = global_cfg.embedding_model
    embedder = Embedder(emb_model, **embed_cfg.model_dump(exclude={'commit_every'}))
    db = connect(VECTOR_DB_DIR)
    table_name = global_cfg.chunks_vec_table

    table: Table | None = None
    try:
        table = db.open_table(table_name)
    except ValueError as ve:
        if not str(ve).endswith('was not found'):
            raise

    (metadata_joined, chunk_texts) = enrich_note_excerpts(chunks, metadata)

    if table is not None:
        existing_ids: set[str] = set(
            table.to_lance().to_table(columns=['chunk_id'])['chunk_id'].to_pylist()
        )
        if existing_ids:
            mask = ~metadata_joined['chunk_id'].is_in(existing_ids)
            indices = [i for i, keep in enumerate(mask.to_list()) if keep]
            metadata_joined = metadata_joined.filter(mask)
            chunk_texts = [chunk_texts[i] for i in indices]
            print(
                f'Resuming: {len(existing_ids):,} already embedded, {len(chunk_texts):,} remaining'
            )

    n_chunks = len(chunk_texts)
    if n_chunks == 0:
        print('Nothing to embed, all chunks already in the table.')
        return

    print(f'Embedding {n_chunks:,} chunks. Model: {emb_model}')

    for start in range(0, n_chunks, embed_cfg.commit_every):
        end = min(start + embed_cfg.commit_every, n_chunks)
        batch_df = metadata_joined.slice(start, end - start)

        embeddings = embedder.embed_corpus(chunk_texts[start:end])
        v_data = FixedSizeListArray.from_arrays(embeddings.flatten(), embeddings.shape[1])
        batch_table = batch_df.to_arrow().append_column(global_cfg.vector_column, v_data)

        if table is None:
            table = db.create_table(table_name, data=batch_table, mode='overwrite')
        else:
            table.add(batch_table)

        print(f'  Committed {end:,}/{n_chunks:,} chunks')

    print(f'Saved {n_chunks:,} rows to {VECTOR_DB_DIR}/{table_name}')


def enrich_note_excerpts(
    chunks: pl.DataFrame, metadata: pl.DataFrame
) -> tuple[pl.DataFrame, list[str]]:
    meta_cols = [
        'hadm_id',
        'age',
        'gender',
        'race',
        'primary_icd_description',
        'top_icd_descriptions',
        'charlson_comorbidity_index',
        'admission_type',
        *CHARLSON_LABELS.keys(),
    ]
    meta_subset = metadata.select(meta_cols).unique(subset=['hadm_id'])
    joined = chunks.join(meta_subset, on='hadm_id', how='left')

    texts: list[str] = []
    for row in joined.iter_rows(named=True):
        row = cast(EmbedJoinedRow, row)
        prefix = build_contextual_prefix(row)
        texts.append(
            f'{prefix}\nExcerpt from the {row["section_name"]} section of a discharge summary.\n{row["text"]}'
        )

    return joined, texts


def build_contextual_prefix(meta_row: EmbedJoinedRow) -> str:
    age = meta_row.get('age')
    if age is not None:
        age_grp = get_age_group(age)
        article = 'an' if age_grp[0] in 'aeiou' else 'a'
        age_part = f'{article} {age_grp} {int(age)}-year-old'
    else:
        age_part = 'a'

    gender = meta_row.get('gender', '')
    if gender == 'F':
        gender_noun, pronoun = 'woman', 'She'
    elif gender == 'M':
        gender_noun, pronoun = 'man', 'He'
    else:
        gender_noun, pronoun = 'patient', 'The patient'

    race = meta_row.get('race', 'unknown')
    primary_dx = meta_row.get('primary_icd_description', 'unknown condition')
    adverb = _admission_adverb(meta_row.get('admission_type'))

    prefix = f'The patient is {age_part} {gender_noun} ({race}), admitted{adverb} for {primary_dx}.'

    chief_complaint = meta_row.get('chief_complaint')
    if chief_complaint and not str(chief_complaint).strip().startswith('"'):
        prefix += f'\nChief complaint: {chief_complaint}.'

    conditions = get_charlson_conditions(meta_row)
    if conditions:
        if len(conditions) == 1:
            cond_str = conditions[0]
        else:
            cond_str = ', '.join(conditions[:-1]) + f', and {conditions[-1]}'
        prefix += f'\n{pronoun} has a history of {cond_str}.'
    else:
        prefix += '\nNo significant chronic comorbidities are recorded.'

    top_icds = meta_row.get('top_icd_descriptions', '')
    if top_icds:
        prefix += f'\nAdditional co-diagnoses from this admission: {top_icds}.'

    return prefix


def _admission_adverb(admission_type: str | None) -> str:
    if not admission_type:
        return ''
    t = admission_type.upper()
    if 'EMER' in t:
        return ' emergently'
    if 'ELECTIVE' in t:
        return ' electively'
    if 'URGENT' in t:
        return ' urgently'
    return ''


if __name__ == '__main__':
    setup_logging()
    run_embed(cfg=EmbedCfg.load())
