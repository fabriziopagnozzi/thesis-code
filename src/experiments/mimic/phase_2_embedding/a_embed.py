import polars as pl
from lancedb import connect

from experiments.mimic.configs import (
    VECTOR_DB_DIR,
    EmbedCfg,
    get_parquet_path,
    global_cfg,
    setup_logging,
)
from helpers.embedder import Embedder

embed_cfg = EmbedCfg.load()


def run_embed(cfg: EmbedCfg | None = None) -> None:
    global embed_cfg
    if cfg is not None:
        embed_cfg = cfg

    chunks = pl.read_parquet(get_parquet_path('chunks'))
    metadata = pl.read_parquet(get_parquet_path('admissions_metadata'))
    emb_model = global_cfg.embedding_model

    print(f'Embedding full corpus: {len(chunks):,} chunks. Model: {emb_model}')

    embedder = Embedder(emb_model, **embed_cfg.model_dump(exclude={'commit_every'}))
    db = connect(VECTOR_DB_DIR)
    table = None
    (metadata_joined, chunk_texts) = prepare_texts(chunks, metadata)
    n_chunks = len(chunk_texts)

    for start in range(0, n_chunks, embed_cfg.commit_every):
        end = min(start + embed_cfg.commit_every, n_chunks)
        batch_df = metadata_joined.slice(start, end - start)

        embeddings = embedder.embed_corpus(chunk_texts[start:end])
        batch_df = batch_df.with_columns(pl.Series(global_cfg.vector_column, embeddings.tolist()))

        if table is None:
            table = db.create_table('chunks', data=batch_df.to_arrow(), mode='overwrite')
        else:
            table.add(batch_df.to_arrow())

        print(f'  Committed {end:,}/{n_chunks:,} chunks')
    # END for start in range(0, n_chunks, embed_cfg.commit_every)

    print(f'Saved {n_chunks:,} rows to {VECTOR_DB_DIR}/chunks')


def prepare_texts(chunks: pl.DataFrame, metadata: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
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
        texts.append(f'{prefix}\nSection: {row["section_name"]}.\n{row["text"]}')

    return joined, texts


def build_contextual_prefix(meta_row: dict) -> str:
    age_grp = get_age_group(meta_row.get('age'))
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


def get_age_group(age: float | None) -> str:
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


if __name__ == '__main__':
    setup_logging()
    run_embed(cfg=EmbedCfg.load())
