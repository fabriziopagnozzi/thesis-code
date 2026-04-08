"""Step 2.1: Build contextual embedding prefixes and generate embeddings."""

import polars as pl

from experiments.mimic.configs import MIMIC_RESULTS_DIR, VECTOR_DB_DIR, EmbedCfg, global_cfg
from experiments.mimic.duck_db_init import connect_mimic_duckdb
from helpers.embedder import Embedder

embed_cfg = EmbedCfg.load()


def run_embed(cfg: EmbedCfg | None = None) -> None:
    import lancedb

    global embed_cfg
    if cfg is not None:
        embed_cfg = cfg

    MIMIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    chunks = pl.read_parquet(MIMIC_RESULTS_DIR / 'chunks.parquet')
    metadata = pl.read_parquet(MIMIC_RESULTS_DIR / 'admissions_metadata.parquet')

    relevant = _relevant_hadm_ids()
    n_before = len(chunks)
    chunks = chunks.filter(pl.col('hadm_id').is_in(relevant))
    print(
        f'Filtered to {len(chunks):,}/{n_before:,} chunks ({global_cfg.num_conditions} conditions)'
    )

    joined, texts = prepare_texts(chunks, metadata)
    print(f'{len(texts):,} texts prepared. Sample:\n  {texts[0][:200]}...\n')

    print(
        f'Embedding with {embed_cfg.model_name} (committing every {embed_cfg.commit_every:,} chunks)...'
    )
    embedder = Embedder(
        embed_cfg.model_name, device=embed_cfg.device, batch_size=embed_cfg.batch_size
    )
    db = lancedb.connect(VECTOR_DB_DIR)
    table = None
    total = len(texts)

    for start in range(0, total, embed_cfg.commit_every):
        end = min(start + embed_cfg.commit_every, total)
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


def _relevant_hadm_ids() -> set[int]:
    """Return hadm_ids that have any ICD-10 diagnosis matching a top condition."""
    conditions = pl.read_parquet(MIMIC_RESULTS_DIR / 'conditions_stats.parquet')
    top_icd3 = conditions.head(global_cfg.num_conditions)['icd10_3char'].to_list()
    placeholders = ','.join(f"'{c}'" for c in top_icd3)

    con = connect_mimic_duckdb()
    rows = con.execute(f"""--sql
        SELECT DISTINCT diagnoses_icd.hadm_id
        FROM diagnoses_icd
        WHERE diagnoses_icd.icd_version = 10
        AND SUBSTR(diagnoses_icd.icd_code, 1, 3) IN ({placeholders})
    """).fetchall()
    return {r[0] for r in rows}


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


if __name__ == '__main__':
    from experiments.mimic.configs import load_config_from_main

    raw = load_config_from_main(phase=2)
    run_embed(cfg=EmbedCfg(**raw['embed']))
