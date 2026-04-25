"""
Generates embedding specifically for the selected queries in previous steps.

Reads queries.parquet (or queries_prompts.parquet) to identify which ICD-10 3-char
conditions have been selected, then embeds only the chunks belonging to those
conditions' admissions - skipping the full-corpus embed in embed_whole_corpus.py.

Run AFTER phase 3.1-3.2 (query generation) and BEFORE phase 3.3 (c_filter_queries.py),
which is the first step that requires vectors.
"""

from typing import cast

import polars as pl
from lancedb import Table, connect
from pyarrow import FixedSizeListArray

from experiments.mimic.configs import (
    VECTOR_DB_DIR,
    EmbedCfg,
    get_table_path,
    global_cfg,
    setup_logging,
)
from experiments.mimic.duck_db_init import connect_mimic_duckdb
from experiments.mimic.phase_2_embedding.embed_whole_corpus import enrich_note_excerpts
from helpers.embedder import Embedder

embed_cfg = EmbedCfg.load()


def run_selective_embed(cfg: EmbedCfg | None = None) -> None:
    global embed_cfg
    if cfg is not None:
        embed_cfg = cfg

    # 1. Get selected ICD-3 codes from generated queries (or fall back to prompts)
    queries_path = get_table_path('queries')
    source_path = queries_path if queries_path.exists() else get_table_path('queries_prompts')
    if not source_path.exists():
        raise FileNotFoundError(
            'Neither queries.parquet nor queries_prompts.parquet found at expected paths. '
            'Run phase 3.1 (a_build_query_prompts.py) before selective embed.'
        )
    source_df = pl.read_parquet(source_path)
    selected_codes = source_df['icd10_3char'].unique().to_list()
    print(
        f'[selective embed] {len(selected_codes)} unique conditions from {source_path.name}: '
        f'{selected_codes[:5]}{"..." if len(selected_codes) > 5 else ""}'
    )

    # 2. Resolve hadm_ids via a small IN clause on ICD-3 codes (~dozens, not thousands)
    con = connect_mimic_duckdb()
    codes_sql = ', '.join(f"'{c}'" for c in selected_codes)
    condition_hadm_ids = con.execute(f"""--sql
        SELECT DISTINCT hadm_id
        FROM unified_diagnoses
        WHERE LEFT(unified_icd10, 3) IN ({codes_sql})
    """).pl()
    print(f'[selective embed] {len(condition_hadm_ids):,} admissions for selected conditions')

    # 3. Filter chunks to relevant admissions - Polars join, no SQL IN on hadm_ids
    all_chunks = pl.read_parquet(get_table_path('chunks'))
    relevant_chunks = all_chunks.join(condition_hadm_ids, on='hadm_id', how='inner')
    print(
        f'[selective embed] {len(relevant_chunks):,} relevant chunks '
        f'(out of {len(all_chunks):,} total in corpus)'
    )
    del all_chunks

    # 4. Anti-join vs existing LanceDB chunk_ids - reads one column, no vectors loaded
    db = connect(VECTOR_DB_DIR)
    table_name = global_cfg.chunks_vec_table
    table: Table | None = None
    try:
        table = db.open_table(table_name)
        existing_arrow = table.to_lance().to_table(columns=['chunk_id'])
        existing_df = cast(pl.DataFrame, pl.from_arrow(existing_arrow))
        n_existing = len(existing_df)
        relevant_chunks = relevant_chunks.join(existing_df, on='chunk_id', how='anti')
        print(
            f'[selective embed] {n_existing:,} already in LanceDB, '
            f'{len(relevant_chunks):,} remaining to embed'
        )
    except ValueError as ve:
        if not str(ve).endswith('was not found'):
            raise

    if relevant_chunks.is_empty():
        print('[selective embed] All relevant chunks already embedded. Nothing to do.')
        return

    # 5. Enrich with contextual prefix metadata
    metadata = pl.read_parquet(get_table_path('admissions_metadata'))
    emb_model = global_cfg.embedding_model
    embedder = Embedder(
        emb_model,
        **embed_cfg.model_dump(exclude={'commit_every'}),
        query_prompt=global_cfg.query_retrieval_instruction,
    )

    metadata_joined, chunk_texts = enrich_note_excerpts(relevant_chunks, metadata)
    n_chunks = len(chunk_texts)
    print(f'[selective embed] Embedding {n_chunks:,} chunks with model: {emb_model}')

    # 6. Embed in batches and commit to LanceDB (same logic as embed_whole_corpus.py)
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

        print(f'  Committed {end:,}/{n_chunks:,}')

    print(f'[selective embed] Done. Added {n_chunks:,} vectors to {VECTOR_DB_DIR}/{table_name}')


if __name__ == '__main__':
    setup_logging()
    run_selective_embed(cfg=EmbedCfg.load())
