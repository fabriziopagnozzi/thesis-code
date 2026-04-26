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

from experiments.mimic.configs import (
    EmbedCfg,
    get_table_path,
    global_cfg,
    setup_logging,
)
from experiments.mimic.embeddings.add_icd_list_col import build_hadm_to_icd
from experiments.mimic.embeddings.embed_whole_corpus import embed_and_commit, enrich_note_excerpts
from experiments.mimic.utils.constants import MimicPaths
from experiments.mimic.utils.duck_db_init import connect_mimic_duckdb
from helpers.embedder import Embedder

embed_cfg = EmbedCfg.load()


def run_selective_embed(cfg: EmbedCfg | None = None) -> None:
    global embed_cfg
    if cfg is not None:
        embed_cfg = cfg

    queries_df = pl.read_parquet(get_table_path('queries'))
    all_chunks = pl.read_parquet(get_table_path('chunks'))

    # 1. Get selected ICD-3 codes from generated queries
    selected_codes = queries_df['icd10_3char'].unique().to_list()
    print(
        f'[selective embed] {len(selected_codes)} unique conditions from {get_table_path("queries").name}: '
        f'{selected_codes[:5]}{"..." if len(selected_codes) > 5 else ""}'
    )

    # 2. Resolve hadm_ids for selected ICD-3 codes
    con = connect_mimic_duckdb()
    condition_hadm_ids = con.execute(
        """--sql
        SELECT DISTINCT hadm_id
        FROM unified_diagnoses
        WHERE list_contains($1::VARCHAR[], LEFT(unified_icd10, 3))
    """,
        [selected_codes],
    ).pl()
    print(f'[selective embed] {len(condition_hadm_ids):,} admissions for selected conditions')

    # 3. Filter chunks to relevant admissions
    relevant_chunks = all_chunks.join(condition_hadm_ids, on='hadm_id', how='inner')
    print(
        f'[selective embed] {len(relevant_chunks):,} relevant chunks '
        f'(out of {len(all_chunks):,} total in corpus)'
    )
    del all_chunks

    # 4. Anti-join vs existing LanceDB chunk_ids
    db = connect(MimicPaths.vector_db)
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

    hadm_to_icd = build_hadm_to_icd(con)

    # 5. Enrich with contextual prefix metadata
    metadata = pl.read_parquet(get_table_path('admissions_metadata'))
    emb_model = global_cfg.embedding_model
    embedder = Embedder(
        emb_model,
        **embed_cfg.model_dump(exclude={'commit_every'}),
        query_prompt=global_cfg.query_retrieval_instruction,
    )

    metadata_joined, chunk_texts = enrich_note_excerpts(relevant_chunks, metadata)
    n_chunks = len(metadata_joined)
    print(f'[selective embed] Embedding {n_chunks:,} chunks with model: {emb_model}')

    # 6. Embed in batches and commit to LanceDB
    embed_and_commit(
        metadata_joined,
        chunk_texts,
        hadm_to_icd,
        embedder,
        db,
        table_name,
        embed_cfg.commit_every,
        table,
    )

    print(
        f'[selective embed] Done. Added {n_chunks:,} vectors to {MimicPaths.vector_db}/{table_name}'
    )


if __name__ == '__main__':
    setup_logging()
    run_selective_embed(cfg=EmbedCfg.load())
