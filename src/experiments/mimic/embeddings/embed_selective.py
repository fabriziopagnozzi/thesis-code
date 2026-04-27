"""
Generates embedding specifically for the selected queries in previous steps.

Reads queries.parquet (or queries_prompts.parquet) to identify which ICD-10 3-char
conditions have been selected, then embeds only the chunks belonging to those
conditions' admissions - skipping the full-corpus embed in embed_whole_corpus.py.

Run AFTER phase 3.1-3.2 (query generation) and BEFORE phase 3.3 (c_filter_queries.py),
which is the first step that requires vectors.
"""

import polars as pl
from lancedb import connect

from experiments.mimic.configs import (
    EmbedCfg,
    get_table_path,
    global_cfg,
    setup_logging,
)
from experiments.mimic.embeddings.add_icd_list_col import build_hadm_to_icd
from experiments.mimic.embeddings.embed_whole_corpus import (
    embed_and_commit,
    enrich_note_excerpts,
    get_embedded_chunk_ids,
    open_vec_table,
)
from experiments.mimic.utils.constants import MimicPaths
from experiments.mimic.utils.duck_db_init import connect_mimic_duckdb
from helpers.embedder import Embedder

embed_cfg = EmbedCfg.load()


def run_selective_embed(cfg: EmbedCfg | None = None) -> None:
    global embed_cfg
    if cfg is not None:
        embed_cfg = cfg

    all_chunks = pl.read_parquet(get_table_path('chunks'))
    admissions_metadata = pl.read_parquet(get_table_path('admissions_metadata'))

    # 1. Get selected ICD-3 codes from generated queries
    selected_icd_prefixes = pl.read_parquet(get_table_path('queries'))['icd10_3char'].unique()

    # 2. Resolve hadm_ids for selected ICD-3 codes
    condition_hadm_ids = (
        pl.read_parquet(get_table_path('unified_diagnoses'))
        .select(pl.col('hadm_id'), pl.col('unified_icd10').str.slice(0, 3).alias('icd10_3char'))
        .join(selected_icd_prefixes.to_frame(), on='icd10_3char', how='inner')
        .select('hadm_id')
        .unique()
    )
    print(f'[selective embed] {len(condition_hadm_ids):,} admissions for selected conditions')

    # 3. Filter chunks to relevant admissions
    relevant_chunks = all_chunks.join(condition_hadm_ids, on='hadm_id', how='inner')
    print(
        f'[selective embed] {len(relevant_chunks):,} relevant chunks (out of {len(all_chunks):,} total in corpus)'
    )
    del all_chunks

    duckdb_con = connect_mimic_duckdb()
    lance_con = connect(MimicPaths.vector_db)
    table_name = global_cfg.chunks_vec_table
    table = open_vec_table(lance_con, table_name)

    for model, batch_size in zip(embed_cfg.models, embed_cfg.batch_sizes, strict=True):
        model_chunks = relevant_chunks

        if table is not None:
            done_ids = get_embedded_chunk_ids(table, model)
            n_done = len(done_ids)
            # Filter the temporary dataframe, NOT relevant_chunks
            model_chunks = model_chunks.filter(~pl.col('chunk_id').is_in(done_ids))
            print(
                f'[selective embed] {n_done:,} already in LanceDB with model '
                f'{model}, {len(model_chunks):,} remaining to embed'
            )
        # end if table is not None

        if model_chunks.is_empty():
            print('[selective embed] All relevant chunks already embedded. Nothing to do.')
            continue

        hadm_to_icd = build_hadm_to_icd(duckdb_con)

        embedder = Embedder(
            model,
            batch_size,
            query_prompt=global_cfg.query_retrieval_instruction,
        )

        metadata_joined, chunk_texts = enrich_note_excerpts(model_chunks, admissions_metadata)
        n_chunks = len(metadata_joined)
        print(f'[selective embed] Embedding {n_chunks:,} chunks with model: {model}')

        table = embed_and_commit(
            metadata_joined,
            chunk_texts,
            hadm_to_icd,
            embedder,
            lance_con,
            table_name,
            embed_cfg.commit_every,
            table,
        )

        print(
            f'[selective embed] Done. Added {n_chunks:,} vectors to {MimicPaths.vector_db}/{table_name} for model {model}'
        )  # end for model in embed_cfg.models


if __name__ == '__main__':
    setup_logging()
    run_selective_embed(cfg=EmbedCfg.load())
