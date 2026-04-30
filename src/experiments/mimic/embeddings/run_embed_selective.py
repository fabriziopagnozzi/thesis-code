"""
Generates embedding specifically for the selected queries in previous steps.

Reads queries.parquet (or queries_prompts.parquet) to identify which ICD-10 3-char
conditions have been selected, then embeds only the chunks belonging to those
conditions' admissions - skipping the full-corpus embed in embed_whole_corpus.py.

Run AFTER phase 3.1-3.2 (query generation) and BEFORE phase 3.3 (c_filter_queries.py),
which is the first step that requires vectors.
"""

import polars as pl
from duckdb import DuckDBPyConnection
from lancedb import connect

from experiments.mimic.embeddings.schemas_embeddings import EmbedCfg
from experiments.mimic.global_configs import (
    MimicPaths,
    duckdb_con,
    global_cfg,
    read_parquet,
    setup_logging,
)
from helpers.embedder import Embedder

from .embed_utils import build_hadm_to_icd
from .run_embed_whole_corpus import (
    embed_and_commit,
    enrich_note_excerpts,
    get_embedded_chunk_ids,
    open_vec_table,
)

embed_cfg = EmbedCfg.load()


def run_selective_embed(
    duckdb_con: DuckDBPyConnection = duckdb_con, cfg: EmbedCfg | None = None
) -> None:
    global embed_cfg
    if cfg is not None:
        embed_cfg = cfg

    lance_con = connect(MimicPaths.vector_db_dir)
    table_name = global_cfg.chunks_vec_table
    table = open_vec_table(lance_con, table_name)

    all_chunks = read_parquet('chunks')
    admissions_metadata = read_parquet('admissions_metadata')
    queries_icd_prefixes = read_parquet('queries')['icd10_3char'].unique()
    queries_hadm_ids = (
        read_parquet('unified_diagnoses')
        .select(pl.col('hadm_id'), pl.col('unified_icd10').str.slice(0, 3).alias('icd10_3char'))
        .join(queries_icd_prefixes.to_frame(), on='icd10_3char', how='inner')
        .select('hadm_id')
        .unique()
    )
    chunks_for_queries = all_chunks.join(queries_hadm_ids, on='hadm_id', how='inner')

    print(
        f'{len(queries_hadm_ids):,} admissions for the selected conditions in the generated queries.'
    )
    print(
        f'{len(chunks_for_queries):,} relevant chunks (out of {len(all_chunks):,} total in corpus)'
    )
    del all_chunks

    for model, batch_size in zip(embed_cfg.models, embed_cfg.batch_sizes, strict=True):
        filtered_chunks_for_model: pl.DataFrame = pl.DataFrame()

        if table is not None:
            done_ids = get_embedded_chunk_ids(table, model)
            n_done = len(done_ids)
            # Filter the temporary dataframe, NOT relevant_chunks
            filtered_chunks_for_model = chunks_for_queries.filter(
                ~pl.col('chunk_id').is_in(done_ids)
            )
            print(
                f'{n_done:,} already in LanceDB with model '
                f'{model}, {len(filtered_chunks_for_model):,} remaining to embed'
            )
        # end if table is not None

        if filtered_chunks_for_model.is_empty():
            print('All relevant chunks already embedded. Nothing to do.')
            continue

        hadm_to_icd = build_hadm_to_icd(duckdb_con)

        metadata_joined, chunk_texts = enrich_note_excerpts(
            filtered_chunks_for_model, admissions_metadata
        )
        n_chunks = len(metadata_joined)
        print(f'Embedding {n_chunks:,} chunks with model: {model}')

        embedder = Embedder(
            model,
            batch_size,
            query_prompt=global_cfg.query_retrieval_instruction,
        )
        try:
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
                f'Done. Added {n_chunks:,} vectors to {MimicPaths.vector_db_dir}/{table_name} for model {model}'
            )
        finally:
            embedder.release()


if __name__ == '__main__':
    setup_logging()
    run_selective_embed()
