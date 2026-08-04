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
from experiments.mimic.utils.utils import build_hadm_to_icd
from helpers.embedder import Embedder

from .embed_utils import (
    build_chunks_df_for_embedding,
    embed_and_commit,
    get_embedded_chunk_ids,
    open_vec_table,
)

embed_cfg = EmbedCfg.load()


def run_embed(duckdb_con: DuckDBPyConnection = duckdb_con, cfg: EmbedCfg | None = None) -> None:
    global embed_cfg
    if cfg is not None:
        embed_cfg = cfg

    lance_con = connect(MimicPaths.vector_db_dir)
    table_name = global_cfg.chunks_vec_table
    table = open_vec_table(lance_con, table_name)

    all_chunks = build_chunks_df_for_embedding(
        read_parquet('chunks').filter(pl.col('section_name').is_in(global_cfg.sections_filter))
    )

    for model, batch_size in zip(embed_cfg.models, embed_cfg.batch_sizes, strict=True):
        model_chunks = all_chunks

        if table is not None:
            done_ids = get_embedded_chunk_ids(table, model)

            if done_ids:
                model_chunks = model_chunks.filter(~model_chunks['chunk_id'].is_in(done_ids))
                print(
                    f'Resuming: {len(done_ids):,} already embedded with {model}, '
                    f'{len(model_chunks):,} remaining'
                )
        # end if table is not None

        n_chunks = len(model_chunks)
        if n_chunks == 0:
            print(f'Nothing to embed for {model}, all chunks already in the table.')
            continue

        hadm_to_icd = build_hadm_to_icd(duckdb_con)
        print(f'Embedding {n_chunks:,} chunks. Model: {model}')

        embedder = Embedder(
            model,
            batch_size,
            query_prompt=global_cfg.query_retrieval_instruction,
        )

        try:
            table = embed_and_commit(
                model_chunks,
                hadm_to_icd,
                embedder,
                lance_con,
                table_name,
                embed_cfg.commit_every,
                table,
            )
            print(
                f'Saved {n_chunks:,} rows to {MimicPaths.vector_db_dir}/{table_name} for model {model}'
            )
        finally:
            embedder.release()


if __name__ == '__main__':
    setup_logging()
    from experiments.mimic.global_configs import load_config_from_main

    raw = load_config_from_main(key='embeddings')
    run_embed(cfg=EmbedCfg.load(**raw))
