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
from helpers.embedder import Embedder

from .contextual_prefix import enrich_note_excerpts

embed_cfg = EmbedCfg.load()


def run_embed(cfg: EmbedCfg | None = None) -> None:
    global embed_cfg
    if cfg is not None:
        embed_cfg = cfg

    chunks = pl.read_parquet(get_table_path('chunks'))
    metadata = pl.read_parquet(get_table_path('admissions_metadata'))
    emb_model = global_cfg.embedding_model
    embedder = Embedder(
        emb_model,
        **embed_cfg.model_dump(exclude={'commit_every'}),
        query_prompt=global_cfg.query_retrieval_instruction,
    )
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


if __name__ == '__main__':
    setup_logging()
    run_embed(cfg=EmbedCfg.load())
