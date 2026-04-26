import polars as pl
import pyarrow as pa
from lancedb import Table, connect
from pyarrow import FixedSizeListArray

from experiments.mimic.configs import (
    EmbedCfg,
    get_table_path,
    global_cfg,
    setup_logging,
)
from experiments.mimic.utils.constants import MimicPaths
from experiments.mimic.utils.duck_db_init import connect_mimic_duckdb
from helpers.embedder import Embedder

from .add_icd_list_col import COL_NAME as ICD_LIST_COL
from .add_icd_list_col import build_hadm_to_icd
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
    db = connect(MimicPaths.vector_db)
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

    con = connect_mimic_duckdb()
    hadm_to_icd = build_hadm_to_icd(con)
    print(f'Embedding {n_chunks:,} chunks. Model: {emb_model}')

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
    print(f'Saved {n_chunks:,} rows to {MimicPaths.vector_db}/{table_name}')


def embed_and_commit(
    metadata_joined: pl.DataFrame,
    chunk_texts: list[str],
    hadm_to_icd: dict[int, list[str]],
    embedder: Embedder,
    db,
    table_name: str,
    commit_every: int,
    table: Table | None = None,
) -> Table:
    n = len(chunk_texts)
    for start in range(0, n, commit_every):
        end = min(start + commit_every, n)
        batch_df = metadata_joined.slice(start, end - start)
        embeddings = embedder.embed_corpus(chunk_texts[start:end])
        v_data = FixedSizeListArray.from_arrays(embeddings.flatten(), embeddings.shape[1])
        icd_lists = [hadm_to_icd.get(h, []) for h in batch_df['hadm_id'].to_list()]
        batch_table = (
            batch_df.to_arrow()
            .append_column(global_cfg.vector_column, v_data)
            .append_column(ICD_LIST_COL, pa.array(icd_lists, type=pa.list_(pa.string())))
        )
        if table is None:
            table = db.create_table(table_name, data=batch_table, mode='overwrite')
        else:
            table.add(batch_table)
        print(f'  Committed {end:,}/{n:,}')
    assert table is not None
    return table


if __name__ == '__main__':
    setup_logging()
    run_embed(cfg=EmbedCfg.load())
