import shutil

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from lancedb import DBConnection, Table, connect
from pyarrow import FixedSizeListArray

from experiments.mimic.configs import (
    EmbedCfg,
    get_table_path,
    global_cfg,
    setup_logging,
)
from experiments.mimic.utils.constants import MimicPaths
from experiments.mimic.utils.duck_db_init import connect_mimic_duckdb
from experiments.mimic.utils.utils import get_vec_col_name
from helpers.embedder import Embedder

from .add_icd_list_col import COL_NAME as ICD_LIST_COL
from .add_icd_list_col import build_hadm_to_icd
from .contextual_prefix import enrich_note_excerpts

embed_cfg = EmbedCfg.load()


def run_embed(cfg: EmbedCfg | None = None) -> None:
    global embed_cfg
    if cfg is not None:
        embed_cfg = cfg

    all_chunks = pl.read_parquet(get_table_path('chunks'))
    admissions_metadata = pl.read_parquet(get_table_path('admissions_metadata'))

    duckdb_con = connect_mimic_duckdb()
    lance_con = connect(MimicPaths.vector_db)
    table_name = global_cfg.chunks_vec_table
    table = open_vec_table(lance_con, table_name)

    (metadata_joined, chunk_texts) = enrich_note_excerpts(all_chunks, admissions_metadata)

    for model in embed_cfg.models:
        embedder = Embedder(
            model,
            **embed_cfg.model_dump(exclude={'commit_every', 'models'}),
            query_prompt=global_cfg.query_retrieval_instruction,
        )

        model_metadata = metadata_joined
        model_texts = chunk_texts

        if table is not None:
            done_ids = get_embedded_chunk_ids(table, model)

            if done_ids:
                # 2. Filter the temporary variables
                mask = ~model_metadata['chunk_id'].is_in(done_ids)
                model_metadata = model_metadata.filter(mask)
                model_texts = [
                    t for t, keep in zip(model_texts, mask.to_list(), strict=True) if keep
                ]
                print(
                    f'Resuming: {len(done_ids):,} already embedded with {model}, '
                    f'{len(model_texts):,} remaining'
                )
        # end if table is not None

        n_chunks = len(model_texts)
        if n_chunks == 0:
            print(f'Nothing to embed for {model}, all chunks already in the table.')
            continue

        hadm_to_icd = build_hadm_to_icd(duckdb_con)
        print(f'Embedding {n_chunks:,} chunks. Model: {model}')

        # 3. Pass the temporary variables into embed_and_commit
        table = embed_and_commit(
            model_metadata,
            model_texts,
            hadm_to_icd,
            embedder,
            lance_con,
            table_name,
            embed_cfg.commit_every,
            table,
        )
        print(
            f'Saved {n_chunks:,} rows to {MimicPaths.vector_db}/{table_name} for model {model}'
        )  # end for model in embed_cfg.models


def open_vec_table(db: DBConnection, table_name: str) -> Table | None:
    try:
        return db.open_table(table_name)
    except ValueError as ve:
        if str(ve).endswith('was not found'):
            return None
        raise


def get_embedded_chunk_ids(table: Table, model: str) -> frozenset[str]:
    """Returns chunk_ids that already have a non-null vector for the current model."""
    lance_ds = table.to_lance()
    vec_col = get_vec_col_name(model)

    if vec_col not in lance_ds.schema.names:
        return frozenset()
    return frozenset(
        lance_ds.scanner(columns=['chunk_id'], filter=f'{vec_col} IS NOT NULL')
        .to_table()['chunk_id']
        .to_pylist()
    )


def embed_and_commit(
    metadata_joined: pl.DataFrame,
    chunk_texts: list[str],
    hadm_to_icd: dict[int, list[str]],
    embedder: Embedder,
    db: DBConnection,
    table_name: str,
    commit_every: int,
    table: Table | None = None,
) -> Table:

    n = len(chunk_texts)
    vec_col = get_vec_col_name(embedder.model_name)

    staging_dir = MimicPaths.experiment / (f'.tmp_embeddings/{embedder.model_name}')
    staging_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # BATCH LOOP (Save progress temporarily to disk, not LanceDB)
    for start in range(0, n, commit_every):
        end = min(start + commit_every, n)
        batch_file = staging_dir / f'batch_{start}.parquet'
        if batch_file.exists():
            continue

        batch_df = metadata_joined.slice(start, end - start)
        embeddings = embedder.embed_docs(chunk_texts[start:end])
        vec_dim = embeddings.shape[1]

        batch_v = FixedSizeListArray.from_arrays(embeddings.flatten(), vec_dim)
        icd_new = [hadm_to_icd.get(h, []) for h in batch_df['hadm_id'].to_list()]

        batch_pa = (
            batch_df.to_arrow()
            .append_column(vec_col, batch_v)
            .append_column(ICD_LIST_COL, pa.array(icd_new, type=pa.list_(pa.string())))
        )

        pq.write_table(batch_pa, batch_file)
        print(f'  Staged {end:,}/{n:,} to temp disk')

    # ---------------------------------------------------------
    # LANCE DB COMMIT (One time mutation)
    print('All batches embedded. Applying changes to LanceDB...')
    # Load all staged batches into a single PyArrow dataset
    staged_ds = ds.dataset(staging_dir, format='parquet')
    final_pa = staged_ds.to_table()

    if table is None:
        table = db.create_table(table_name, data=final_pa, mode='overwrite')
    else:
        table = db.open_table(table_name)
        existing_ids = frozenset(
            table.to_lance().to_table(columns=['chunk_id'])['chunk_id'].to_pylist()
        )

        is_new = [cid not in existing_ids for cid in final_pa['chunk_id'].to_pylist()]
        new_pa = final_pa.filter(pa.array(is_new))
        upd_pa = final_pa.filter(pa.compute.invert(pa.array(is_new)))

        # 1. UPDATE EXISTING CHUNKS
        if upd_pa.num_rows > 0:
            lance_ds = table.to_lance()
            all_vecs_for_column = upd_pa.select(['chunk_id', vec_col])

            # If the column already exists, backup the old vectors before dropping it
            if vec_col in lance_ds.schema.names:
                existing_vecs = lance_ds.scanner(
                    columns=['chunk_id', vec_col], filter=f'{vec_col} IS NOT NULL'
                ).to_table()

                if existing_vecs.num_rows > 0:
                    upd_ids = set(upd_pa['chunk_id'].to_pylist())
                    mask = [cid not in upd_ids for cid in existing_vecs['chunk_id'].to_pylist()]
                    valid_old_vecs = existing_vecs.filter(pa.array(mask))
                    all_vecs_for_column = pa.concat_tables([valid_old_vecs, all_vecs_for_column])

                table.drop_columns([vec_col])
                table = db.open_table(table_name)

            table.to_lance().merge(all_vecs_for_column, left_on='chunk_id')
            table = db.open_table(table_name)

        # 2. ADD BRAND NEW CHUNKS
        if new_pa.num_rows > 0:
            table = db.open_table(table_name)
            table_schema = table.schema

            for field in table_schema:
                if field.name not in new_pa.column_names:
                    null_arr = pa.nulls(new_pa.num_rows, type=field.type)
                    new_pa = new_pa.append_column(field.name, null_arr)

            new_pa = new_pa.select(table_schema.names)
            table.add(new_pa)

    shutil.rmtree(staging_dir)
    assert table is not None
    return table


if __name__ == '__main__':
    setup_logging()
    run_embed(cfg=EmbedCfg.load())
