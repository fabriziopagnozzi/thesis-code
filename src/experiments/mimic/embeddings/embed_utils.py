import shutil
from pathlib import Path
from typing import cast

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from lancedb import DBConnection, Table
from pyarrow import FixedSizeListArray

from experiments.mimic.global_configs import MimicPaths
from experiments.mimic.utils.utils import get_vec_col_name
from helpers.embedder import Embedder


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
    else:
        return cast(
            frozenset[str],
            frozenset(
                lance_ds
                .scanner(columns=['chunk_id'], filter=f'{vec_col} IS NOT NULL')
                .to_table()['chunk_id']
                .to_pylist()
            ),
        )


def embed_and_commit(
    chunks_df: pl.DataFrame,
    hadm_to_icd: dict[int, list[str]],
    embedder: Embedder,
    lancedb_con: DBConnection,
    table_name: str,
    commit_every: int,
    table: Table | None = None,
) -> Table:
    vec_col = get_vec_col_name(embedder.model_name)

    staging_dir = MimicPaths.experiment_dir / f'.tmp_embeddings/{embedder.model_name}'
    staging_dir.mkdir(parents=True, exist_ok=True)

    chunks_df, n_existing = _resume_previous_run(chunks_df, staging_dir)
    n = len(chunks_df)

    # ---------------------------------------------------------
    # BATCH LOOP (Save progress temporarily to disk)
    for batch_idx, start in enumerate(range(0, n, commit_every)):
        end = min(start + commit_every, n)
        batch_file = staging_dir / f'batch_{n_existing + batch_idx:06d}.parquet'

        batch_df = chunks_df.slice(start, end - start)
        vecs = embedder.embed_docs(batch_df['text_to_embed'].to_list())

        batch_v = FixedSizeListArray.from_arrays(pa.array(vecs.flatten()), vecs.shape[1])
        icd10_3char_list = [
            hadm_to_icd.get(curr_hadm_id, []) for curr_hadm_id in batch_df['hadm_id'].to_list()
        ]

        batch_pa = (
            batch_df
            .to_arrow()
            .append_column(field_=vec_col, column=batch_v)
            .append_column(
                'icd10_3char_list', pa.array(icd10_3char_list, type=pa.list_(pa.string()))
            )
        )

        pq.write_table(batch_pa, batch_file)
        print(f'  Staged {end:,}/{n:,} to temp disk')

    # ---------------------------------------------------------
    # LANCE DB COMMIT (One time mutation)
    print('All batches embedded. Applying changes to LanceDB...')
    final_pa = ds.dataset(staging_dir, format='parquet').to_table()

    if table is None:
        table = lancedb_con.create_table(table_name, data=final_pa, mode='overwrite')
    else:
        table = lancedb_con.open_table(table_name)
        existing_ids = frozenset(
            table.to_lance().to_table(columns=['chunk_id'])['chunk_id'].to_pylist()
        )

        is_new = [cid not in existing_ids for cid in final_pa['chunk_id'].to_pylist()]
        bool_mask = pa.array(is_new, type=pa.bool_())
        new_pa = final_pa.filter(bool_mask)
        upd_pa = final_pa.filter(pc.invert(bool_mask))

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
                    valid_old_vecs = existing_vecs.filter(pa.array(mask, type=pa.bool_()))
                    all_vecs_for_column = pa.concat_tables([valid_old_vecs, all_vecs_for_column])

                table.drop_columns([vec_col])
                table = lancedb_con.open_table(table_name)

            table.to_lance().merge(all_vecs_for_column, left_on='chunk_id')
            table = lancedb_con.open_table(table_name)

        # 2. ADD BRAND NEW CHUNKS
        if new_pa.num_rows > 0:
            table = lancedb_con.open_table(table_name)
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


def build_chunks_df_for_embedding(chunks: pl.DataFrame) -> pl.DataFrame:
    """Build embed texts from contextual_prefix already stored in the chunks DataFrame.
    Returns (augmented_chunks) sorted ascending by text length for efficient batching.
    """
    return (
        chunks
        .with_columns(
            text_to_embed=(
                pl.col('contextual_prefix').fill_null('')
                + pl.lit('\nExcerpt from the ')
                + pl.col('section_name')
                + pl.lit(' section of a discharge summary.\n')
                + pl.col('text')
            )
        )
        .with_columns(text_len=pl.col('text_to_embed').str.len_chars())
        .sort('text_len', descending=False)
        .select(['chunk_id', 'hadm_id', 'text_to_embed'])
    )


def _resume_previous_run(augmented_chunks_df: pl.DataFrame, staging_dir: Path):
    existing_batch_files = sorted(staging_dir.glob('batch_*.parquet'))
    already_staged_ids: frozenset[str] = frozenset()

    if existing_batch_files:
        staged_ds = ds.dataset(staging_dir, format='parquet')
        already_staged_ids = cast(
            frozenset[str],
            frozenset(staged_ds.to_table(columns=['chunk_id'])['chunk_id'].to_pylist()),
        )
        print(
            f'  Resuming from prior staged data: {len(already_staged_ids):,} chunks already '
            f'staged across {len(existing_batch_files)} file(s)'
        )
        staged_list = list(already_staged_ids)

        mask_series = ~augmented_chunks_df['chunk_id'].is_in(staged_list)
        augmented_chunks_df = augmented_chunks_df.filter(mask_series)

    return augmented_chunks_df, len(existing_batch_files)
