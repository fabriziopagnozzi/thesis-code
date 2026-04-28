import shutil
from pathlib import Path
from typing import cast

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from lancedb import DBConnection, Table, connect
from pyarrow import FixedSizeListArray

from experiments.mimic.configs import (
    EmbedCfg,
    global_cfg,
    read_parquet,
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

    duckdb_con = connect_mimic_duckdb()
    lance_con = connect(MimicPaths.vector_db)
    table_name = global_cfg.chunks_vec_table
    table = open_vec_table(lance_con, table_name)

    all_chunks = read_parquet('chunks')
    admissions_metadata = read_parquet('admissions_metadata')

    (metadata_adm_joined, chunk_texts) = enrich_note_excerpts(all_chunks, admissions_metadata)

    for model, batch_size in zip(embed_cfg.models, embed_cfg.batch_sizes, strict=True):
        model_admissions_metadata = pl.DataFrame()
        model_texts = chunk_texts

        if table is not None:
            done_ids = get_embedded_chunk_ids(table, model)

            if done_ids:
                mask = ~metadata_adm_joined['chunk_id'].is_in(done_ids)
                model_admissions_metadata = metadata_adm_joined.filter(mask)
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

        embedder = Embedder(
            model,
            batch_size,
            query_prompt=global_cfg.query_retrieval_instruction,
        )

        try:
            table = embed_and_commit(
                model_admissions_metadata,
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
            )
        finally:
            embedder.release()


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
    return cast(
        frozenset[str],
        frozenset(
            lance_ds.scanner(columns=['chunk_id'], filter=f'{vec_col} IS NOT NULL')
            .to_table()['chunk_id']
            .to_pylist()
        ),
    )


def embed_and_commit(
    admissions_metadata: pl.DataFrame,
    chunk_texts: list[str],
    hadm_to_icd: dict[int, list[str]],
    embedder: Embedder,
    db: DBConnection,
    table_name: str,
    commit_every: int,
    table: Table | None = None,
) -> Table:
    vec_col = get_vec_col_name(embedder.model_name)

    staging_dir = MimicPaths.experiment / f'.tmp_embeddings/{embedder.model_name}'
    staging_dir.mkdir(parents=True, exist_ok=True)

    admissions_metadata, chunk_texts, n_existing = resume_previous_run(
        admissions_metadata, chunk_texts, staging_dir
    )
    n = len(chunk_texts)

    # ---------------------------------------------------------
    # BATCH LOOP (Save progress temporarily to disk)
    for batch_idx, start in enumerate(range(0, n, commit_every)):
        end = min(start + commit_every, n)
        batch_file = staging_dir / f'batch_{n_existing + batch_idx:06d}.parquet'

        batch_df = admissions_metadata.slice(start, end - start)
        embeddings = embedder.embed_docs(chunk_texts[start:end])
        vec_dim = embeddings.shape[1]

        batch_v = FixedSizeListArray.from_arrays(pa.array(embeddings.flatten()), vec_dim)
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


def resume_previous_run(
    admissions_metadata: pl.DataFrame, chunk_texts: list[str], staging_dir: Path
):
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
        mask_series = ~admissions_metadata['chunk_id'].is_in(staged_list)

        admissions_metadata = admissions_metadata.filter(mask_series)
        keep_list = mask_series.to_list()
        chunk_texts = [t for t, keep in zip(chunk_texts, keep_list, strict=True) if keep]

    return admissions_metadata, chunk_texts, len(existing_batch_files)


if __name__ == '__main__':
    setup_logging()
    run_embed(cfg=EmbedCfg.load())
