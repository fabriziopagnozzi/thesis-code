"""Add a new embedding vector column to an existing LanceDB chunks table.

Reads text chunks from the table in batches, re-embeds them with the specified
model, and writes the result as a new column alongside the existing data.
"""

import argparse

import lancedb
import pyarrow as pa

from experiments.mimic.configs import global_cfg, setup_logging
from experiments.mimic.utils.constants import MimicPaths
from helpers.embedder import Embedder


def add_vector_column(
    model_name: str,
    col_name: str | None = None,
    batch_size: int = 256,
    device: str = 'cpu',
    table_name: str = global_cfg.chunks_vec_table,
    text_col: str = 'text',
) -> None:
    if col_name is None:
        # e.g. "BAAI/bge-small-en-v1.5" -> "vector_bge_small_en_v1_5"
        safe = model_name.replace('/', '_').replace('-', '_').replace('.', '_')
        col_name = f'vector_{safe}'

    db = lancedb.connect(MimicPaths.vector_db)
    table = db.open_table(table_name)

    existing_cols = table.schema.names
    if col_name in existing_cols:
        raise ValueError(
            f"Column '{col_name}' already exists in table '{table_name}'. "
            'Choose a different --col name or drop the column first.'
        )
    if text_col not in existing_cols:
        raise ValueError(
            f"Text column '{text_col}' not found in table '{table_name}'. "
            f'Available columns: {existing_cols}'
        )

    total = table.count_rows()
    print(f"Table '{table_name}': {total:,} rows")
    print(f'Embedding model : {model_name}')
    print(f'New column      : {col_name}')
    print(f'Batch size      : {batch_size}')
    print(f'Device          : {device}')

    embedder = Embedder(model_name, batch_size=batch_size)

    all_embeddings: list[list[float]] = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = table.search().limit(batch_size).offset(start).select([text_col]).to_arrow()
        texts: list[str] = batch[text_col].to_pylist()

        embs = embedder.embed_docs(texts)
        all_embeddings.extend(embs.tolist())

        print(f'  Embedded {end:,}/{total:,} rows', flush=True)

    emb_array = pa.array(
        all_embeddings,
        type=pa.list_(pa.float32(), embedder.dim),
    )

    print('Writing column to table...')
    full_data = table.to_arrow()
    new_data = full_data.append_column(
        pa.field(col_name, pa.list_(pa.float32(), embedder.dim)),
        pa.chunked_array([emb_array]),
    )
    db.create_table(f'{table_name}', new_data, mode='overwrite')

    print(f"\nDone. Column '{col_name}' ({embedder.dim}-dim) added to '{table_name}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Sentence-Transformers model name or path')
    parser.add_argument(
        '--col',
        default=None,
        help='Name for the new vector column (default: derived from model name)',
    )
    parser.add_argument(
        '--batch-size', type=int, default=256, help='Embedding batch size (default: 256)'
    )
    parser.add_argument(
        '--device', default='cuda', help='Torch device: cpu / cuda / mps (default: cuda)'
    )
    parser.add_argument('--table', default='chunks', help='LanceDB table name (default: chunks)')
    parser.add_argument(
        '--text-col',
        default='text',
        help='Column containing the text to embed (default: text)',
    )
    args = parser.parse_args()

    add_vector_column(
        model_name=args.model,
        col_name=args.col,
        batch_size=args.batch_size,
        device=args.device,
        table_name=args.table,
        text_col=args.text_col,
    )


if __name__ == '__main__':
    setup_logging()
    main()
