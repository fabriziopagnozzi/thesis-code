"""Build HNSW indexes for all vector_ columns in a LanceDB table."""

import argparse

import lancedb

from experiments.mimic.configs import VECTOR_DB_DIR, setup_logging


def build_indexes(
    table_name: str = 'chunks',
    metric: str = 'cosine',
) -> None:
    db = lancedb.connect(VECTOR_DB_DIR)
    table = db.open_table(table_name)

    vector_cols = [c for c in table.schema.names if c.startswith('vector_')]
    if not vector_cols:
        print(f"No 'vector_' columns found in table '{table_name}'.")
        return

    already_indexed = {idx.columns[0] for idx in table.list_indices() if idx.columns}

    print(f'Table        : {table_name}  ({table.count_rows():,} rows)')
    print(f'Vector cols  : {vector_cols}')
    print(f'Already indexed: {sorted(already_indexed)}\n')

    for col in vector_cols:
        if col in already_indexed:
            print(f'  {col}: skipped (already indexed)')
            continue

        print(f'  {col}: building HNSW index (metric={metric}) ...', flush=True)
        table.create_index(
            metric=metric,
            vector_column_name=col,
            index_type='IVF_HNSW_SQ',
            replace=False,
        )
        print(f'  {col}: done')

    print('\nAll done.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', default='chunks', help='LanceDB table name (default: chunks)')
    parser.add_argument(
        '--metric',
        default='cosine',
        choices=['cosine', 'l2', 'dot'],
        help='Distance metric (default: cosine)',
    )
    args = parser.parse_args()

    build_indexes(table_name=args.table, metric=args.metric)


if __name__ == '__main__':
    setup_logging()
    main()
