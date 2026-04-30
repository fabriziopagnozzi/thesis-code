"""Build HNSW indexes for all "vector_" columns in a LanceDB table."""

import argparse

import lancedb

from experiments.mimic.global_configs import (
    MimicPaths,
    setup_logging,
)


def build_indexes(
    table_name: str,
    metric: str = 'cosine',
    db_dir_override: str | None = None,
) -> None:
    dir = (
        MimicPaths.vector_db_dir / db_dir_override if db_dir_override else MimicPaths.vector_db_dir
    )
    db = lancedb.connect(dir)
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


if __name__ == '__main__':
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vec_dir', help='Vector DB directory override')
    parser.add_argument('--table', help='LanceDB table name (default: chunks)')
    parser.add_argument(
        '--metric',
        choices=['cosine', 'l2', 'dot'],
        default='cosine',
        help='Distance metric (default: cosine)',
    )
    args = parser.parse_args()

    build_indexes(
        table_name=args.table,
        metric=args.metric,
        db_dir_override=args.vec_dir,
    )
