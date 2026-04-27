"""Compare retrieval results across vector columns on chunks_test.

Embeds a query with each model and prints top-k results side by side.
The column name for each model is derived by replacing separators (/ - .)
with underscores and prepending "vector_".

Usage:
    python compare_embeddings.py [--query "..."] [--k 5] [--table chunks_test] [--device cuda]
        [--models MODEL [MODEL ...]]
"""

import argparse

import lancedb

from experiments.mimic.configs import setup_logging
from experiments.mimic.utils.constants import MimicPaths
from experiments.mimic.utils.utils import get_vec_col_name
from helpers.embedder import Embedder

DEFAULT_MODELS = [
    'multi-qa-mpnet-base-cos-v1',
    'Simonlee711/Clinical_ModernBERT',
]

DEFAULT_QUERY = 'patient with congestive heart failure and shortness of breath'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', default=DEFAULT_QUERY)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--table', default='chunks_test')
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--models',
        nargs='+',
        default=DEFAULT_MODELS,
        metavar='MODEL',
        help='One or more model names to compare (default: %(default)s)',
    )
    args = parser.parse_args()

    models = args.models
    cols = [get_vec_col_name(m) for m in models]

    db = lancedb.connect(MimicPaths.vector_db)
    table = db.open_table(args.table)

    print(f'Query : "{args.query}"')
    print(f'Table : {args.table}  ({table.count_rows()} rows)\n')

    vecs = []
    for model, col in zip(models, cols, strict=True):
        print(f'Loading {model} (col: {col}) ...')
        emb = Embedder(model)
        vec = emb._model.encode(args.query, normalize_embeddings=True)
        vecs.append(vec)
    print()

    all_results = [
        table.search(vec.tolist(), vector_column_name=col)
        .limit(args.k)
        .select(['chunk_id', 'section_name', 'text'])
        .to_list()
        for vec, col in zip(vecs, cols, strict=True)
    ]

    sep = '=' * 80
    thin = '-' * 80

    labels = [f'[{chr(65 + i)}] {m}' for i, m in enumerate(models)]

    for rank in range(args.k):
        print(sep)
        print(f'Rank {rank + 1}')
        print(thin)
        for label, results in zip(labels, all_results, strict=True):
            row = results[rank]
            section = (row.get('section_name') or '').strip()
            text = (row.get('text') or '').strip()
            score = row.get('_distance', float('nan'))
            print(f'{label}  dist={score:.4f}  section={section}')
            print(text)
            print()

    print(sep)

    # Pairwise overlap: how many chunk_ids appear in both result lists
    if len(models) > 1:
        print('\nPairwise overlap:')
        id_sets = [{r['chunk_id'] for r in results} for results in all_results]
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                overlap = id_sets[i] & id_sets[j]
                print(f'  {chr(65 + i)} vs {chr(65 + j)}: {len(overlap)}/{args.k} chunks in common')
                if overlap:
                    print('    Shared chunk_ids:', ', '.join(sorted(overlap)))


if __name__ == '__main__':
    setup_logging()
    main()
