from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from experiments.mimic.pool_analysis.schemas_pool_analysis import PoolAnalysisCfg
from experiments.mimic.queries.schemas_queries import QueryModifier, QueryRow
from experiments.mimic.utils.chunk_pools import ChunkPool, ChunkPoolBuilder


@dataclass
class QueryPool:
    query_id: int
    icd10_3char: str
    stratum: int
    query_text: str
    modifier_labels: list[str]
    pool: ChunkPool
    query_vec: NDArray[np.float32]
    facet_onehot: NDArray[np.bool_]
    facet_combined: NDArray[np.object_]


def iter_query_pools(
    queries_df: pl.DataFrame,
    builder: ChunkPoolBuilder,
    cfg: PoolAnalysisCfg,
) -> Iterator[QueryPool]:
    for row in queries_df.iter_rows(named=True):
        row = cast(QueryRow, row)
        modifiers: list[QueryModifier] = QueryModifier.parse_list(
            row.get('modifiers_json', '') or ''
        )
        if not modifiers:
            continue

        query_vec = builder.embed_query(row['query_text'])
        pool = builder.topk_cosine_for_condition(query_vec, row['icd10_3char'], k=cfg.pool_n)
        if pool.n < 5:
            continue

        labels, onehot, combined = _build_facet_arrays(pool, row['icd10_3char'], modifiers, builder)
        if not onehot.any():
            continue

        yield QueryPool(
            query_id=int(row['query_id']),
            icd10_3char=row['icd10_3char'],
            stratum=row.get('stratum'),
            query_text=row['query_text'],
            modifier_labels=labels,
            pool=pool,
            query_vec=query_vec,
            facet_onehot=onehot,
            facet_combined=combined,
        )


def _build_facet_arrays(
    pool: ChunkPool,
    icd10_3char: str,
    modifiers: list[QueryModifier],
    builder: ChunkPoolBuilder,
) -> tuple[list[str], NDArray[np.bool_], NDArray[np.object_]]:
    labels: list[str] = []
    qualifying: list[set[int]] = []
    for mod in modifiers:
        labels.append(mod.label)
        qualifying.append(builder.get_hadm_ids_by_condition_modifier(icd10_3char, mod))

    n_chunks, n_mods = pool.n, len(labels)
    onehot = np.zeros((n_chunks, n_mods), dtype=bool)
    hadm_list = pool.hadm_ids.tolist()
    for j, qset in enumerate(qualifying):
        for i, hid in enumerate(hadm_list):
            if hid in qset:
                onehot[i, j] = True

    combined = np.empty(n_chunks, dtype=object)
    for i in range(n_chunks):
        flags = onehot[i]
        if not flags.any():
            combined[i] = 'neither'
        elif flags.all() and n_mods > 1:
            combined[i] = 'both'
        else:
            j = int(np.argmax(flags))
            combined[i] = f'{labels[j]}_only'
    return labels, onehot, combined
