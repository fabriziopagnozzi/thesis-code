import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from experiments.mimic.configs import PoolAnalysisCfg
from experiments.mimic.evaluation.candidate_pool import CandidatePool, CandidatePoolBuilder
from experiments.mimic.evaluation.evaluate import _load_queries_for_eval
from experiments.mimic.utils.schemas import DivergenceStatsRow
from experiments.mimic.utils.utils import modifier_to_snake_label


@dataclass
class QueryPool:
    query_id: int
    icd10_3char: str
    stratum: int | None
    query_text: str
    modifier_labels: list[str]
    pool: CandidatePool
    query_vec: NDArray[np.float32]
    facet_onehot: NDArray[np.bool_]
    facet_combined: NDArray[np.object_]


def _build_facet_arrays(
    pool: CandidatePool,
    icd10_3char: str,
    modifiers_json: list[dict],
    builder: CandidatePoolBuilder,
) -> tuple[list[str], NDArray[np.bool_], NDArray[np.object_]]:
    cond_hadms = builder.icd3_hadm_ids(icd10_3char)
    labels: list[str] = []
    qualifying: list[set[int]] = []
    for m in modifiers_json:
        labels.append(modifier_to_snake_label(m['text']))
        qualifying.append(cond_hadms & builder.modifier_hadm_ids(m['text']))

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


def iter_query_pools(
    builder: CandidatePoolBuilder,
    cfg: PoolAnalysisCfg,
    limit: int | None = None,
) -> Iterator[QueryPool]:
    queries_df = _load_queries_for_eval()
    if limit is not None:
        queries_df = queries_df.head(limit)
    print(f'[pool_loader] loaded {len(queries_df):,} queries')

    for row in queries_df.iter_rows(named=True):
        row = cast(DivergenceStatsRow, row)
        modifiers = json.loads(row.get('modifiers_json', '') or '[]')
        if not modifiers:
            continue

        query_vec = builder.embed_query(row['query_text'])
        pool = builder.for_query_cosine_condition(query_vec, row['icd10_3char'], n=cfg.pool_n)
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
