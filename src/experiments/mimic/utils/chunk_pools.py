from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypedDict, Unpack, cast

import numpy as np
import polars as pl
import pyarrow as pa
from numpy.typing import NDArray

from experiments.mimic.global_configs import (
    duckdb_con,
    global_cfg,
    lancedb_con,
    read_parquet,
)
from experiments.mimic.queries.schemas_queries import QueryModifier
from experiments.mimic.utils.utils import get_vec_col_name
from helpers.embedder import Embedder

MAX_CANDIDATES = 100_000


class ChunkPoolBuilderArgs(TypedDict):
    model_name: str
    device: NotRequired[Literal['cpu', 'cuda']]


class ChunkPoolBuilder:
    def __init__(
        self,
        **kwargs: Unpack[ChunkPoolBuilderArgs],
    ):
        self._con = duckdb_con
        self._con.register('admissions_meta', read_parquet('admissions_metadata'))
        self._embedder = Embedder(
            **kwargs,
            batch_size=1,
            query_prompt=global_cfg.query_retrieval_instruction,
        )
        self._table = lancedb_con.open_table(global_cfg.chunks_vec_table)
        self._vec_col_name = get_vec_col_name(kwargs['model_name'])
        self._vec_dim: int = self._table.schema.field(self._vec_col_name).type.list_size
        self._modifier_to_hadm_ids_cache: dict[str, set[int]] = {}

        print(
            f'[CandidatePoolBuilder] table: {global_cfg.chunks_vec_table!r}, '
            f'vector column: {self._vec_col_name!r} ({self._vec_dim}-dim), '
            f'{self._table.count_rows():,} rows'
        )

    def _empty_pool(self) -> ChunkPool:
        return ChunkPool(
            chunk_ids=[],
            hadm_ids=np.empty(0, dtype=np.int64),
            vectors=np.empty((0, self._vec_dim), dtype=np.float32),
            texts=[],
            section_names=[],
            contextual_prefixes=[],
        )

    def _pool_from_arrow(self, result: pa.Table) -> ChunkPool:
        """Convert a LanceDB search result (Arrow table) into a CandidatePool."""
        if result.num_rows == 0:
            return self._empty_pool()

        vec_col = cast(pa.FixedSizeListArray, result.column(self._vec_col_name).combine_chunks())
        vectors = (
            vec_col.values
            .to_numpy(zero_copy_only=False)
            .reshape(-1, vec_col.type.list_size)
            .astype(np.float32)
        )
        drop_cols = [self._vec_col_name]

        if '_distance' in result.schema.names:
            drop_cols.append('_distance')

        df = pl.DataFrame(result.drop(drop_cols))

        return ChunkPool(
            chunk_ids=df['chunk_id'].to_list(),
            hadm_ids=df['hadm_id'].to_numpy().astype(np.int64),
            vectors=vectors,
            texts=df['text'].to_list(),
            section_names=df['section_name'].to_list(),
            contextual_prefixes=df['contextual_prefix'].fill_null('').to_list(),
        )

    def topk_cosine(self, vec: NDArray[np.float32], k: int, predicate: str = '') -> ChunkPool:
        result = (
            self._table
            .search(vec, vector_column_name=self._vec_col_name)
            .where(f'{predicate + ("AND" if predicate else "")} {global_cfg.sections_filter_sql}')
            .limit(k)
            .to_arrow()
        )
        return self._pool_from_arrow(result)

    def topk_cosine_for_condition(
        self, vec: NDArray[np.float32], icd10_3char: str, k: int
    ) -> ChunkPool:
        result = (
            self._table
            .search(vec, vector_column_name=self._vec_col_name)
            .where(
                f"array_has(icd10_3char_list, '{icd10_3char}') AND {global_cfg.sections_filter_sql}"
            )
            .limit(k)
            .to_arrow()
        )
        return self._pool_from_arrow(result)

    def topk_cosine_for_chunk_ids(self, ids: set[str]) -> ChunkPool:
        if not ids:
            return self._empty_pool()

        placeholders = ', '.join(f"'{cid}'" for cid in ids)
        return self._pool_from_arrow(
            self._table.search().where(f'chunk_id IN ({placeholders})').limit(len(ids)).to_arrow()
        )

    def topk_cosine_for_modifiers(
        self,
        vec: NDArray[np.float32],
        icd10_3char: str,
        modifiers: list[QueryModifier],
        k: int,
    ) -> ChunkPool:
        """Top-N cosine pool for chunks matching condition AND any modifier.
        Builds SQL predicates directly from the lance table's structured columns.
        The pool covers the union across modifiers: condition AND (mod1 OR mod2 OR ...).
        """
        if k <= 0 or not modifiers:
            return self._empty_pool()

        modifier_preds = ' OR '.join(f'({m.sql_predicate()})' for m in modifiers)
        predicate = (
            f"array_has(icd10_3char_list, '{icd10_3char}') "
            f'AND ({modifier_preds}) '
            f'AND {global_cfg.sections_filter_sql}'
        )

        return self._pool_from_arrow(
            self._table
            .search(vec, vector_column_name=self._vec_col_name)
            .where(predicate)
            .limit(k)
            .to_arrow()
        )

    def topk_cosine_for_hadm_ids(
        self, vec: NDArray[np.float32], hadm_ids: set[int], k: int
    ) -> ChunkPool:
        """Top-N cosine pool restricted to a given hadm_id set."""
        if not hadm_ids:
            return self._empty_pool()

        hadm_list = ', '.join(str(h) for h in hadm_ids)
        return self._pool_from_arrow(
            self._table
            .search(vec, vector_column_name=self._vec_col_name)
            .where(f'hadm_id IN ({hadm_list}) AND {global_cfg.sections_filter_sql}')
            .limit(k)
            .to_arrow()
        )

    def get_modifier_hadm_ids(self, m: QueryModifier) -> set[int]:
        if m.label in self._modifier_to_hadm_ids_cache:
            return self._modifier_to_hadm_ids_cache[m.label]

        if m.type == 'comorbidity':
            sql = f"""--sql SELECT DISTINCT hadm_id FROM charlson WHERE {m.sql_predicate()}"""
        else:
            sql = (
                f"""--sql SELECT DISTINCT hadm_id FROM admissions_meta WHERE {m.sql_predicate()}"""
            )

        result = set(self._con.execute(sql).pl()['hadm_id'].to_list())
        self._modifier_to_hadm_ids_cache[m.label] = result
        return result

    def get_icd3_hadm_ids(self, icd10_3char: str) -> set[int]:
        cache_key = f'__icd3__{icd10_3char}'
        if cache_key in self._modifier_to_hadm_ids_cache:
            return self._modifier_to_hadm_ids_cache[cache_key]

        result = set(
            self._con
            .execute(f"""--sql
                SELECT DISTINCT hadm_id
                FROM unified_diagnoses
                WHERE LEFT(unified_icd10, 3) = '{icd10_3char}'
            """)
            .pl()['hadm_id']
            .to_list()
        )
        self._modifier_to_hadm_ids_cache[cache_key] = result
        return result

    def get_hadm_ids_by_condition_modifier(self, icd10_3char: str, mod: QueryModifier) -> set[int]:
        if mod.type == 'comorbidity':
            return self.get_hadm_ids_by_condition_comorbidity(icd10_3char, mod)
        if mod.type == 'demographic':
            return self.get_hadm_ids_by_condition_demographic(icd10_3char, mod)
        return set()

    def get_hadm_ids_by_condition_comorbidity(
        self, icd10_3char: str, mod: QueryModifier
    ) -> set[int]:
        return self.get_icd3_hadm_ids(icd10_3char) & self.get_modifier_hadm_ids(mod)

    def get_hadm_ids_by_condition_demographic(
        self, icd10_3char: str, mod: QueryModifier
    ) -> set[int]:
        return self.get_icd3_hadm_ids(icd10_3char) & self.get_modifier_hadm_ids(mod)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self._embedder.embed_query(text)


@dataclass
class ChunkPool:
    chunk_ids: list[str]
    hadm_ids: NDArray[np.int64]
    vectors: NDArray[np.float32]  # (n, d)
    texts: list[str]
    section_names: list[str]
    contextual_prefixes: list[str]
    _sim_matrix: NDArray[np.float32] | None = field(default=None, repr=False)

    @property
    def n(self) -> int:
        return self.vectors.shape[0]

    def sim_matrix(self) -> NDArray[np.float32]:
        """Pairwise cosine similarity (cached, vectors assumed normalized)."""
        if self._sim_matrix is None:
            if self.n > MAX_CANDIDATES:
                raise ValueError(
                    'Exceeded size {MAX_CANDIDATES}. Apply prefilter_n to reduce pool size first.'
                )
            self._sim_matrix = self.vectors @ self.vectors.T

        return self._sim_matrix

    def sim_scores(self, vec: NDArray[np.float32]) -> NDArray[np.float32]:
        return self.vectors @ vec

    def iter_batches(self, batch_size: int) -> Iterator[tuple[int, ChunkPool]]:
        for start in range(0, self.n, batch_size):
            end = min(start + batch_size, self.n)
            yield start, self.slice(np.arange(start, end, dtype=np.intp))

    def filter_hadm_ids(self, hadm_ids: set[int]) -> ChunkPool:
        if not hadm_ids:
            return self.slice(np.empty(0, dtype=np.intp))
        keep = np.flatnonzero(np.isin(self.hadm_ids, list(hadm_ids))).astype(np.intp)
        return self.slice(keep)

    def slice(self, indices: NDArray[np.intp]) -> ChunkPool:
        idx_list = indices.tolist()
        return ChunkPool(
            chunk_ids=[self.chunk_ids[i] for i in idx_list],
            hadm_ids=self.hadm_ids[indices],
            vectors=self.vectors[indices],
            texts=[self.texts[i] for i in idx_list],
            section_names=[self.section_names[i] for i in idx_list],
            contextual_prefixes=[self.contextual_prefixes[i] for i in idx_list],
        )

    @classmethod
    def concat(cls, pools: list[ChunkPool]) -> ChunkPool:
        return ChunkPool(
            chunk_ids=[cid for p in pools for cid in p.chunk_ids],
            hadm_ids=np.concatenate([p.hadm_ids for p in pools]),
            vectors=np.concatenate([p.vectors for p in pools]),
            texts=[t for p in pools for t in p.texts],
            section_names=[s for p in pools for s in p.section_names],
            contextual_prefixes=[s for p in pools for s in p.contextual_prefixes],
        )

    @classmethod
    def merge(cls, pools: list[ChunkPool]) -> ChunkPool:
        """Concatenate pools and deduplicate by chunk_id, keeping first occurrence."""
        merged = cls.concat(pools)
        seen: set[str] = set()
        keep: list[int] = []
        for i, cid in enumerate(merged.chunk_ids):
            if cid not in seen:
                seen.add(cid)
                keep.append(i)
        return merged.slice(np.array(keep, dtype=np.intp))
