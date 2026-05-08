from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypedDict, Unpack, cast

import numpy as np
import polars as pl
import pyarrow as pa
from lancedb import Table
from numpy.typing import NDArray

from experiments.mimic.global_configs import (
    duckdb_con,
    global_cfg,
    lancedb_con,
    read_parquet,
)
from experiments.mimic.queries.schemas_queries import QueryModifier, QueryModifierLabelId
from experiments.mimic.utils.utils import get_vec_col_name
from helpers.embedder import Embedder
from helpers.query_algorithms import ScoringFunction, select

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
        self._con.register('admissions_metadata', read_parquet('admissions_metadata'))
        self._embedder = Embedder(
            **kwargs,
            batch_size=1,
            query_prompt=global_cfg.query_retrieval_instruction,
        )
        self._table: Table = lancedb_con.open_table(global_cfg.chunks_vec_table)
        self._vec_col_name: str = get_vec_col_name(kwargs['model_name'])
        self._vec_dim: int = self._table.schema.field(self._vec_col_name).type.list_size
        self._modifier_to_hadm_ids_cache: dict[QueryModifierLabelId, set[int]] = {}
        print(
            f'[ChunkPoolBuilder] table: {global_cfg.chunks_vec_table!r}, '
            f'vector column: {self._vec_col_name!r} ({self._vec_dim}-dim), '
            f'{self._table.count_rows():,} rows'
        )

    def empty_pool(self) -> ChunkPool:
        return ChunkPool(
            chunk_ids=[],
            hadm_ids=np.empty(0, dtype=np.int64),
            vectors=np.empty((0, self._vec_dim), dtype=np.float32),
            texts=[],
            section_names=[],
            contextual_prefixes=[],
        )

    def from_arrow(self, pyarrow_table: pa.Table) -> ChunkPool:
        """Convert a LanceDB search result (Arrow table) into a ChunkPool."""
        if pyarrow_table.num_rows == 0:
            return self.empty_pool()

        vec_col = cast(
            pa.FixedSizeListArray, pyarrow_table.column(self._vec_col_name).combine_chunks()
        )
        vectors = (
            vec_col.values
            .to_numpy(zero_copy_only=False)
            .reshape(-1, vec_col.type.list_size)
            .astype(np.float32)
        )
        drop_cols = [self._vec_col_name]

        if '_distance' in pyarrow_table.schema.names:
            drop_cols.append('_distance')

        df = pl.DataFrame(pyarrow_table.drop(drop_cols))

        return ChunkPool(
            chunk_ids=df['chunk_id'].to_list(),
            hadm_ids=df['hadm_id'].to_numpy().astype(np.int64),
            texts=df['text'].to_list(),
            section_names=df['section_name'].to_list(),
            contextual_prefixes=df['contextual_prefix'].fill_null('').to_list(),
            vectors=vectors,
        )

    def topk_cosine(
        self, vec: NDArray[np.float32], k: int, predicate: str | None = None
    ) -> ChunkPool:
        where_clause = (
            f'{predicate} AND {global_cfg.sections_filter_sql}'
            if predicate
            else global_cfg.sections_filter_sql
        )
        result = (
            self._table
            .search(vec, vector_column_name=self._vec_col_name)
            .where(where_clause)
            .limit(k)
            .to_arrow()
        )
        return self.from_arrow(result)

    def topk_cosine_for_condition(
        self, vec: NDArray[np.float32], condition_icd10_prefix: str, k: int
    ) -> ChunkPool:
        result = (
            self._table
            .search(vec, vector_column_name=self._vec_col_name)
            .where(
                f"array_has(icd10_3char_list, '{condition_icd10_prefix}') AND {global_cfg.sections_filter_sql}"
            )
            .limit(k)
            .to_arrow()
        )
        return self.from_arrow(result)

    def topk_cosine_for_modifiers(
        self,
        vec: NDArray[np.float32],
        icd10_3char: str,
        modifiers: list[QueryModifier],
        k: int,
    ) -> ChunkPool:
        """Top-N cosine pool for chunks matching condition AND any modifier.
        The pool covers the union across modifiers: condition AND (mod1 OR mod2 OR ...).
        """
        if k <= 0 or not modifiers:
            return self.empty_pool()

        modifier_preds = ' OR '.join(f'({m.sql_metadata_table_predicate()})' for m in modifiers)
        predicate = (
            f"array_has(icd10_3char_list, '{icd10_3char}') "
            f'AND ({modifier_preds}) '
            f'AND {global_cfg.sections_filter_sql}'
        )

        return self.from_arrow(
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
            return self.empty_pool()

        hadm_list = ', '.join(str(h) for h in hadm_ids)
        return self.from_arrow(
            self._table
            .search(vec, vector_column_name=self._vec_col_name)
            .where(f'hadm_id IN ({hadm_list}) AND {global_cfg.sections_filter_sql}')
            .limit(k)
            .to_arrow()
        )

    def get_by_chunk_ids(self, ids: set[str]) -> ChunkPool:
        if not ids:
            return self.empty_pool()

        placeholders = ', '.join(f"'{cid}'" for cid in ids)
        return self.from_arrow(
            self._table.search().where(f'chunk_id IN ({placeholders})').limit(len(ids)).to_arrow()
        )

    def get_modifier_hadm_ids(self, m: QueryModifier) -> set[int]:
        if m.label in self._modifier_to_hadm_ids_cache:
            return self._modifier_to_hadm_ids_cache[m.label]

        match m.type:
            case 'comorbidity':
                sql = f"""--sql
                    SELECT DISTINCT hadm_id FROM charlson WHERE {m.sql_metadata_table_predicate()}
                """
            case 'demographic':
                sql = f"""--sql
                    SELECT DISTINCT hadm_id FROM admissions_metadata WHERE {m.sql_metadata_table_predicate()}
                """
            case _:
                raise RuntimeError(f'[ERROR] Unsupported: {m.type=}')

        result = set(self._con.execute(sql).pl()['hadm_id'].to_list())
        self._modifier_to_hadm_ids_cache[m.label] = result
        return result

    def get_icd3_hadm_ids(self, icd10_3char: str) -> set[int]:
        if icd10_3char in self._modifier_to_hadm_ids_cache:
            return self._modifier_to_hadm_ids_cache[icd10_3char]

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
        self._modifier_to_hadm_ids_cache[icd10_3char] = result
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
class ChunkPoolRetrievalResult:
    strategy: str
    k: int
    lam: float | None
    selected_indices: NDArray[np.intp]
    selected_chunk_ids: list[str]
    selected_hadm_ids: list[int]
    sim_to_query: NDArray[np.float32]


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

    def iter_batches(self, each: int) -> Iterator[tuple[int, ChunkPool]]:
        for start in range(0, self.n, each):
            end = min(start + each, self.n)
            yield start, self.select_by_indices(np.arange(start, end, dtype=np.intp))

    def filter_by_hadm_ids(self, ids: set[int]) -> ChunkPool:
        if not ids:
            return self.select_by_indices(np.empty(0, dtype=np.intp))

        return self.select_by_indices(
            np.flatnonzero(np.isin(self.hadm_ids, list(ids))).astype(np.intp)
        )

    def select_by_indices(self, indices: NDArray[np.intp]) -> ChunkPool:
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
        return merged.select_by_indices(np.array(keep, dtype=np.intp))

    def run_retrieval(
        self,
        query_embedding: NDArray[np.float32],
        strategies: list[ScoringFunction],
        k_values: list[int],
        lam_values: list[float],
    ) -> list[ChunkPoolRetrievalResult]:
        valid_k_values = [k for k in k_values if k <= self.n]
        if not valid_k_values:
            return []
        max_k = max(valid_k_values)

        q_sim_scores = self.sim_scores(query_embedding)
        sim_matrix = self.sim_matrix()

        results = []
        for strategy in strategies:
            lambdas = lam_values if strategy in ('mmr', 'gmmr', 'fac_loc') else [None]

            for lam in lambdas:
                all_selected_indices = select(
                    strategy=strategy,
                    sim_to_query=q_sim_scores,
                    k=max_k,
                    sim_matrix=sim_matrix,
                    embeddings=self.vectors,
                    query_embedding=query_embedding,
                    lam=lam if lam is not None else 0.5,
                )
                for k in valid_k_values:
                    selected = all_selected_indices[:k]
                    results.append(
                        ChunkPoolRetrievalResult(
                            strategy=strategy,
                            k=k,
                            lam=lam,
                            selected_indices=selected,
                            selected_chunk_ids=[self.chunk_ids[i] for i in selected],
                            selected_hadm_ids=[int(self.hadm_ids[i]) for i in selected],
                            sim_to_query=q_sim_scores[selected],
                        )
                    )

        return results
