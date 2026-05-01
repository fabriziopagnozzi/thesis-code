import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypedDict, Unpack, cast

import lancedb
import numpy as np
import polars as pl
import pyarrow as pa
from duckdb import DuckDBPyConnection
from numpy.typing import NDArray

from experiments.mimic.global_configs import (
    MimicPaths,
    global_cfg,
    read_parquet,
)
from experiments.mimic.queries.schemas_queries import BuildQueryPromptsCfg
from experiments.mimic.utils.utils import get_vec_col_name
from helpers.embedder import Embedder
from helpers.query_algorithms import ScoringFunction, select

MAX_CANDIDATES = 100_000


@dataclass
class RetrievalResult:
    strategy: str
    k: int
    lam: float | None
    selected_indices: NDArray[np.intp]
    selected_chunk_ids: list[str]
    selected_hadm_ids: list[int]
    sim_to_query: NDArray[np.float32]


def run_retrieval(
    pool: CandidatePool,
    query_vec: NDArray[np.float32],
    strategies: list[ScoringFunction],
    k_values: list[int],
    lam_values: list[float],
    prefilter_n: int | None = 500,
) -> list[RetrievalResult]:
    sim_to_query = pool.sim_to_query(query_vec)

    if prefilter_n is not None and pool.n > prefilter_n:
        top_indices = np.argsort(sim_to_query)[::-1][:prefilter_n].copy()
        pool = pool.slice(top_indices)
        sim_to_query = sim_to_query[top_indices]

    sim_matrix = pool.sim_matrix()

    valid_k_values = [k for k in k_values if k <= pool.n]
    if not valid_k_values:
        return []
    max_k = max(valid_k_values)

    results = []
    for strategy in strategies:
        needs_lambda = strategy in ('mmr', 'gmmr', 'fac_loc')
        lams = lam_values if needs_lambda else [None]

        for lam in lams:
            all_selected = select(
                strategy=strategy,
                sim_to_query=sim_to_query,
                k=max_k,
                sim_matrix=sim_matrix,
                embeddings=pool.vectors,
                query_embedding=query_vec,
                lam=lam if lam is not None else 0.5,
            )

            for k in valid_k_values:
                selected = all_selected[:k]
                results.append(
                    RetrievalResult(
                        strategy=strategy,
                        k=k,
                        lam=lam,
                        selected_indices=selected,
                        selected_chunk_ids=[pool.chunk_ids[i] for i in selected],
                        selected_hadm_ids=[int(pool.hadm_ids[i]) for i in selected],
                        sim_to_query=sim_to_query[selected],
                    )
                )

    return results


@dataclass
class CandidatePool:
    chunk_ids: list[str]
    hadm_ids: NDArray[np.int64]
    vectors: NDArray[np.float32]  # (n, d)
    texts: list[str]
    section_names: list[str]
    metadata_df: pl.DataFrame

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

    def sim_to_query(self, query_vec: NDArray[np.float32]) -> NDArray[np.float32]:
        return self.vectors @ query_vec

    def slice(self, indices: NDArray[np.intp]) -> CandidatePool:
        idx_list = indices.tolist()
        return CandidatePool(
            chunk_ids=[self.chunk_ids[i] for i in idx_list],
            hadm_ids=self.hadm_ids[indices],
            vectors=self.vectors[indices],
            texts=[self.texts[i] for i in idx_list],
            section_names=[self.section_names[i] for i in idx_list],
            metadata_df=self.metadata_df[idx_list],
        )

    @classmethod
    def concat(cls, pools: list[CandidatePool]) -> CandidatePool:
        return CandidatePool(
            chunk_ids=[cid for p in pools for cid in p.chunk_ids],
            hadm_ids=np.concatenate([p.hadm_ids for p in pools]),
            vectors=np.concatenate([p.vectors for p in pools]),
            texts=[t for p in pools for t in p.texts],
            section_names=[s for p in pools for s in p.section_names],
            metadata_df=pl.concat([p.metadata_df for p in pools]),
        )

    @classmethod
    def merge(cls, pools: list[CandidatePool]) -> CandidatePool:
        """Concatenate pools and deduplicate by chunk_id, keeping first occurrence."""
        merged = cls.concat(pools)
        seen: set[str] = set()
        keep: list[int] = []
        for i, cid in enumerate(merged.chunk_ids):
            if cid not in seen:
                seen.add(cid)
                keep.append(i)
        return merged.slice(np.array(keep, dtype=np.intp))


class CandidatePoolBuilderArgs(TypedDict):
    embedding_model: str


class CandidatePoolBuilder:
    _OPS: ClassVar[dict[str, Callable[[Any, Any], bool]]] = {
        '>': operator.gt,
        '<': operator.lt,
        '>=': operator.ge,
        '<=': operator.le,
        '==': operator.eq,
    }

    def __init__(self, con: DuckDBPyConnection, **kwargs: Unpack[CandidatePoolBuilderArgs]):
        self._con = con
        self._embedder = Embedder(
            kwargs['embedding_model'],
            batch_size=1,
            query_prompt=global_cfg.query_retrieval_instruction,
        )
        self._table = lancedb.connect(MimicPaths.vector_db_dir).open_table(
            global_cfg.chunks_vec_table
        )
        self._vec_col_name = get_vec_col_name(kwargs['embedding_model'])
        self._vec_dim: int = self._table.schema.field(self._vec_col_name).type.list_size

        self._modifier_to_hadm_ids_cache: dict[str, set[int]] = {}
        self._demographic_filters = BuildQueryPromptsCfg.load().demographic_filters

        self._admissions_meta = read_parquet('admissions_metadata')

        print(
            f'[CandidatePoolBuilder] table: {global_cfg.chunks_vec_table!r}, '
            f'vector column: {self._vec_col_name!r} ({self._vec_dim}-dim), '
            f'{self._table.count_rows():,} rows'
        )

    def _empty_pool(self) -> CandidatePool:
        return CandidatePool(
            chunk_ids=[],
            hadm_ids=np.empty(0, dtype=np.int64),
            vectors=np.empty((0, self._vec_dim), dtype=np.float32),
            texts=[],
            section_names=[],
            metadata_df=pl.DataFrame(),
        )

    def _pool_from_arrow(self, result: pa.Table) -> CandidatePool:
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

        return CandidatePool(
            chunk_ids=df['chunk_id'].to_list(),
            hadm_ids=df['hadm_id'].to_numpy().astype(np.int64),
            vectors=vectors,
            texts=df['text'].to_list(),
            section_names=df['section_name'].to_list(),
            metadata_df=df,
        )

    def for_query_cosine(
        self,
        query_vec: NDArray[np.float32],
        n: int,
    ) -> CandidatePool:
        """Top-N most similar chunks across the full corpus."""
        result = (
            self._table.search(query_vec, vector_column_name=self._vec_col_name).limit(n).to_arrow()
        )
        return self._pool_from_arrow(result)

    def for_query_cosine_condition(
        self,
        query_vec: NDArray[np.float32],
        icd10_3char: str,
        n: int,
    ) -> CandidatePool:
        """Top-N cosine pool restricted to admissions with the given ICD-10 3-char code."""
        result = (
            self._table
            .search(query_vec, vector_column_name=self._vec_col_name)
            .where(
                f"array_has(icd10_3char_list, '{icd10_3char}') AND {global_cfg.sections_filter_sql}"
            )
            .limit(n)
            .to_arrow()
        )
        return self._pool_from_arrow(result)

    def for_hadm_ids_cosine(
        self,
        query_vec: NDArray[np.float32],
        hadm_ids: set[int],
        n: int,
    ) -> CandidatePool:
        """Top-N cosine pool restricted to a given hadm_id set."""
        if not hadm_ids:
            return self._empty_pool()
        hadm_list = ', '.join(str(h) for h in hadm_ids)
        result = (
            self._table
            .search(query_vec, vector_column_name=self._vec_col_name)
            .where(f'hadm_id IN ({hadm_list}) AND {global_cfg.sections_filter_sql}')
            .limit(n)
            .to_arrow()
        )
        return self._pool_from_arrow(result)

    def for_gold_chunks(self, gold_chunk_ids: set[str]) -> CandidatePool:
        """Fetch specific chunks by chunk_id (no vector ranking)."""
        if not gold_chunk_ids:
            return self._empty_pool()
        placeholders = ', '.join(f"'{cid}'" for cid in gold_chunk_ids)
        result = (
            self._table
            .search()
            .where(f'chunk_id IN ({placeholders})')
            .limit(len(gold_chunk_ids))
            .to_arrow()
        )
        return self._pool_from_arrow(result)

    def modifier_hadm_ids(self, modifier_text: str) -> set[int]:
        """hadm_ids matching a modifier - tries Charlson comorbidity first, then demographic."""
        if modifier_text in self._modifier_to_hadm_ids_cache:
            return self._modifier_to_hadm_ids_cache[modifier_text]

        # Comorbidity modifier (Charlson table)
        col = global_cfg.label_to_charlson_col.get(modifier_text)
        if col is not None:
            hadm_ids = self._con.execute(f"""--sql
                SELECT DISTINCT charlson.hadm_id
                FROM charlson
                WHERE charlson.{col} > 0
            """).pl()['hadm_id']
            result = set(hadm_ids.to_list())
            self._modifier_to_hadm_ids_cache[modifier_text] = result
            return result

        # Demographic modifier (admissions_metadata)
        demo = self._demographic_filters.get(modifier_text)
        if demo is not None and self._admissions_meta is not None:
            column, op_str, value = demo
            op_fn = self._OPS.get(op_str)
            if op_fn is not None:
                filtered = self._admissions_meta.filter(op_fn(pl.col(column), value))
                result = set(filtered['hadm_id'].to_list())
                self._modifier_to_hadm_ids_cache[modifier_text] = result
                return result

        self._modifier_to_hadm_ids_cache[modifier_text] = set()
        return set()

    def icd3_hadm_ids(self, icd10_3char: str) -> set[int]:
        """hadm_ids with the given ICD-10 3-char prefix. Cached."""
        cache_key = f'__icd3__{icd10_3char}'
        if cache_key in self._modifier_to_hadm_ids_cache:
            return self._modifier_to_hadm_ids_cache[cache_key]
        result = set(
            self._con
            .execute(f"""--sql
                SELECT DISTINCT hadm_id FROM unified_diagnoses
                WHERE LEFT(unified_icd10, 3) = '{icd10_3char}'
            """)
            .pl()['hadm_id']
            .to_list()
        )
        self._modifier_to_hadm_ids_cache[cache_key] = result
        return result

    def filter_condition_comorbidity(self, icd10_3char: str, modifier_text: str) -> set[int]:
        """hadm_ids that have the condition (ICD-10 3-char) AND the Charlson comorbidity."""
        return self.icd3_hadm_ids(icd10_3char) & self.modifier_hadm_ids(modifier_text)

    def filter_condition_demographic(self, icd10_3char: str, modifier_text: str) -> set[int]:
        """hadm_ids that have the condition (ICD-10 3-char) AND match the demographic filter."""
        return self.icd3_hadm_ids(icd10_3char) & self.modifier_hadm_ids(modifier_text)

    def filter_by_condition_modifier(
        self, icd10_3char: str, modifier_text: str, modifier_type: str
    ) -> set[int]:
        """Dispatch to the appropriate condition-scoped filter based on modifier_type."""
        if modifier_type == 'comorbidity':
            return self.filter_condition_comorbidity(icd10_3char, modifier_text)
        if modifier_type == 'demographic':
            return self.filter_condition_demographic(icd10_3char, modifier_text)
        return set()

    def embed_query(self, query_text: str) -> NDArray[np.float32]:
        return self._embedder.embed_query(query_text)
