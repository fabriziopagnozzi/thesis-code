"""
Candidate pool construction for MIMIC-IV retrieval experiments.

Builds per-condition candidate pools from the existing LanceDB embeddings,
filtered by hadm_id sets derived from DuckDB. Strategy: one global vector
store, per-query metadata filtering.
"""

import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import polars as pl
from duckdb import DuckDBPyConnection
from numpy.typing import NDArray

from experiments.mimic.configs import (
    MIMIC_RESULTS_DIR,
    VECTOR_DB_DIR,
    BuildQueryPromptsCfg,
    EvaluateCfg,
    global_cfg,
)
from helpers.embedder import Embedder
from helpers.query_algorithms import ScoringFunction, select


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

    results = []
    for strategy in strategies:
        needs_lambda = strategy in ('mmr', 'gmmr', 'facility_location')
        lams = lam_values if needs_lambda else [None]

        for lam in lams:
            for k in k_values:
                if k > pool.n:
                    break

                selected = select(
                    strategy=strategy,
                    sim_to_query=sim_to_query,
                    k=k,
                    sim_matrix=sim_matrix,
                    embeddings=pool.vectors,
                    query_embedding=query_vec,
                    lam=lam if lam is not None else 0.5,
                )

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

    MAX_SIM_MATRIX_SIZE = 5_000

    def sim_matrix(self) -> NDArray[np.float32]:
        """Pairwise cosine similarity (cached, vectors assumed normalized)."""

        if self._sim_matrix is None:
            if self.n > self.MAX_SIM_MATRIX_SIZE:
                raise ValueError(
                    'Exceeded size {MAX_SIM_MATRIX_SIZE}. Apply prefilter_n to reduce pool size first.'
                )

            self._sim_matrix = self.vectors @ self.vectors.T

        return self._sim_matrix

    def sim_to_query(self, query_vec: NDArray[np.float32]) -> NDArray[np.float32]:
        return self.vectors @ query_vec

    def slice(self, indices: NDArray[np.intp]) -> CandidatePool:
        """Return a new pool containing only the given row indices."""
        idx_list = indices.tolist()
        return CandidatePool(
            chunk_ids=[self.chunk_ids[i] for i in idx_list],
            hadm_ids=self.hadm_ids[indices],
            vectors=self.vectors[indices],
            texts=[self.texts[i] for i in idx_list],
            section_names=[self.section_names[i] for i in idx_list],
            metadata_df=self.metadata_df[idx_list],
        )

    def top_k_by_similarity(self, query_vec: NDArray[np.float32], k: int) -> CandidatePool:
        sim = self.sim_to_query(query_vec)
        take = min(k, self.n)
        idx = np.argsort(sim)[::-1][:take].copy()
        return self.slice(idx)

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


class CandidatePoolBuilder:
    _OPS: ClassVar[dict[str, Callable[[Any, Any], bool]]] = {
        '>': operator.gt,
        '<': operator.lt,
        '>=': operator.ge,
        '<=': operator.le,
        '==': operator.eq,
    }

    def __init__(self, con: DuckDBPyConnection, cfg: EvaluateCfg, device: str = 'cuda'):
        self._con = con
        self._embedder = Embedder(cfg.embedding_model, device=device, batch_size=1)
        self._condition_to_hadm_ids_cache: dict[str, set[int]] = {}
        self._modifier_to_hadm_ids_cache: dict[str, set[int]] = {}

        prompts_cfg = BuildQueryPromptsCfg.load()
        self._charlson_label_to_col_name = global_cfg.shared_queries_cfg.label_to_charlson_col
        self._demographic_filters = prompts_cfg.demographic_filters

        meta_path = MIMIC_RESULTS_DIR / 'admissions_metadata.parquet'
        self._admissions_meta = pl.read_parquet(meta_path) if meta_path.exists() else None

        import lancedb

        db = lancedb.connect(VECTOR_DB_DIR)
        arrow_table = db.open_table('chunks').to_arrow()

        vec_column = arrow_table.column('vector')
        combined = vec_column.combine_chunks()
        dim = combined.type.list_size
        self._corpus_vectors = (
            combined.values.to_numpy(zero_copy_only=False).reshape(-1, dim).astype(np.float32)
        )

        self._corpus_df = pl.DataFrame(arrow_table.drop('vector'))
        self._hadm_id_array: NDArray[np.int64] = self._corpus_df['hadm_id'].to_numpy()

    def for_query_filtered(
        self,
        icd3: str,
        query_vec: NDArray[np.float32],
        n: int,
        modifier_text: str | None = None,
    ) -> CandidatePool:
        """Top-N by cosine from chunks belonging to patients with the condition+modifier."""
        hadm_ids = self._condition_hadm_ids(icd3)
        if modifier_text:
            modifier_ids = self._modifier_hadm_ids(modifier_text)
            if modifier_ids:
                hadm_ids = hadm_ids & modifier_ids
        pool = self._build_pool(hadm_ids)
        return pool.top_k_by_similarity(query_vec, n)

    def for_query_cosine(
        self,
        query_vec: NDArray[np.float32],
        n: int,
    ) -> CandidatePool:
        """Build a candidate pool from the top-N most similar chunks in the full corpus."""
        sim = self._corpus_vectors @ query_vec
        take = min(n, len(sim))
        top_idx = np.argsort(sim)[::-1][:take].copy()
        top_idx_list = top_idx.tolist()
        return CandidatePool(
            chunk_ids=[self._corpus_df['chunk_id'][i] for i in top_idx_list],
            hadm_ids=self._hadm_id_array[top_idx],
            vectors=self._corpus_vectors[top_idx],
            texts=[self._corpus_df['text'][i] for i in top_idx_list],
            section_names=[self._corpus_df['section_name'][i] for i in top_idx_list],
            metadata_df=self._corpus_df[top_idx_list],
        )

    def _condition_hadm_ids(self, icd3: str) -> set[int]:
        if icd3 not in self._condition_to_hadm_ids_cache:
            hadm_ids = self._con.execute(f"""--sql
                SELECT DISTINCT diagnoses_icd.hadm_id
                FROM diagnoses_icd
                WHERE diagnoses_icd.icd_version = 10
                AND SUBSTR(diagnoses_icd.icd_code, 1, 3) = '{icd3}'
            """).pl()['hadm_id']
            self._condition_to_hadm_ids_cache[icd3] = set(hadm_ids.to_list())
        return self._condition_to_hadm_ids_cache[icd3]

    def _build_pool(self, hadm_ids: set[int]) -> CandidatePool:
        mask = np.isin(self._hadm_id_array, np.fromiter(hadm_ids, dtype=np.int64))
        pool_df = self._corpus_df.filter(pl.Series(mask))
        pool_vectors = self._corpus_vectors[mask]

        return CandidatePool(
            chunk_ids=pool_df['chunk_id'].to_list(),
            hadm_ids=pool_df['hadm_id'].to_numpy(),
            vectors=pool_vectors,
            texts=pool_df['text'].to_list(),
            section_names=pool_df['section_name'].to_list(),
            metadata_df=pool_df,
        )

    def _modifier_hadm_ids(self, modifier_text: str) -> set[int]:
        """hadm_ids matching a modifier — tries Charlson comorbidity first, then demographic."""
        if modifier_text in self._modifier_to_hadm_ids_cache:
            return self._modifier_to_hadm_ids_cache[modifier_text]

        # Comorbidity modifier (Charlson table)
        col = self._charlson_label_to_col_name.get(modifier_text)
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

    def embed_query(self, query_text: str) -> NDArray[np.float32]:
        return self._embedder.embed_corpus([query_text])[0]
