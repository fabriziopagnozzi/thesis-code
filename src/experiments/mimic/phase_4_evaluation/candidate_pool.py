"""
Candidate pool construction for MIMIC-IV retrieval experiments.

Builds per-condition candidate pools from the existing LanceDB embeddings,
filtered by hadm_id sets derived from DuckDB. Strategy: one global vector
store, per-query metadata filtering.
"""

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from duckdb import DuckDBPyConnection
from numpy.typing import NDArray

from experiments.mimic.configs import VECTOR_DB_DIR, BuildQueryPromptsCfg, EvaluateCfg
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
                    continue

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


class CandidatePoolBuilder:
    def __init__(self, con: DuckDBPyConnection, cfg: EvaluateCfg, device: str = 'cuda'):
        self._con = con
        self._embedder = Embedder(cfg.embedding_model, device=device, batch_size=1)
        self._condition_to_hadm_ids_cache: dict[str, set[int]] = {}
        self._comorbidity_to_hadm_ids_cache: dict[str, set[int]] = {}
        self._charlson_label_to_col_name = BuildQueryPromptsCfg.load().label_to_charlson_col

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

    def for_query_stratified(
        self,
        icd3: str,
        query_vec: NDArray[np.float32],
        prefilter_n: int,
        modifier_text: str | None = None,
        strata_other_frac: float = 0.2,
    ) -> CandidatePool:
        """Build a stratified candidate pool that preserves multi-cluster structure.

        For comorbidity modifiers, splits hadm_ids into three strata:
        intersection (primary & modifier), primary-only, modifier-only.
        Allocates prefilter_n slots proportionally and fills each stratum
        by cosine similarity to the query, then concatenates.

        For demographic/unknown modifiers, falls back to a single stratum.
        """
        primary_hadm_ids = self._condition_hadm_ids(icd3)

        if modifier_text:
            modifier = self._comorbidity_hadm_ids(modifier_text)
            if modifier:
                intersection_ids = primary_hadm_ids & modifier
                primary_only_ids = primary_hadm_ids - modifier
                modifier_only_ids = modifier - primary_hadm_ids

                n_other = int(prefilter_n * strata_other_frac)
                n_intersection = prefilter_n - 2 * n_other

                strata = [
                    ('intersection', intersection_ids, n_intersection),
                    ('primary_only', primary_only_ids, n_other),
                    ('modifier_only', modifier_only_ids, n_other),
                ]

                pools: list[tuple[CandidatePool, str, int]] = []
                unused = 0
                for name, hadm_ids, budget in strata:
                    if not hadm_ids:
                        unused += budget
                        continue
                    stratum_pool = self._build_pool(hadm_ids)
                    take = min(budget, stratum_pool.n)
                    if take < budget:
                        unused += budget - take
                    pools.append((stratum_pool.top_k_by_similarity(query_vec, take), name, take))

                # Redistribute unused slots to intersection pool
                if unused > 0 and pools:
                    _, first_name, first_take = pools[0]
                    parent_pool = self._build_pool(
                        intersection_ids if first_name == 'intersection' else primary_hadm_ids
                    )
                    new_take = min(first_take + unused, parent_pool.n)
                    pools[0] = (
                        parent_pool.top_k_by_similarity(query_vec, new_take),
                        first_name,
                        new_take,
                    )

                return CandidatePool.concat([p for p, _, _ in pools])

        # Fallback: single stratum (demographic modifier or no modifier)
        pool = self._build_pool(primary_hadm_ids)
        return pool.top_k_by_similarity(query_vec, prefilter_n)

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

    def _comorbidity_hadm_ids(self, modifier_text: str) -> set[int]:
        """All hadm_ids carrying this comorbidity (via Charlson table)."""
        if modifier_text not in self._comorbidity_to_hadm_ids_cache:
            col = self._charlson_label_to_col_name.get(modifier_text)
            if col is None:
                self._comorbidity_to_hadm_ids_cache[modifier_text] = set()
            else:
                hadm_ids = self._con.execute(f"""--sql
                    SELECT DISTINCT charlson.hadm_id
                    FROM charlson
                    WHERE charlson.{col} > 0
                """).pl()['hadm_id']
                self._comorbidity_to_hadm_ids_cache[modifier_text] = set(hadm_ids.to_list())
        return self._comorbidity_to_hadm_ids_cache[modifier_text]

    def embed_query(self, query_text: str) -> NDArray[np.float32]:
        return self._embedder.embed_corpus([query_text])[0]
