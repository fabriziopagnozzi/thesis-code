"""
Candidate pool construction for MIMIC-IV retrieval experiments.

Builds per-condition candidate pools from the existing LanceDB embeddings,
filtered by hadm_id sets derived from DuckDB. Strategy: one global vector
store, per-query metadata filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from numpy.typing import NDArray

from experiments.mimic.duck_db_init import MIMIC_RESULTS_DIR
from experiments.mimic.phase_2_embedding.a_embed import MODEL_NAME
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
                    f'Candidate pool has {self.n:,} chunks - dense sim matrix would be '
                    f'{self.n}x{self.n} ({self.n**2 * 4 / 1e9:.1f} GB). '
                    f'Apply prefilter_n to reduce pool size first.'
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


class CandidatePoolBuilder:
    def __init__(self, con, device: str = 'cuda'):
        self._con = con
        self._device = device
        self._corpus_df: pl.DataFrame | None = None
        self._corpus_vectors: NDArray[np.float32] | None = None
        self._hadm_id_array: NDArray[np.int64] | None = None
        self._embedder: Embedder | None = None
        self._condition_hadm_cache: dict[str, set[int]] = {}

    def _load_corpus(self) -> None:
        if self._corpus_df is not None:
            return

        import lancedb

        db = lancedb.connect(MIMIC_RESULTS_DIR / '_lancedb')
        arrow_table = db.open_table('chunks').to_arrow()

        # FixedSizeList column: extract flat value buffer then reshape
        vec_column = arrow_table.column('vector')
        combined = vec_column.combine_chunks()
        dim = combined.type.list_size
        self._corpus_vectors = (
            combined.values.to_numpy(zero_copy_only=False).reshape(-1, dim).astype(np.float32)
        )

        self._corpus_df = pl.DataFrame(arrow_table.drop('vector'))
        self._hadm_id_array = self._corpus_df['hadm_id'].to_numpy()

        n, d = self._corpus_vectors.shape  # type: ignore
        print(f'Loaded corpus: {n:,} chunks, dim={d}')

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(MODEL_NAME, device=self._device, batch_size=1)
        return self._embedder

    def _condition_hadm_ids(self, icd3: str) -> set[int]:
        if icd3 not in self._condition_hadm_cache:
            rows = self._con.execute(f"""--sql
                SELECT DISTINCT diagnoses_icd.hadm_id
                FROM diagnoses_icd
                WHERE diagnoses_icd.icd_version = 10
                AND SUBSTR(diagnoses_icd.icd_code, 1, 3) = '{icd3}'
            """).fetchall()
            self._condition_hadm_cache[icd3] = {r[0] for r in rows}
        return self._condition_hadm_cache[icd3]

    def _build_pool(self, hadm_ids: set[int]) -> CandidatePool:
        self._load_corpus()
        assert self._corpus_df is not None
        assert self._corpus_vectors is not None
        assert self._hadm_id_array is not None

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

    def for_condition(self, icd3: str) -> CandidatePool:
        return self._build_pool(self._condition_hadm_ids(icd3))

    def for_hadm_ids(self, hadm_ids: set[int]) -> CandidatePool:
        return self._build_pool(hadm_ids)

    def embed_query(self, query_text: str) -> NDArray[np.float32]:
        return self._get_embedder().embed_corpus([query_text])[0]


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


# -- CLI: inspect a candidate pool --
if __name__ == '__main__':
    import argparse

    from experiments.mimic.duck_db_init import connect_mimic_duckdb

    parser = argparse.ArgumentParser(description='Inspect a per-condition candidate pool')
    parser.add_argument('icd3', help='3-char ICD-10 prefix (e.g. I63 for stroke)')
    parser.add_argument('--query', help='Optional query text to run retrieval', default=None)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--prefilter', type=int, default=500)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    con = connect_mimic_duckdb()

    builder = CandidatePoolBuilder(con, device=args.device)
    pool = builder.for_condition(args.icd3)

    print(
        f'\nCandidate pool for {args.icd3}:\n'
        f'  Chunks: {pool.n:,}\n'
        f'  Unique hadm_ids: {len(set(pool.hadm_ids)):,}\n'
        f'  Sections: {pl.Series(pool.section_names).value_counts().sort("count", descending=True)}'
    )

    if args.query:
        query_vec = builder.embed_query(args.query)
        results = run_retrieval(
            pool,
            query_vec,
            strategies=['top_k', 'mmr', 'facility_location'],
            k_values=[args.k],
            lam_values=[0.5],
            prefilter_n=args.prefilter,
        )
        for r in results:
            print(
                f'\n--- {r.strategy} (k={r.k}, lam={r.lam}) ---\n'
                f'  Unique hadm_ids selected: {len(set(r.selected_hadm_ids))}\n'
                f'  Avg sim to query: {r.sim_to_query.mean():.4f}'
            )
            for i, idx in enumerate(r.selected_indices[:5]):
                print(f'  [{i + 1}] sim={r.sim_to_query[i]:.4f}  {pool.texts[idx][:120]}...')
