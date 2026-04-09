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

from experiments.mimic.configs import VECTOR_DB_DIR, EvaluateCfg
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
    def __init__(self, con: DuckDBPyConnection, cfg: EvaluateCfg, device: str = 'cuda'):
        self._con = con
        self._cfg = cfg
        self._device = device
        self._corpus_df: pl.DataFrame | None = None
        self._corpus_vectors: NDArray[np.float32] | None = None
        self._hadm_id_array: NDArray[np.int64] | None = None
        self._embedder: Embedder | None = None
        self._condition_hadm_cache: dict[str, set[int]] = {}
        self._comorbidity_hadm_cache: dict[str, set[int]] = {}
        self._charlson_label_to_col: dict[str, str] | None = None

    def _load_corpus(self) -> None:
        if self._corpus_df is not None:
            return

        import lancedb

        db = lancedb.connect(VECTOR_DB_DIR)
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
            self._embedder = Embedder(self._cfg.embedding_model, device=self._device, batch_size=1)
        return self._embedder

    def _condition_hadm_ids(self, icd3: str) -> set[int]:
        if icd3 not in self._condition_hadm_cache:
            hadm_ids = self._con.execute(f"""--sql
                SELECT DISTINCT diagnoses_icd.hadm_id
                FROM diagnoses_icd
                WHERE diagnoses_icd.icd_version = 10
                AND SUBSTR(diagnoses_icd.icd_code, 1, 3) = '{icd3}'
            """).pl()['hadm_id']
            self._condition_hadm_cache[icd3] = set(hadm_ids.to_list())
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

    def _get_charlson_mapping(self) -> dict[str, str]:
        """modifier_text -> charlson column name."""
        if self._charlson_label_to_col is None:
            from experiments.mimic.configs import BuildQueryPromptsCfg

            self._charlson_label_to_col = BuildQueryPromptsCfg.load().label_to_charlson_col
        return self._charlson_label_to_col

    def _comorbidity_hadm_ids(self, modifier_text: str) -> set[int]:
        """All hadm_ids carrying this comorbidity (via Charlson table)."""
        if modifier_text not in self._comorbidity_hadm_cache:
            col = self._get_charlson_mapping().get(modifier_text)
            if col is None:
                self._comorbidity_hadm_cache[modifier_text] = set()
            else:
                hadm_ids = self._con.execute(f"""--sql
                    SELECT DISTINCT charlson.hadm_id
                    FROM charlson
                    WHERE charlson.{col} > 0
                """).pl()['hadm_id']
                self._comorbidity_hadm_cache[modifier_text] = set(hadm_ids.to_list())
        return self._comorbidity_hadm_cache[modifier_text]

    def for_condition(self, icd3: str) -> CandidatePool:
        return self._build_pool(self._condition_hadm_ids(icd3))

    def for_query(self, icd3: str, modifier_text: str | None = None) -> CandidatePool:
        """Build pool from primary condition + modifier condition admissions.

        For comorbidity modifiers, unions hadm_ids from both conditions to
        create multi-cluster structure (primary cluster + modifier cluster).
        For demographic modifiers or unknown modifiers, falls back to
        per-condition pool.
        """
        primary = self._condition_hadm_ids(icd3)
        if modifier_text:
            modifier = self._comorbidity_hadm_ids(modifier_text)
            if modifier:
                all_hadm_ids = primary | modifier
                print(
                    f'  Pool for {icd3}+"{modifier_text}": '
                    f'{len(primary):,} primary + {len(modifier):,} modifier '
                    f'= {len(all_hadm_ids):,} total hadm_ids'
                )
                return self._build_pool(all_hadm_ids)
        return self._build_pool(primary)

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
        intersection (primary ∩ modifier), primary-only, modifier-only.
        Allocates prefilter_n slots proportionally and fills each stratum
        by cosine similarity to the query, then concatenates.

        For demographic/unknown modifiers, falls back to a single stratum.
        """
        primary = self._condition_hadm_ids(icd3)

        if modifier_text:
            modifier = self._comorbidity_hadm_ids(modifier_text)
            if modifier:
                intersection = primary & modifier
                primary_only = primary - modifier
                modifier_only = modifier - primary

                n_other = int(prefilter_n * strata_other_frac)
                n_intersection = prefilter_n - 2 * n_other

                strata = [
                    ('intersection', intersection, n_intersection),
                    ('primary_only', primary_only, n_other),
                    ('modifier_only', modifier_only, n_other),
                ]

                pools = []
                unused = 0
                for name, hadm_ids, budget in strata:
                    if not hadm_ids:
                        unused += budget
                        continue
                    stratum_pool = self._build_pool(hadm_ids)
                    sim = stratum_pool.sim_to_query(query_vec)
                    take = min(budget, stratum_pool.n)
                    if take < budget:
                        unused += budget - take
                    top_idx = np.argsort(sim)[::-1][:take].copy()
                    pools.append((stratum_pool.slice(top_idx), name, take))

                # Redistribute unused slots to intersection pool
                if unused > 0 and pools:
                    first_pool, first_name, first_take = pools[0]
                    parent_pool = self._build_pool(
                        intersection if first_name == 'intersection' else primary
                    )
                    sim = parent_pool.sim_to_query(query_vec)
                    new_take = min(first_take + unused, parent_pool.n)
                    top_idx = np.argsort(sim)[::-1][:new_take].copy()
                    pools[0] = (parent_pool.slice(top_idx), first_name, new_take)

                # Concatenate strata
                all_chunk_ids: list[str] = []
                all_hadm_ids_list: list[NDArray[np.int64]] = []
                all_vectors: list[NDArray[np.float32]] = []
                all_texts: list[str] = []
                all_sections: list[str] = []
                all_meta_dfs: list[pl.DataFrame] = []

                for p, name, take in pools:
                    all_chunk_ids.extend(p.chunk_ids)
                    all_hadm_ids_list.append(p.hadm_ids)
                    all_vectors.append(p.vectors)
                    all_texts.extend(p.texts)
                    all_sections.extend(p.section_names)
                    all_meta_dfs.append(p.metadata_df)

                total = sum(t for _, _, t in pools)
                print(
                    f'  Stratified pool for {icd3}+"{modifier_text}": '
                    + ', '.join(f'{name}={t}' for _, name, t in pools)
                    + f' --> {total} chunks'
                )

                return CandidatePool(
                    chunk_ids=all_chunk_ids,
                    hadm_ids=np.concatenate(all_hadm_ids_list),
                    vectors=np.concatenate(all_vectors),
                    texts=all_texts,
                    section_names=all_sections,
                    metadata_df=pl.concat(all_meta_dfs),
                )

        # Fallback: single stratum (demographic modifier or no modifier)
        pool = self._build_pool(primary)
        sim = pool.sim_to_query(query_vec)
        take = min(prefilter_n, pool.n)
        top_idx = np.argsort(sim)[::-1][:take].copy()
        return pool.slice(top_idx)

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

    builder = CandidatePoolBuilder(con, cfg=EvaluateCfg.load(), device=args.device)
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
