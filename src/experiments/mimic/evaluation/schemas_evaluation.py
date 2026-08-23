from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel

from experiments.mimic.global_configs import load_default_config
from experiments.mimic.utils.chunk_pools import ChunkPoolRetrievalResult
from experiments.mimic.utils.prompts_default import MimicDefaultPrompts
from helpers.query_algorithms import ScoringFunction

__all__ = ['ChunkPoolRetrievalResult']


class GoldAnnotationCfg(BaseModel):
    model_config = {'populate_by_name': True, 'extra': 'ignore'}

    batch_size: int
    resume_batch_size: int | None = None
    min_per_modifier: int = 0
    num_ctx: int | None = None
    num_predict: int | None = None
    model: str | None = None
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    think: bool = False
    stream: bool = False
    # Map phase: fact extraction per (batch, facet)
    fact_extract_system_prompt: str = MimicDefaultPrompts.fact_extract_system
    fact_extract_template: str = MimicDefaultPrompts.fact_extract_template
    # Reduce phase: unified comparative answer synthesis per query
    answer_system_prompt: str = MimicDefaultPrompts.answer_gen_system
    answer_gen_template: str = MimicDefaultPrompts.answer_gen_template
    dump_prompts: bool = True

    @classmethod
    def load(cls) -> GoldAnnotationCfg:
        return cls(**load_default_config(key='evaluation')['gold_annotation'])


class EvaluateCfg(BaseModel):
    model_config = {'extra': 'ignore'}

    strategies: list[ScoringFunction]
    k_values: list[int]
    lam_values: list[float]
    gold_mode: Literal['llm', 'structural'] = 'llm'

    pool_preretrieval_mode: Literal['full_corpus', 'primary_condition_restricted'] = (
        'primary_condition_restricted'
    )
    """
        Valid when gold_mode == 'llm'.
        Decides how to construct the evaluation pool:
           - either drawing from the full_corpus
           - or first restricting to the primary_condition for the query and
         In both cases global_cfg.prefilter_n chunks are retrieved by cosine similarity and then merged with the full golden chunks set to guarantee full reachability of golden chunks
         (NOTE: TBD if this makes sense)
    """
    figures_subdir: str = '.'

    @classmethod
    def load(cls) -> EvaluateCfg:
        return cls(**load_default_config(key='evaluation'))


class QueryEvalResult(TypedDict):
    """One element of the list returned by evaluate_query."""

    strategy: str
    k: int
    lam: float | None
    aspect_recall: float
    weighted_aspect_recall: float
    gold_precision: float
    gold_recall: float
    fac_cov_score: float
    avg_cos: float
    jaccard_vs_topk: float
    n_unique_hadms: int
    answer_rouge1_recall: float
    answer_rouge1_precision: float
    answer_tfidf_cosine: float
    answer_rouge1_f1: float


class EvaluationRsesultRow(QueryEvalResult, total=False):
    """evaluation_results.parquet - EvaluationMetrics plus query-level keys."""

    query_id: str
    icd10_3char: str
    n_facets: int


class EvaluationStatsRow(TypedDict):
    """evaluation_stats.parquet - mean metrics per (strategy, lam, k) across all queries."""

    strategy: str
    lam: float | None
    AR: float
    WAR: float
    GP: float
    GR: float
    fac: float
    cos: float
    jac: float
    ans_rouge1_rec: float
    ans_rouge1_prec: float
    ans_tfidf: float
    ans_rouge1_f1: float
    n: int
    k: int


class BestPerMetricRow(TypedDict):
    """evaluation_best_per_metric.parquet - winning (strategy, k, lam) per metric at each (k, lam)."""

    k: int
    lam: float | None
    best_for: str
    strategy: list[str]  # list because ties are possible
    AR: float
    WAR: float
    GP: float
    GR: float
    ans_rouge1_rec: float
    ans_rouge1_prec: float
    ans_tfidf: float
    ans_rouge1_f1: float


class BestPerMestricFixedLamRow(TypedDict):
    """evaluation_best_per_metric_fixed_lam.parquet - winning (strategy, k) per metric at each fixed lam."""

    lam: float | None
    best_for: str
    strategy: list[str]
    k: list[int]  # list because ties across k values are possible
    AR: float
    WAR: float
    GP: float
    GR: float
    ans_rouge1_rec: float
    ans_rouge1_prec: float
    ans_tfidf: float
    ans_rouge1_f1: float


class ExtractedFact(TypedDict):
    chunk_id: str
    fact: str


type AnnotationCacheKey = tuple[int, str, tuple[str, ...]]
