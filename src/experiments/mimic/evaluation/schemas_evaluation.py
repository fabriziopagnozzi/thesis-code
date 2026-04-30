from typing import Literal, TypedDict

from pydantic import BaseModel

from experiments.mimic.global_configs import load_default_config
from experiments.mimic.utils.prompts_default import MimicDefaultPrompts
from helpers.query_algorithms import ScoringFunction


class GoldAnnotationCfg(BaseModel):
    model_config = {'populate_by_name': True, 'extra': 'ignore'}

    batch_size: int
    resume_batch_size: int | None = None
    wide_pool_n: int = 10000
    final_pool_n: int = 3000
    min_per_modifier: int = 50
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

    @classmethod
    def load(cls) -> GoldAnnotationCfg:
        return cls(**load_default_config(key='queries')['gold_annotation'])


class EvaluateCfg(BaseModel):
    strategies: list[ScoringFunction]
    k_values: list[int]
    lam_values: list[float]
    gold_mode: Literal['llm', 'structural'] = 'llm'

    @classmethod
    def load(cls) -> EvaluateCfg:
        return cls(**load_default_config(key='evaluation'))


class EvaluationMetrics(TypedDict):
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


class EvaluationRsesultRow(EvaluationMetrics, total=False):
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
    n: int
    k: int


class BestPerMetricRow(TypedDict):
    """evaluation_best_per_metric.parquet - winning (strategy, k, lam) per metric at each (k, lam)."""

    k: int
    lam: float | None
    best_for: str  # 'AR' | 'WAR' | 'GP' | 'GR'
    strategy: list[str]  # list because ties are possible
    AR: float
    WAR: float
    GP: float
    GR: float


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
