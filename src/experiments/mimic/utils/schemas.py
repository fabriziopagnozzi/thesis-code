"""TypedDict schemas for every .parquet file in the MIMIC pipeline.

Each class maps 1-to-1 to a parquet produced/consumed by a pipeline phase.
Use cast() at the start of iter_rows loops to get full IDE inference:

    for row in df.iter_rows(named=True):
        row = cast(GoldAnnotationRow, row)
"""

from typing import TypedDict

# ---------------------------------------------------------------------------
# Phase 1 — corpus construction
# ---------------------------------------------------------------------------


class ConditionStatsRow(TypedDict):
    """conditions_stats.parquet — one row per ICD-10 3-char prefix condition."""

    icd10_3char: str
    condition_name: str
    n_admissions: int
    mean_comorbidity_count: float
    top_comorbidity_mods_json: str  # JSON array of modifier dicts


class ChunkRow(TypedDict):
    """chunks.parquet — one row per overlapping text window."""

    text: str
    chunk_id: str
    note_id: str
    subject_id: int
    hadm_id: int
    section_name: str
    chief_complaint: str | None
    char_count: int
    approx_tokens: int


class _AdmissionMetadataBase(TypedDict):
    """Always-present fields sourced from the admissions table."""

    hadm_id: int
    subject_id: int
    age: float | None
    gender: str
    race: str
    insurance: str
    marital_status: str
    admission_type: str
    discharge_location: str
    hospital_expire_flag: int | None
    primary_icd_code: str
    primary_icd_description: str
    top_icd_descriptions: str


class AdmissionMetadataRow(_AdmissionMetadataBase, total=False):
    """admissions_metadata.parquet — base fields + Charlson fields (null when absent from Charlson view)."""

    age_score: int
    charlson_comorbidity_index: int
    myocardial_infarct: int | None
    congestive_heart_failure: int | None
    peripheral_vascular_disease: int | None
    cerebrovascular_disease: int | None
    dementia: int | None
    chronic_pulmonary_disease: int | None
    rheumatic_disease: int | None
    peptic_ulcer_disease: int | None
    mild_liver_disease: int | None
    severe_liver_disease: int | None
    diabetes_without_cc: int | None
    diabetes_with_cc: int | None
    paraplegia: int | None
    renal_disease: int | None
    malignant_cancer: int | None
    metastatic_solid_tumor: int | None
    aids: int | None


class AdmissionMetaSlimRow(TypedDict):
    """Slim projection of admissions_metadata used in query-prompt grounding."""

    hadm_id: int
    age: float | None
    gender: str
    race: str


# ---------------------------------------------------------------------------
# Phase 2 — embedding (joined row passed to prefix builder)
# ---------------------------------------------------------------------------


class EmbedJoinedRow(ChunkRow, total=False):
    """chunks LEFT JOIN admissions_metadata (selected cols) iterated in embed_whole_corpus.py."""

    age: float | None
    gender: str
    race: str
    primary_icd_description: str
    top_icd_descriptions: str
    admission_type: str | None
    charlson_comorbidity_index: int
    myocardial_infarct: int | None
    congestive_heart_failure: int | None
    peripheral_vascular_disease: int | None
    cerebrovascular_disease: int | None
    dementia: int | None
    chronic_pulmonary_disease: int | None
    rheumatic_disease: int | None
    peptic_ulcer_disease: int | None
    mild_liver_disease: int | None
    severe_liver_disease: int | None
    diabetes_without_cc: int | None
    diabetes_with_cc: int | None
    paraplegia: int | None
    renal_disease: int | None
    malignant_cancer: int | None
    metastatic_solid_tumor: int | None
    aids: int | None


# ---------------------------------------------------------------------------
# Phase 3 — query generation
# ---------------------------------------------------------------------------


class GroundingChunkSample(TypedDict):
    """Single grounding example assembled in _sample_patient."""

    header: str
    text: str
    hadm_id: int


class QueryPromptRow(TypedDict):
    """queries_prompts.parquet — one row per (condition, modifier-set) prompt."""

    query_id: int
    icd10_3char: str
    condition_name: str
    stratum: int
    modifiers_json: str  # JSON list of {text, type} dicts
    n_modifiers: int
    n_condition_admissions: int
    n_condition_chunks: int
    modifier_stats_json: str  # JSON: {modifier_text: {n_admissions: int, n_chunks: int}}
    n_grounding_chunks: int
    grounding_hadm_ids: list[int]
    full_prompt: str


class QueryRow(TypedDict):
    """queries.parquet — QueryPromptRow minus full_prompt, plus query_text."""

    query_id: int
    icd10_3char: str
    condition_name: str
    stratum: int
    modifiers_json: str
    n_modifiers: int
    n_condition_admissions: int
    n_condition_chunks: int
    modifier_stats_json: str  # JSON: {modifier_text: {n_admissions: int, n_chunks: int}}
    n_grounding_chunks: int
    grounding_hadm_ids: list[int]
    query_text: str


class DivergenceStatsRow(QueryRow, total=False):
    """divergence_stats.parquet — QueryRow plus pre-filter divergence metrics."""

    jaccard_div: float
    fac_gap: float
    fac_topk: float
    fac_fl: float
    pool_size: int
    passes_filter: bool


# ---------------------------------------------------------------------------
# Phase 4 — annotation & evaluation
# ---------------------------------------------------------------------------


class GoldAnnotationRow(TypedDict):
    """gold_annotations.parquet — one row per annotated query."""

    query_id: int
    icd10_3char: str
    condition_name: str
    modifiers_json: str
    query_text: str
    facets_json: str  # JSON dict: facet_label → list[chunk_id]
    n_facets: int
    n_gold_chunks: int


class DivergenceMetrics(TypedDict):
    """Return type of compute_divergence in c_filter_queries.py."""

    jaccard: float
    jaccard_div: float
    fac_gap: float
    fac_topk: float
    fac_fl: float
    pool_size: int


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


class EvaluationResultRow(EvaluationMetrics, total=False):
    """evaluation_results.parquet — EvaluationMetrics plus query-level keys."""

    query_id: str
    icd10_3char: str
    n_facets: int


class EvaluationStatsRow(TypedDict):
    """evaluation_stats.parquet — mean metrics per (strategy, lam, k) across all queries."""

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
    """evaluation_best_per_metric.parquet — winning (strategy, k, lam) per metric at each (k, lam)."""

    k: int
    lam: float | None
    best_for: str  # 'AR' | 'WAR' | 'GP' | 'GR'
    strategy: list[str]  # list because ties are possible
    AR: float
    WAR: float
    GP: float
    GR: float


class BestPerMetricFixedLamRow(TypedDict):
    """evaluation_best_per_metric_fixed_lam.parquet — winning (strategy, k) per metric at each fixed lam."""

    lam: float | None
    best_for: str
    strategy: list[str]
    k: list[int]  # list because ties across k values are possible
    AR: float
    WAR: float
    GP: float
    GR: float
