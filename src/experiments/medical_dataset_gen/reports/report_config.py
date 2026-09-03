"""Static configuration for experiment-comparison aggregation and figures."""

from dataclasses import dataclass

from experiments.medical_dataset_gen.reports.analysis_constants import DeltaMetricLabel
from experiments.medical_dataset_gen.reports.models import BudgetCategory, DeltaMetricPlotSpec

BUDGET_CATEGORIES: tuple[BudgetCategory, ...] = ('low_budget', 'medium_budget', 'high_budget')
BUDGET_CATEGORY_LABELS: dict[BudgetCategory, str] = {
    'low_budget': 'Low Budget',
    'medium_budget': 'Medium Budget',
    'high_budget': 'High Budget',
}
# The thesis-level low-budget comparison is fixed across experiments so its
# aggregates and wording contrasts share one retrieval budget.
LOW_BUDGET_K = 6
AGGREGATE_PLOT_EXCLUDED_FAMILY_LABELS = frozenset(
    {
        'Budget sweeps',
        'Embedding comparison',
        'Interaction experiments',
        'Unknown',
    }
)


@dataclass(frozen=True)
class EmbeddingModelReportSpec:
    model_name: str
    display_label: str
    macro_token: str
    wording_macro_token: str
    color: str


EMBEDDING_MODEL_REPORT_SPECS: tuple[EmbeddingModelReportSpec, ...] = (
    EmbeddingModelReportSpec(
        'Qwen/Qwen3-Embedding-0.6B',
        'Qwen3-0.6B',
        'QwenSmall',
        'Qwen',
        '#3D71B7',
    ),
    EmbeddingModelReportSpec(
        'Qwen/Qwen3-Embedding-4B',
        'Qwen3-4B',
        'QwenFourB',
        'QwenFourB',
        '#6F58B0',
    ),
    EmbeddingModelReportSpec(
        'abhinand/MedEmbed-large-v0.1',
        'MedEmbed-large',
        'MedEmbed',
        'MedEmbed',
        '#C65A3E',
    ),
    EmbeddingModelReportSpec(
        'multi-qa-mpnet-base-cos-v1',
        'MultiQA-MPNet',
        'MultiMpnet',
        'MultiMpnet',
        '#2D8871',
    ),
    EmbeddingModelReportSpec('BAAI/bge-m3', 'BGE-M3', 'Bge', 'Bge', '#7A7A7A'),
    EmbeddingModelReportSpec(
        'Qwen/Qwen3-Embedding-8B',
        'Qwen3-8B',
        'QwenEightB',
        'QwenEightB',
        '#8E79C6',
    ),
    EmbeddingModelReportSpec(
        'jinaai/jina-embeddings-v5-text-small',
        'Jina-v5-small',
        'Jina',
        'Jina',
        '#B279A2',
    ),
)
_EMBEDDING_MODEL_REPORT_SPEC_BY_NAME = {
    spec.model_name: spec for spec in EMBEDDING_MODEL_REPORT_SPECS
}
PREFERRED_EMBEDDING_MODEL_ORDER: tuple[str, ...] = tuple(
    spec.model_name for spec in EMBEDDING_MODEL_REPORT_SPECS
)
OBJECTIVE_COLORS: dict[str, str] = {
    'top_k': '#333333',
    'mmr': '#1F77B4',
    'fac_loc': '#D62728',
}


def embedding_model_display_label(model_name: str) -> str:
    spec = _EMBEDDING_MODEL_REPORT_SPEC_BY_NAME.get(model_name)
    return spec.display_label if spec is not None else model_name.rsplit('/', 1)[-1]


def embedding_model_macro_token(model_name: str) -> str | None:
    spec = _EMBEDDING_MODEL_REPORT_SPEC_BY_NAME.get(model_name)
    return spec.macro_token if spec is not None else None


def embedding_model_wording_macro_token(model_name: str) -> str | None:
    spec = _EMBEDDING_MODEL_REPORT_SPEC_BY_NAME.get(model_name)
    return spec.wording_macro_token if spec is not None else None


def embedding_model_color(model_name: str) -> str:
    spec = _EMBEDDING_MODEL_REPORT_SPEC_BY_NAME.get(model_name)
    return spec.color if spec is not None else '#808080'


# Backward-compatible alias for older imports; model inclusion is controlled by
# report discovery plus --embedding-models, not by this tuple.
EMBEDDING_MODEL_FACETED_PLOT_MODELS: tuple[str, ...] = PREFERRED_EMBEDDING_MODEL_ORDER
LEGACY_LOW_BUDGET_TOKEN = ''.join(('head', 'line'))

# Fixed color ranges for aggregate heatmaps. The ranges are shared across
# reports so the same color intensity has the same interpretation when comparing
# different wording triples or experiment-family slices. Values outside the
# configured range are saturated; the numeric cell labels remain authoritative.
DELTA_HEATMAP_ABS_SCALE_BY_VALUE_FIELD: dict[str, float] = {
    'MeanDeltaFacLocMMR': 0.5,
    'MeanDeltaFacLocTopK': 1.0,
    'MeanDeltaMMRTopK': 1.0,
}
DELTA_HEATMAP_DEFAULT_ABS_SCALE = 1.0

# This tuple is the source of truth for report-facing comparison metrics. Add,
# remove, or reorder metrics here to affect markdown, CSV aggregate summaries,
# LaTeX tables, and aggregate/per-metric plots together.
REPORT_METRIC_SPECS: tuple[DeltaMetricPlotSpec, ...] = (
    DeltaMetricPlotSpec('FCP', 'FCP', 'fcp', 'FacetCoveragePurity@k'),
    DeltaMetricPlotSpec('FacetCoverage', 'FacetCoverage@k', 'facet_coverage', 'FacetCoverage@k'),
    DeltaMetricPlotSpec(
        'AllFacetCoverageRate',
        'AllFacetCoverageRate@k',
        'all_facet_coverage_rate',
        'AllFacetCoverageRate@k',
    ),
    DeltaMetricPlotSpec(
        'AllFacetCleanRate',
        'AllFacetCleanRate@k',
        'all_facet_clean_rate',
        'AllFacetCleanRate@k',
    ),
    DeltaMetricPlotSpec('Precision', 'Precision@k', 'precision', 'Precision@k'),
    # DeltaMetricPlotSpec(
    #     'FacetWeightedRecall',
    #     'FacetWeightedRecall@k',
    #     'facet_weighted_recall',
    #     'FacetWeightedRecall@k',
    # ),
    # DeltaMetricPlotSpec('Recall', 'Recall@k', 'recall', 'Recall@k'),
    DeltaMetricPlotSpec('alpha_nDCG', 'alpha-nDCG@k', 'alpha_ndcg', 'alpha-nDCG@k'),
    DeltaMetricPlotSpec(
        'NearMissDistractorRate',
        'NearMissDistractorRate',
        'near_miss_distractor_rate',
        'NearMissDistractorRate',
        higher_is_better=False,
    ),
    DeltaMetricPlotSpec(
        'BackgroundOutlierRate',
        'BackgroundOutlierRate',
        'background_outlier_rate',
        'BackgroundOutlierRate',
        higher_is_better=False,
    ),
)
REPORT_METRIC_LABELS: tuple[DeltaMetricLabel, ...] = tuple(
    spec.metric_label for spec in REPORT_METRIC_SPECS
)
REPORT_METRIC_LABEL_SET = frozenset(REPORT_METRIC_LABELS)
REPORT_METRIC_NAME_TO_LABEL: dict[str, DeltaMetricLabel] = {
    spec.source_metric_name: spec.metric_label for spec in REPORT_METRIC_SPECS
}
REPORT_METRIC_LABEL_TO_SPEC: dict[DeltaMetricLabel, DeltaMetricPlotSpec] = {
    spec.metric_label: spec for spec in REPORT_METRIC_SPECS
}
