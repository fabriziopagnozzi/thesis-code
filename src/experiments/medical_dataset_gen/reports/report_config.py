"""Static configuration for experiment-comparison aggregation and figures."""

from experiments.medical_dataset_gen.reports.analysis_constants import DeltaMetricLabel
from experiments.medical_dataset_gen.reports.models import BudgetCategory, DeltaMetricPlotSpec

BUDGET_CATEGORIES: tuple[BudgetCategory, ...] = ('low_budget', 'medium_budget', 'high_budget')
BUDGET_CATEGORY_LABELS: dict[BudgetCategory, str] = {
    'low_budget': 'Low Budget',
    'medium_budget': 'Medium Budget',
    'high_budget': 'High Budget',
}
AGGREGATE_PLOT_EXCLUDED_FAMILY_LABELS = frozenset(
    {
        'Budget sweeps',
        'Embedding comparison',
        'Unknown',
    }
)
EMBEDDING_MODEL_FACETED_PLOT_MODELS: tuple[str, ...] = (
    'BAAI/bge-m3',
    'Qwen/Qwen3-Embedding-0.6B',
)
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
