"""Static configuration for experiment-comparison aggregation and figures."""

from experiments.medical_dataset_gen.analysis.analysis_constants import DeltaMetricLabel
from experiments.medical_dataset_gen.analysis.models import BudgetCategory, DeltaMetricPlotSpec

BUDGET_CATEGORIES: tuple[BudgetCategory, ...] = ('low_budget', 'medium_budget', 'high_budget')
BUDGET_CATEGORY_LABELS: dict[BudgetCategory, str] = {
    'low_budget': 'Low Budget',
    'medium_budget': 'Medium Budget',
    'high_budget': 'High Budget',
}
AGGREGATE_PLOT_EXCLUDED_FAMILY_LABELS = frozenset({
    'Budget sweeps',
    'Embedding comparison',
    'Unknown',
})
LEGACY_LOW_BUDGET_TOKEN = ''.join(('head', 'line'))

# This tuple is the source of truth for report-facing comparison metrics. Add,
# remove, or reorder metrics here to affect markdown, CSV aggregate summaries,
# LaTeX tables, and aggregate/per-metric plots together.
REPORT_METRIC_SPECS: tuple[DeltaMetricPlotSpec, ...] = (
    DeltaMetricPlotSpec('FCP', 'FCP', 'fcp', 'FacetCoveragePurity@k'),
    DeltaMetricPlotSpec('FacetCoverage', 'FacetCoverage@k', 'facet_coverage', 'FacetCoverage@k'),
    DeltaMetricPlotSpec(
        'AllFacetCleanRate',
        'AllFacetCleanRate@k',
        'all_facet_clean_rate',
        'AllFacetCleanRate@k',
    ),
    DeltaMetricPlotSpec('Precision', 'Precision@k', 'precision', 'Precision@k'),
    # DeltaMetricPlotSpec('Recall', 'Recall@k', 'recall', 'Recall@k'),
    DeltaMetricPlotSpec('alpha_nDCG', 'alpha-nDCG@k', 'alpha_ndcg', 'alpha-nDCG@k'),
)
REPORT_METRIC_LABELS: tuple[DeltaMetricLabel, ...] = tuple(
    spec.metric_label for spec in REPORT_METRIC_SPECS
)
REPORT_METRIC_LABEL_SET = frozenset(REPORT_METRIC_LABELS)
REPORT_METRIC_NAME_TO_LABEL: dict[str, DeltaMetricLabel] = {
    spec.source_metric_name: spec.metric_label for spec in REPORT_METRIC_SPECS
}
