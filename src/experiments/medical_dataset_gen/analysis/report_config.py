"""Static configuration for experiment-comparison aggregation and figures."""

from experiments.medical_dataset_gen.analysis.analysis_constants import DeltaMetricLabel
from experiments.medical_dataset_gen.analysis.models import BudgetCategory, DeltaMetricPlotSpec

BUDGET_CATEGORIES: tuple[BudgetCategory, ...] = ('low_budget', 'medium_budget', 'high_budget')
BUDGET_CATEGORY_LABELS: dict[BudgetCategory, str] = {
    'low_budget': 'Low Budget',
    'medium_budget': 'Medium Budget',
    'high_budget': 'High Budget',
}
AGGREGATE_METRIC_ORDER: tuple[DeltaMetricLabel, ...] = (
    'FCP',
    'FacetCoverage',
    'Precision',
    'alpha_nDCG',
    'AllFacetCleanRate',
    'Recall',
)
AGGREGATE_PLOT_EXCLUDED_FAMILY_LABELS = frozenset(
    {
        'Budget sweeps',
        'Embedding comparison',
        'Unknown',
    }
)
LEGACY_LOW_BUDGET_TOKEN = ''.join(('head', 'line'))

DELTA_METRIC_PLOT_SPECS: tuple[DeltaMetricPlotSpec, ...] = (
    DeltaMetricPlotSpec('FCP', 'FCP', 'fcp'),
    DeltaMetricPlotSpec('FacetCoverage', 'FacetCoverage@k', 'facet_coverage'),
    DeltaMetricPlotSpec('AllFacetCleanRate', 'AllFacetCleanRate@k', 'all_facet_clean_rate'),
    DeltaMetricPlotSpec('Precision', 'Precision@k', 'precision'),
    DeltaMetricPlotSpec('Recall', 'Recall@k', 'recall'),
    DeltaMetricPlotSpec('alpha_nDCG', 'alpha-nDCG@k', 'alpha_ndcg'),
)
