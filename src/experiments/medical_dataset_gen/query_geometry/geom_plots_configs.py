from typing import Literal, get_args

# Canonical geometry-plot names used by the CLI selector and plot dispatch.
type GeomPlotFileName = Literal[
    'cluster_quality_overview',
    'full_strategy_selection_overlay',
    'query_overview_4panel',
    'strategy_overlay',
]
GEOM_PLOT_FILE_NAMES = set[GeomPlotFileName](get_args(GeomPlotFileName.__value__))

FIXED_LABEL_COLORS = {
    'soft distractor: same condition, wrong subgroup': '#f3a6a6',
    'hard distractor: wrong condition, same subgroup': '#d62728',
    'hard distractor: wrong condition, same answer axis': '#8c1d18',
    'background outlier: clinical cluster': '#d62728',
    'hard distractor': '#d62728',
    'off-query wrong-condition chunks': '#d62728',
}
DISTRACTOR_LABELS = set(FIXED_LABEL_COLORS)
BACKGROUND_OUTLIER_LABEL = 'background outlier: clinical cluster'
BACKGROUND_OUTLIER_LABEL_ID = 'background_clinical_cluster'
BACKGROUND_OUTLIER_ROLE = 'background_outlier'
BACKGROUND_OUTLIER_COLOR = '#d62728'
UNSELECTED_BACKGROUND_COLOR = '#b8b8b8'
