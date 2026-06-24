from typing import Literal, get_args

# Canonical geometry-plot names used by the CLI selector and plot dispatch.
type GeomPlotFileName = Literal[
    'cluster_quality_overview',
    'full_strategy_selection_overlay',
    'query_overview_4panel',
    'strategy_overlay',
]
GEOM_PLOT_FILE_NAMES = set[GeomPlotFileName](get_args(GeomPlotFileName.__value__))

POINT_DISTRACTOR_TYPE_COLORS = {
    'same_condition_wrong_subgroup': '#f3a6a6',
    'same_subgroup_wrong_condition': '#d62728',
    'same_axis_wrong_condition': '#8c1d18',
    'same_condition_wrong_axis': '#ff7f0e',
    'hard_distractor': '#d62728',
}
BACKGROUND_OUTLIER_LABEL_ID = 'background_clinical_cluster'
BACKGROUND_OUTLIER_ROLE = 'background_outlier'
BACKGROUND_OUTLIER_COLOR = '#d62728'
SAME_CONDITION_WRONG_AXIS_LABEL_ID = 'same_condition_wrong_axis'
SAME_CONDITION_WRONG_AXIS_ROLE = 'same_condition_wrong_axis'
SAME_CONDITION_WRONG_AXIS_COLOR = '#ff7f0e'
UNSELECTED_BACKGROUND_COLOR = '#b8b8b8'
