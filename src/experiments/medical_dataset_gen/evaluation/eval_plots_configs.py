from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import polars as pl

type EvalPlotCallContext = dict[str, pl.DataFrame | Path]

# Canonical plot output names used by the CLI selector and plot dispatch.
type EvalPlotFileName = Literal[
    'strategy_comparison',
    'strategy_comparison_heatmap',
    'strategy_comparison_heatmap_html',
    'lambda_sensitivity_metrics',
    'lambda_sensitivity_diagnostics',
    'per_query_distributions',
    'gain_over_topk',
    'gain_over_topk_html',
    'gain_over_topk_best_facet_hit_lambda',
    'lambda_agreement',
    'selection_diagnostics',
    'answer_rouge_comparison',
    'answer_rouge_lambda_sensitivity',
]
EVAL_PLOT_FILE_NAMES = set[EvalPlotFileName](get_args(EvalPlotFileName.__value__))

# Shared line colors, labels, and line styles for retrieval strategies across figures.
STRATEGY_STYLE: dict[str, dict[str, str]] = {
    'top_k': {'color': '#333333', 'ls': '--', 'label': 'top-k'},
    'mmr': {'color': '#1f77b4', 'ls': '-', 'label': 'MMR'},
    'fac_loc': {'color': '#d62728', 'ls': '-', 'label': 'FacLoc'},
}

# Human-readable titles for metrics when rendering subplot titles and labels.
PLOT_METRIC_TITLES = {
    'MeanFacetHitRate@k': 'MeanFacetHitRate@k',
    'MeanFacetRecall@k': 'MeanFacetRecall@k',
    'DistractorRate': 'DistractorRate@k',
    'Recall@k': 'Recall@k',
    'alpha-nDCG@k': 'alpha-nDCG@k',
    'AnswerROUGE2Recall@k': 'Answer ROUGE-2 Recall@k',
    'NearMissDistractorRate': 'NearMissDistractorRate',
    'BackgroundOutlierRate': 'BackgroundOutlierRate',
    'DominantFacetRate': 'DominantFacetRate',
    'RedundantGoldRate': 'RedundantGoldRate',
    'fac': 'Facility-Location Objective',
    'avg_cos': 'Average Query Cosine',
    'jac': 'Jaccard vs top-k',
    'AnswerROUGE1Recall@k': 'Answer ROUGE-1 Recall@k',
    'AnswerROUGE1Precision@k': 'Answer ROUGE-1 Precision@k',
    'MacroFacetAnswerROUGE1Recall@k': 'Macro Facet Answer ROUGE-1 Recall@k',
}


# Reusable subplot geometry for standard matplotlib grid figures.
@dataclass(frozen=True)
class PlotGridLayout:
    rows: int
    cols: int
    width_per_col: float
    height_per_row: float
    footer_height: float


# Per-plot grid defaults for figures that use regular matplotlib subplot grids.
# Fixed-grid plots use these `rows` / `cols` values directly.
# Dynamic-grid plots may override `rows` / `cols` in `plots.py` when the panel count depends on data.
# Leftover custom figures keep their sizing logic in `plots.py`:
# - `plot_strategy_comparison_heatmap(...)` uses a custom GridSpec/figure composition.
# - `plot_strategy_comparison_heatmap_html(...)` & `plot_gain_over_topk_html(...)` are plotly-based.
# - `plot_lambda_agreement(...)` depends on lambda counts and shared y-axis behavior.
DEFAULT_PLOT_GRID_LAYOUTS: dict[EvalPlotFileName, PlotGridLayout] = {
    'strategy_comparison': PlotGridLayout(2, 3, 4.2, 3.4, 1.4),
    'lambda_sensitivity_metrics': PlotGridLayout(1, 1, 4.0, 2.0, 2.0),
    'lambda_sensitivity_diagnostics': PlotGridLayout(1, 1, 4.0, 2.0, 2.0),
    'per_query_distributions': PlotGridLayout(2, 3, 4.2, 3.4, 1.4),
    'gain_over_topk': PlotGridLayout(1, 1, 1.0, 2.9, 1.8),
    'gain_over_topk_best_facet_hit_lambda': PlotGridLayout(2, 3, 4.83, 3.4, 1.4),
    'selection_diagnostics': PlotGridLayout(3, 3, 4.2, 3.4, 1.4),
    'answer_rouge_comparison': PlotGridLayout(2, 3, 4.2, 3.4, 1.4),
    'answer_rouge_lambda_sensitivity': PlotGridLayout(1, 1, 4.0, 2.0, 2.0),
    'lambda_agreement': PlotGridLayout(1, 1, 1.0, 1.0, 1.7),
}


# Main benchmark metrics shown in the primary strategy-comparison figures.
PLOTTED_MAIN_METRIC_NAMES = [
    'MeanFacetHitRate@k',
    'MeanFacetRecall@k',
    'DistractorRate',
    'Recall@k',
    'alpha-nDCG@k',
    'AnswerROUGE2Recall@k',
]

# Diagnostic metrics used to explain wins, failures, and retrieval behavior.
PLOTTED_DIAGNOSTIC_METRIC_NAMES = [
    'DistractorRate',
    'NearMissDistractorRate',
    'BackgroundOutlierRate',
    'DominantFacetRate',
    'RedundantGoldRate',
    'fac',
    'avg_cos',
    'jac',
]

# Answer-overlap metrics used only in the auxiliary answer-ROUGE plots.
PLOTTED_ANSWER_ROUGE_METRIC_NAMES = [
    'AnswerROUGE1Recall@k',
    'AnswerROUGE1Precision@k',
    'AnswerROUGE2Recall@k',
    'MacroFacetAnswerROUGE1Recall@k',
]

# Metric ranking order used when selecting the best lambda / best-performing rows.
PRIMARY_SORT = [
    'MeanFacetHitRate@k',
    'Precision@k',
    'DistractorRate',
    'alpha-nDCG@k',
]
# Sort direction paired with `PRIMARY_SORT`; `True` means descending.
PRIMARY_DESC = [True, True, False, True]

# Shared note describing the lambda-selection policy shown in figure footers.
LAMBDA_POLICY_NOTE = 'lambda*: max mean MeanFacetHitRate@k within strategy x k'


type PlotMetricSpec = tuple[str, str, str, bool]


class NamedPlotMetric:
    def __init__(
        self,
        *,
        stats_col: str,
        result_col: str,
        title: str,
        higher_is_better: bool,
    ) -> None:
        self.stats_col = stats_col
        self.result_col = result_col
        self.title = title
        self.higher_is_better = higher_is_better

    def as_tuple(self) -> PlotMetricSpec:
        return (self.stats_col, self.result_col, self.title, self.higher_is_better)
