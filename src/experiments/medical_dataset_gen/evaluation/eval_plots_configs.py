from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import polars as pl

from experiments.medical_dataset_gen.schemas.global_config_schemas import LambdaSelectionCfg

type EvalPlotCallContext = dict[str, pl.DataFrame | Path | LambdaSelectionCfg]

type EvalPlotFileName = Literal[
    # -------------------------------------------
    # retrieval metrics
    'metrics_k_curves_for_lambda',
    'metrics_at_best_lambda_for_k',
    'metrics_heatmap_k_lambda_grid',
    'metrics_heatmap_k_lambda_grid_html',
    'metrics_distributions',
    # -------------------------------------------
    # answer metrics
    'answer_metrics_at_best_lambda_for_k',
    'answer_metrics_k_curves_for_lambda',
    # -------------------------------------------
    # diagnostics
    'diagnostics_k_curves_for_lambda',
    'diagnostics_at_best_lambda_for_k',
    # -------------------------------------------
    # deltas
    'delta_vs_topk_metrics_k_curves_for_lambda',
    'delta_vs_topk_metrics_at_best_lambda_for_k',
    # -------------------------------------------
    # profiles at best lambda
    'profiles_metrics_by_k_at_best_lambda',
    'profiles_diagnostics_by_k_at_best_lambda',
    # -------------------------------------------
    # lambda-agreement
    'lambda_agreement',
    'metrics_at_agreeing_lambda_wrt_best_lambda',
    # -------------------------------------------
]
EVAL_PLOT_FILE_NAMES = set[EvalPlotFileName](get_args(EvalPlotFileName.__value__))

# Default ordered list of evaluation plots to generate and save in the evaluation stage.
DEFAULT_ENABLED_EVAL_PLOT_NAMES: list[EvalPlotFileName] = [
    'metrics_at_best_lambda_for_k',
    'metrics_k_curves_for_lambda',
    # 'metrics_heatmap_k_lambda_grid',
    # 'metrics_heatmap_k_lambda_grid_html',
    'metrics_distributions',
    'delta_vs_topk_metrics_k_curves_for_lambda',
    'delta_vs_topk_metrics_at_best_lambda_for_k',
    'diagnostics_at_best_lambda_for_k',
    'diagnostics_k_curves_for_lambda',
    'profiles_metrics_by_k_at_best_lambda',
    'profiles_diagnostics_by_k_at_best_lambda',
    # 'lambda_agreement',
    # 'metrics_at_agreeing_lambda_wrt_best_lambda',
    'answer_metrics_at_best_lambda_for_k',
    'answer_metrics_k_curves_for_lambda',
]

ANSWER_ROUGE_EVAL_PLOT_FILE_NAMES = {
    'answer_metrics_at_best_lambda_for_k',
    'answer_metrics_k_curves_for_lambda',
}

UNKNOWN_DEFAULT_ENABLED_EVAL_PLOT_NAMES = sorted(
    set(DEFAULT_ENABLED_EVAL_PLOT_NAMES) - EVAL_PLOT_FILE_NAMES
)
if UNKNOWN_DEFAULT_ENABLED_EVAL_PLOT_NAMES:
    unknown = ', '.join(UNKNOWN_DEFAULT_ENABLED_EVAL_PLOT_NAMES)
    raise ValueError(f'Unknown default-enabled evaluation plot name(s): {unknown}')

# Shared line colors, labels, and line styles for retrieval strategies across figures.
STRATEGY_STYLE: dict[str, dict[str, str]] = {
    'top_k': {'color': '#333333', 'ls': '--', 'label': 'top-k'},
    'mmr': {'color': '#1f77b4', 'ls': '-', 'label': 'MMR'},
    'fac_loc': {'color': '#d62728', 'ls': '-', 'label': 'FacLoc'},
}

# Marker size for raw-lambda figures that draw one curve per k.
FOR_LAMBDA_K_CURVE_MARKER_SIZE = 0.25

# Human-readable titles for metrics when rendering subplot titles and labels.
PLOT_METRIC_TITLES = {
    'FacetCoverage@k': 'FacetCoverage@k',
    'MeanFacetRecall@k': 'FacetRecall@k',
    'FacetCoveragePurity@k': 'FacetCoveragePurity@k',
    'AllFacetCleanRate@k': 'AllFacetCleanRate@k',
    'Precision@k': 'Precision@k',
    'DistractorRate': 'DistractorRate@k',
    'F1@k': 'F1@k',
    'Recall@k': 'Recall@k',
    'alpha-nDCG@k': 'alpha-nDCG@k',
    'AnswerROUGE1F1@k': 'Answer ROUGE-1 F1@k',
    'AnswerROUGE2Recall@k': 'Answer ROUGE-2 Recall@k',
    'NearMissDistractorRate': 'NearMissDistractorRate',
    'BackgroundOutlierRate': 'BackgroundOutlierRate',
    'PrimaryAxisRate': 'PrimaryAxisRate',
    'CalibratedFacetRate': 'CalibratedFacetRate',
    'RedundantGoldRate': 'RedundantGoldRate',
    'fac': 'Facility-Location Objective',
    'avg_cos': 'Average Query Cosine',
    'jac': 'Jaccard vs top-k',
    'AnswerROUGE1Recall@k': 'Answer ROUGE-1 Recall@k',
    'AnswerROUGE1Precision@k': 'Answer ROUGE-1 Precision@k',
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
# - `plot_metrics_heatmap_k_lambda_grid(...)` uses a custom GridSpec/figure composition.
# - `plot_metrics_heatmap_k_lambda_grid_html(...)` is plotly-based.
# - `plot_lambda_agreement(...)` depends on lambda counts and shared y-axis behavior.
DEFAULT_PLOT_GRID_LAYOUTS: dict[EvalPlotFileName, PlotGridLayout] = {
    # -------------------------------------------
    # Fixed rows/cols layout: adjust configs based on how many metrics you define in
    'metrics_at_best_lambda_for_k': PlotGridLayout(2, 3, 4.2, 3.4, 1.4),
    'delta_vs_topk_metrics_at_best_lambda_for_k': PlotGridLayout(2, 3, 4.83, 3.4, 1.4),
    'answer_metrics_at_best_lambda_for_k': PlotGridLayout(1, 3, 4.2, 3.4, 1.4),
    'diagnostics_at_best_lambda_for_k': PlotGridLayout(3, 3, 4.2, 3.4, 1.4),
    # -------------------------------------------
    # Dynamically adapted rows/cols based on num of subplots to handle
    'metrics_k_curves_for_lambda': PlotGridLayout(1, 1, 4.0, 2.0, 2.0),
    'delta_vs_topk_metrics_k_curves_for_lambda': PlotGridLayout(1, 1, 1.0, 2.9, 1.8),
    'answer_metrics_k_curves_for_lambda': PlotGridLayout(1, 1, 4.0, 2.0, 2.0),
    'diagnostics_k_curves_for_lambda': PlotGridLayout(1, 1, 4.0, 2.0, 2.0),
    # ---
    'profiles_metrics_by_k_at_best_lambda': PlotGridLayout(1, 1, 1.0, 5.0, 2.0),
    'profiles_diagnostics_by_k_at_best_lambda': PlotGridLayout(1, 1, 1.0, 5.0, 2.0),
    'metrics_distributions': PlotGridLayout(4, 3, 4.2, 3.4, 1.4),
    'lambda_agreement': PlotGridLayout(1, 1, 1.0, 1.0, 1.7),
    'metrics_at_agreeing_lambda_wrt_best_lambda': PlotGridLayout(2, 1, 1.0, 1.0, 1.6),
}


# Main benchmark metrics shown in the primary strategy-comparison figures.
PLOTTED_MAIN_METRIC_NAMES = [
    'FacetCoveragePurity@k',
    'AllFacetCleanRate@k',
    'FacetCoverage@k',
    'Precision@k',
    'Recall@k',
    'alpha-nDCG@k',
    # 'MeanFacetRecall@k',
    # 'F1@k',
    # 'AnswerROUGE1F1@k',
    # 'AnswerROUGE2Recall@k',
]

# Diagnostic metrics used to explain wins, failures, and retrieval behavior.
PLOTTED_DIAGNOSTIC_METRIC_NAMES = [
    'DistractorRate',
    'NearMissDistractorRate',
    'BackgroundOutlierRate',
    'PrimaryAxisRate',
    'RedundantGoldRate',
    'fac',
    'avg_cos',
    'jac',
    # 'CalibratedFacetRate',
]

# Answer-overlap metrics used only in the auxiliary answer-ROUGE plots.
PLOTTED_ANSWER_ROUGE_METRIC_NAMES = [
    'AnswerROUGE1Recall@k',
    'AnswerROUGE2Recall@k',
    'AnswerROUGE1Precision@k',  # this is too low like emilia said in the 30/06 call, see how to treat it better
]

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
