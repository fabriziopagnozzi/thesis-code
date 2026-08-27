from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl

from experiments.medical_dataset_gen.evaluation.eval_plot_data import EvaluationResultLookup
from experiments.medical_dataset_gen.utils.global_schemas import LambdaSelectionCfg
from experiments.medical_dataset_gen.utils.global_utils import get_literals

if TYPE_CHECKING:
    from experiments.medical_dataset_gen.utils.global_schemas import EvalPlotTheme

DEFAULT_EVAL_PLOT_THEME: EvalPlotTheme = 'light'

type EvalPlotCallContext = dict[
    str, pl.DataFrame | Path | LambdaSelectionCfg | EvalPlotTheme | str | EvaluationResultLookup
]

type EvalPlotFileName = Literal[
    # -------------------------------------------
    # retrieval metrics
    'metrics_k_curves_for_lambda',
    'metrics_at_best_lambda_for_k',
    'metrics_heatmap_k_lambda_grid',
    'metrics_heatmap_k_lambda_grid_html',
    'metrics_delta_vs_topk_k_curves_for_lambda',
    'metrics_delta_vs_topk_at_best_lambda_for_k',
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
    # profiles at best lambda
    'profiles_metrics_by_k_at_best_lambda',
    'profiles_diagnostics_by_k_at_best_lambda',
    # -------------------------------------------
]
EVAL_PLOT_FILE_NAMES = set[EvalPlotFileName](get_literals(EvalPlotFileName))

# Default ordered list of evaluation plots to generate and save in the evaluation stage.
DEFAULT_ENABLED_EVAL_PLOT_NAMES: list[EvalPlotFileName] = [
    'metrics_at_best_lambda_for_k',
    'metrics_k_curves_for_lambda',
    # 'metrics_heatmap_k_lambda_grid',
    # 'metrics_heatmap_k_lambda_grid_html',
    # 'metrics_distributions',
    'metrics_delta_vs_topk_k_curves_for_lambda',
    'metrics_delta_vs_topk_at_best_lambda_for_k',
    'diagnostics_at_best_lambda_for_k',
    'diagnostics_k_curves_for_lambda',
    # 'profiles_metrics_by_k_at_best_lambda',
    # 'profiles_diagnostics_by_k_at_best_lambda',
    # 'answer_metrics_at_best_lambda_for_k',
    # 'answer_metrics_k_curves_for_lambda',
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

# Theme palettes shared by every Matplotlib and Plotly evaluation figure.
# Keep all non-data colors here so PNG and HTML outputs remain visually consistent.
EVAL_PLOT_LIGHT_THEME: dict[str, str] = {
    'figure_facecolor': '#FFFFFF',
    'axes_facecolor': '#FFFFFF',
    'panel_highlight_facecolor': '#F4EFE2',
    'annotation_facecolor': '#FFFFFF',
    'text_color': '#111827',
    'muted_text_color': '#444444',
    'tick_color': '#333333',
    'grid_color': '#D0D0D0',
    'heatmap_grid_color': '#FFFFFF',
    'spine_color': '#666666',
    'zero_line_color': '#000000',
    'marker_facecolor': '#FFFFFF',
    'selection_color': '#FFFFFF',
    'heatmap_text_light': '#FFFFFF',
    'heatmap_text_dark': '#111827',
    'fallback_color': '#AAAAAA',
    'plotly_template': 'plotly_white',
}

EVAL_PLOT_DARK_THEME: dict[str, str] = {
    'figure_facecolor': '#0B1020',
    'axes_facecolor': '#111827',
    'panel_highlight_facecolor': '#1E293B',
    'annotation_facecolor': '#0F172A',
    'text_color': '#E5E7EB',
    'muted_text_color': '#A7B0C4',
    'tick_color': '#CBD5E1',
    'grid_color': '#334155',
    'heatmap_grid_color': '#0B1020',
    'spine_color': '#475569',
    'zero_line_color': '#F8FAFC',
    'marker_facecolor': '#111827',
    'selection_color': '#FDE68A',
    'heatmap_text_light': '#F8FAFC',
    'heatmap_text_dark': '#0B1020',
    'fallback_color': '#94A3B8',
    'plotly_template': 'plotly_dark',
}

EVAL_PLOT_THEMES: dict[EvalPlotTheme, dict[str, str]] = {
    'dark': EVAL_PLOT_DARK_THEME,
    'light': EVAL_PLOT_LIGHT_THEME,
}

# Shared line colors, labels, and line styles for retrieval strategies across figures.
EVAL_PLOT_STRATEGY_STYLES: dict[EvalPlotTheme, dict[str, dict[str, str]]] = {
    'dark': {
        'top_k': {'color': '#E5E7EB', 'ls': '--', 'label': 'top-k'},
        'mmr': {'color': '#60A5FA', 'ls': '-', 'label': 'MMR'},
        'fac_loc': {'color': '#F87171', 'ls': '-', 'label': 'FacLoc'},
        'reranker': {'color': '#34D399', 'ls': '-.', 'label': 'Qwen3 reranker'},
    },
    'light': {
        'top_k': {'color': '#333333', 'ls': '--', 'label': 'top-k'},
        'mmr': {'color': '#1F77B4', 'ls': '-', 'label': 'MMR'},
        'fac_loc': {'color': '#D62728', 'ls': '-', 'label': 'FacLoc'},
        'reranker': {'color': '#2CA02C', 'ls': '-.', 'label': 'Qwen3 reranker'},
    },
}

# Backward-compatible default for any external code that imports STRATEGY_STYLE directly.
STRATEGY_STYLE: dict[str, dict[str, str]] = EVAL_PLOT_STRATEGY_STYLES[DEFAULT_EVAL_PLOT_THEME]

# Colormaps are sampled away from their darkest end in plots.py where line
# visibility matters. Viridis remains perceptually ordered on both backgrounds.
EVAL_PLOT_K_COLORMAP = 'viridis'
EVAL_PLOT_HEATMAP_CMAP = 'viridis'
EVAL_PLOT_DIVERGING_CMAP = 'RdBu_r'

# Marker size for raw-lambda figures that draw one curve per k.
FOR_LAMBDA_K_CURVE_MARKER_SIZE = 0.25
FOR_LAMBDA_K_CURVE_BEST_MARKER_SIZE = 18  # points^2 area for matplotlib
# The lambda-grid footer contains only the shared k legend. Keep it close to
# the final subplot row instead of reserving the former explanatory-note space.
LAMBDA_K_CURVE_BOTTOM_MARGIN = 0.025
LAMBDA_K_CURVE_LEGEND_Y = 0.0

# Human-readable titles for metrics when rendering subplot titles and labels.
PLOT_METRIC_TITLES = {
    'FacetCoverage@k': 'FacetCoverage@k',
    'AllFacetCoverageRate@k': 'AllFacetCoverageRate@k',
    'FacetWeightedRecall@k': 'FacetWeightedRecall@k',
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
    'DominantFacetRate': 'DominantFacetRate',
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
DEFAULT_PLOT_GRID_LAYOUTS: dict[EvalPlotFileName, PlotGridLayout] = {
    # -------------------------------------------
    # Fixed rows/cols layout: adjust configs based on how many metrics you define in
    'metrics_at_best_lambda_for_k': PlotGridLayout(2, 3, 4.2, 3.4, 1.4),
    'metrics_delta_vs_topk_at_best_lambda_for_k': PlotGridLayout(2, 3, 4.83, 3.4, 1.4),
    'answer_metrics_at_best_lambda_for_k': PlotGridLayout(1, 3, 4.2, 3.4, 1.4),
    'diagnostics_at_best_lambda_for_k': PlotGridLayout(4, 3, 4.2, 3.4, 1.4),
    # -------------------------------------------
    # Dynamically adapted rows/cols based on num of subplots to handle
    'metrics_k_curves_for_lambda': PlotGridLayout(1, 1, 4.0, 2.0, 2.0),
    'metrics_delta_vs_topk_k_curves_for_lambda': PlotGridLayout(1, 1, 1.0, 2.9, 1.8),
    'answer_metrics_k_curves_for_lambda': PlotGridLayout(1, 1, 4.0, 2.0, 2.0),
    'diagnostics_k_curves_for_lambda': PlotGridLayout(1, 1, 4.0, 2.0, 2.0),
    # ---
    'profiles_metrics_by_k_at_best_lambda': PlotGridLayout(1, 1, 1.0, 5.0, 2.0),
    'profiles_diagnostics_by_k_at_best_lambda': PlotGridLayout(1, 1, 1.0, 5.0, 2.0),
    'metrics_distributions': PlotGridLayout(4, 3, 4.2, 3.4, 1.4),
}


# Main benchmark metrics shown in the primary strategy-comparison figures.
PLOTTED_MAIN_METRIC_NAMES = [
    'FacetCoveragePurity@k',
    'FacetCoverage@k',
    'AllFacetCoverageRate@k',
    'Precision@k',
    'AllFacetCleanRate@k',
    'alpha-nDCG@k',
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
    'FacetWeightedRecall@k',
    'Recall@k',
    'fac',
    'avg_cos',
    'jac',
    # 'DominantFacetRate',
]

# Answer-overlap metrics used only in the auxiliary answer-ROUGE plots.
PLOTTED_ANSWER_ROUGE_METRIC_NAMES = [
    'AnswerROUGE1Recall@k',
    'AnswerROUGE2Recall@k',
    'AnswerROUGE1Precision@k',
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
