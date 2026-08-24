from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

from experiments.medical_dataset_gen.dataset_generation.schemas import ClusterRole
from experiments.medical_dataset_gen.utils.global_utils import get_literals

# Canonical geometry-plot names used by the CLI selector and plot dispatch.
type GeomPlotName = Literal[
    'candidate_pool_umap',
    'candidate_pool_umap_with_legend',
    'cluster_quality_overview',
    'full_strategy_selection_overlay',
    'pairwise_cosine_heatmap',
    'query_cosine_heatmap',
    'query_overview_4panel',
    'strategy_overlay',
]
GEOM_PLOT_FILE_NAMES = set[GeomPlotName](get_literals(GeomPlotName))


@dataclass(frozen=True, slots=True)
class GeomPlotsSettings:
    use_title: bool = False
    unselected_data_point_color: str = '#a1a1a1'

    candidate_pool_umap_only_figure_size: tuple[float, float] = (6.4, 6.2)
    candidate_pool_umap_figure_size: tuple[float, float] = (13.0, 5.6)
    umap_axis_label_font_size: int = 15
    umap_tick_label_font_size: int = 12
    pairwise_cosine_figure_size: tuple[float, float] = (7.6, 6.8)
    pairwise_cosine_cmap: str = 'coolwarm'
    pairwise_cosine_vmin: float = -1.0
    pairwise_cosine_vmax: float = 1.0
    query_cosine_heatmap_figure_size: tuple[float, float] = (6.9, 6.2)
    query_cosine_colorbar_width_in: float = 0.28
    query_cosine_colorbar_pad_in: float = 0.16
    query_cosine_colorbar_label_font_size: int = 15
    query_cosine_colorbar_tick_font_size: int = 13

    background_outlier_label_id: str = 'background_clinical_cluster'
    background_outlier_role: str = 'background_outlier'
    background_outlier_color: str = '#d62728'
    background_outlier_map_marker_size: int = 64
    background_outlier_similarity_map_marker_size: int = 82
    background_outlier_rank_marker_size: int = 60
    background_outlier_selection_unselected_marker_size: int = 50
    background_outlier_selection_selected_marker_size: int = 68

    full_strategy_panel_size_in: float = 2.50
    full_strategy_panel_gap_x_in: float = 1.00
    full_strategy_panel_gap_y_in: float = 1.25
    full_strategy_left_margin_in: float = 0.50
    full_strategy_right_margin_in: float = 0.40
    full_strategy_bottom_margin_in: float = 0.42
    full_strategy_top_margin_in: float = 0.90
    full_strategy_colorbar_width_in: float = 0.28
    full_strategy_colorbar_gap_in: float = 0.16
    full_strategy_legend_width_in: float = 4.75
    full_strategy_legend_gap_in: float = 0.90
    full_strategy_overlay_legend_fontsize: int = 7

    strategy_overlay_legend_fontsize: int = 8

    query_overview_figure_size: tuple[int, int] = (14, 14)
    query_overview_legend_wrap_width: int = 60
    query_overview_legend_font_size: int = 10
    query_overview_legend_bbox_to_anchor_x: float = 1.00
    query_title_condition_color: str = '#9C7A00'
    query_title_subgroup_color: str = '#7B2CBF'
    query_title_axis_color: str = '#2D6A4F'

    legend_match_color: str = 'black'
    legend_non_match_color: str = '#9A9A9A'
    legend_child_indent: str = '     '
    legend_header_label: str = 'Primary Condition / Subgroup / Clinical Axis\n'
    background_outlier_legend_prefix: str = 'background outlier: '
    distractor_mix_legend_prefix: str = '__distractor_mix__:'
    background_outlier_mix_legend_prefix: str = '__background_outlier_mix__:'
    background_outlier_legend_color: str = '#4A4A4A'

    gold_facet_role_labels: dict[ClusterRole, str] = field(
        default_factory=lambda: dict(
            {
                'dominant_primary_gold': 'Dominant primary',
                'primary_gold': 'Other primary',
                'secondary_gold': 'Secondary',
                'niche_gold': 'Niche',
            }
        )
    )
    gold_facet_role_order: dict[ClusterRole, int] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'gold_facet_role_order',
            dict({role: index for index, role in enumerate(self.gold_facet_role_labels)}),
        )


SETTINGS: Final = GeomPlotsSettings()
