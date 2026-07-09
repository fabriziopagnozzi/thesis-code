from typing import Literal, get_args

from experiments.medical_dataset_gen.schemas.metrics_schemas import METRIC_NAME_TO_FIELD

FCP_TIE_EPSILON = 0.05

type StrategyName = Literal['top_k', 'mmr', 'fac_loc']
type DeltaMetricLabel = Literal[
    'FCP',
    'FacetCoverage',
    'AllFacetCleanRate',
    'Precision',
    'Recall',
    'alpha_nDCG',
]
type ExperimentFamilyId = Literal[
    'balanced_clean',
    'dominance',
    'sparse_niche',
    'near_miss_heavy',
    'background_variant',
    'budget_sweep',
    'embedding_comparison',
    'unknown',
]

STRATEGIES: tuple[StrategyName, ...] = ('top_k', 'mmr', 'fac_loc')
DIVERSIFYING_STRATEGIES: tuple[StrategyName, ...] = ('mmr', 'fac_loc')
DELTA_METRIC_LABELS = tuple[DeltaMetricLabel, ...](get_args(DeltaMetricLabel.__value__))
EXPERIMENT_FAMILIES: tuple[ExperimentFamilyId, ...] = tuple[ExperimentFamilyId](
    get_args(ExperimentFamilyId.__value__)
)
EXPERIMENT_FAMILY_LABELS: dict[ExperimentFamilyId, str] = {
    'balanced_clean': 'Balanced clean',
    'dominance': 'Dominance',
    'sparse_niche': 'Sparse niche',
    'near_miss_heavy': 'Near-miss heavy',
    'background_variant': 'Background variant',
    'budget_sweep': 'Budget sweep',
    'embedding_comparison': 'Embedding comparison',
    'unknown': 'Unknown',
}
EXPERIMENT_FAMILY_COLORS: dict[ExperimentFamilyId, str] = {
    'balanced_clean': '#287C8E',
    'dominance': '#C44E52',
    'sparse_niche': '#5F8F3F',
    'near_miss_heavy': '#C47A3A',
    'background_variant': '#6F63A6',
    'budget_sweep': '#8C6D31',
    'embedding_comparison': '#4C78A8',
    'unknown': '#808080',
}
HELDOUT_SELECTION_COLUMNS = frozenset({
    'lambda_selection_split',
    'report_split',
    'lambda_selection_metric',
    'lambda_selection_metric_value',
})
REPORT_FILES = (
    'report.md',
    'report_interesting_findings.md',
    'warnings.txt',
    'manifest.json',
    'experiment_manifest.csv',
    'dataset_distribution.csv',
    'geometry_filter_summary.csv',
    'strategy_by_k.csv',
    'comparison_by_k.csv',
    'experiment_family_summary.csv',
    'experiment_family_budget_summary.csv',
    'metric_aggregate_summary.csv',
    'budget_strategy_summary.csv',
    'headline_strategy_summary.csv',
    'lambda_stability.csv',
    'lambda_grid_fcp_delta.csv',
    'lambda_safety_summary.csv',
    'near_optimal_lambda_width.csv',
    'embedding_model_summary.csv',
)
ANALYSIS_EXCLUDED_METRICS = frozenset({
    'MAP@k',
    'AnswerROUGE1Recall@k',
    'AnswerROUGE1Precision@k',
    'AnswerROUGE1F1@k',
    'AnswerROUGE2Recall@k',
})
EVALUATION_METRICS = (
    'n_queries',
    *(
        metric_name
        for metric_name in METRIC_NAME_TO_FIELD
        if metric_name not in ANALYSIS_EXCLUDED_METRICS
    ),
)
METRIC_LABEL_OVERRIDES = {
    'FacetCoveragePurity@k': 'FCP',
    'alpha-nDCG@k': 'alpha_nDCG',
}
METRIC_LABELS = {
    metric_name: METRIC_LABEL_OVERRIDES.get(metric_name, metric_name.removesuffix('@k'))
    for metric_name in EVALUATION_METRICS
    if metric_name not in {'n_queries', 'fac', 'avg_cos', 'jac'}
}
TABLE_HEADERS = {
    'ShortExperiment': 'ID',
    'ShortDistribution': 'Dist',
    'EmbeddingModel': 'Embedding',
    'OnlyPassGeometry': 'Pass-only',
    'QueryScope': 'Scope',
    'ExperimentFamily': 'Family ID',
    'ExperimentFamilyLabel': 'Family',
    'BudgetCategoryLabel': 'Budget',
    'DistributionCategory': 'Distribution',
    'GoldPercentage': 'Gold%',
    'NearMissDistractorPercentage': 'NearMiss%',
    'BackgroundOutlierPercentage': 'BgOut%',
    'DominantPrimaryGoldCountMean': 'DomGold',
    'OtherPrimaryGoldCountMean': 'OtherGold',
    'SecondaryGoldCountMean': 'SecGold',
    'NicheGoldCountMean': 'NicheGold',
    'HardDistractorCountMean': 'HardDistr',
    'GeometryPassRate': 'PassRate',
    'GeometryQueries': 'GeomQ',
    'GeometryPassQueries': 'PassQ',
    'NTopkRetrievedFacetsMean': 'TopKFacets',
    'PrimaryAxisTopkFractionMean': 'PrimaryAxis',
    'DominantPrimaryTopkFractionMean': 'DomPrimary',
    'TopFailureModes': 'Top failures',
    'Delta_FacLoc_MMR_FCP': 'F-MM FCP',
    'Delta_FacLoc_TopK_FCP': 'F-Top FCP',
    'Delta_FacLoc_MMR_AllFacetCleanRate': 'F-MM Clean',
    'FacLocVsMMR_FCPOutcome': 'F vs M',
    'FacLocBetterRows': 'F > M',
    'FacLocTiedRows': 'Tie',
    'FacLocWorseRows': 'F < M',
    'Metric': 'Metric',
    'BudgetView': 'Budget view',
    'FacLocTopKBetterRows': 'F > TopK',
    'MMRTopKBetterRows': 'M > TopK',
    'MeanDeltaFacLocMMR': 'Mean F-M',
    'MedianDeltaFacLocMMR': 'Median F-M',
    'MeanDeltaFacLocTopK': 'Mean F-Top',
    'MeanDeltaMMRTopK': 'Mean M-Top',
    'Delta_FacLoc_MMR_FCP_mean': 'Mean F-M FCP',
    'Delta_FacLoc_MMR_FCP_median': 'Median F-M FCP',
    'Delta_FacLoc_MMR_FCP_min': 'Min F-M FCP',
    'Delta_FacLoc_MMR_FCP_max': 'Max F-M FCP',
    'Delta_FacLoc_TopK_FCP_mean': 'Mean F-Top FCP',
    'Delta_MMR_TopK_FCP_mean': 'Mean M-Top FCP',
    'Delta_FacLoc_MMR_AllFacetCleanRate_mean': 'Mean F-M Clean',
    'TopK_AllFacetCleanRate': 'Top Clean',
    'MMR_AllFacetCleanRate': 'MMR Clean',
    'FacLoc_AllFacetCleanRate': 'FacLoc Clean',
    'PassFilterRuns': 'Pass-filter runs',
    'selected_lambda_norm_mean': 'lambda* norm mean',
    'selected_lambda_norm_std': 'lambda* norm std',
    'near_optimal_fraction_mean': 'near-opt frac',
    'near_optimal_span_norm_mean': 'near-opt width',
    'SafeLambdaFraction': 'safe lambda frac',
    'WorstDeltaStrategyTopK_FCP': 'worst dTopK FCP',
    'MedianDeltaStrategyTopK_FCP': 'median dTopK FCP',
    'BestDeltaStrategyTopK_FCP': 'best dTopK FCP',
    'WorstLambda': 'worst lambda',
    'BestLambda': 'best lambda',
}
TABLE_COL_WIDTHS = {
    'ShortExperiment': 10,
    'ShortDistribution': 8,
    'EmbeddingModel': 24,
    'DistributionCategory': 28,
    'ExperimentFamilyLabel': 20,
    'TopFailureModes': 34,
    'strategy': 8,
}
DEFAULT_TABLE_COL_WIDTH = 14
INTEGER_TABLE_COLUMNS = frozenset({
    'k',
    'EmbeddingDimension',
    'GeometryQueries',
    'GeometryPassQueries',
    'Runs',
    'Rows',
    'FacLocBetterRows',
    'FacLocTiedRows',
    'FacLocWorseRows',
    'FacLocTopKBetterRows',
    'MMRTopKBetterRows',
    'PassFilterRuns',
    'n_selected',
    'distinct_lambda_count',
})
ROLE_COUNT_COLUMNS = {
    'dominant_primary_gold': 'DominantPrimaryGoldCount',
    'primary_gold': 'OtherPrimaryGoldCount',
    'secondary_gold': 'SecondaryGoldCount',
    'niche_gold': 'NicheGoldCount',
    'hard_distractor': 'HardDistractorCount',
}
TABLEFMT_OPTS = [
    'plain',
    'simple',
    'github',
    'grid',
    'simple_grid',
    'rounded_grid',
    'heavy_grid',
    'mixed_grid',
    'double_grid',
    'fancy_grid',
    'outline',
    'simple_outline',
    'rounded_outline',
    'heavy_outline',
    'mixed_outline',
    'double_outline',
    'fancy_outline',
    'pipe',
    'orgtbl',
    'asciidoc',
    'jira',
    'presto',
    'pretty',
    'psql',
    'rst',
    'mediawiki',
    'moinmoin',
    'youtrack',
    'html',
    'unsafehtml',
    'latex',
    'latex_raw',
    'latex_booktabs',
    'latex_longtable',
    'textile',
    'tsv',
]
