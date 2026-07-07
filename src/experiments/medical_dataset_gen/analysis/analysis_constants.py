from experiments.medical_dataset_gen.analysis.__main__ import StrategyName
from experiments.medical_dataset_gen.schemas.metrics_schemas import METRIC_NAME_TO_FIELD

STRATEGIES: tuple[StrategyName, ...] = ('top_k', 'mmr', 'fac_loc')
DIVERSIFYING_STRATEGIES: tuple[StrategyName, ...] = ('mmr', 'fac_loc')
HELDOUT_SELECTION_COLUMNS = frozenset({
    'lambda_selection_split',
    'report_split',
    'lambda_selection_metric',
    'lambda_selection_metric_value',
})
REPORT_FILES = (
    'report.md',
    'interesting_findings.md',
    'warnings.txt',
    'manifest.json',
    'experiment_manifest.csv',
    'dataset_distribution.csv',
    'geometry_filter_summary.csv',
    'strategy_by_k.csv',
    'comparison_by_k.csv',
    'headline_strategy_summary.csv',
    'lambda_stability.csv',
    'near_optimal_lambda_width.csv',
    'embedding_model_summary.csv',
    'embedding_query_scope_pairs.csv',
)
FCP_TIE_EPSILON = 0.01
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
    'TopK_AllFacetCleanRate': 'Top Clean',
    'MMR_AllFacetCleanRate': 'MMR Clean',
    'FacLoc_AllFacetCleanRate': 'FacLoc Clean',
    'PassOnlyShortExperiment': 'Pass-only',
    'AllQueriesShortExperiment': 'All-query',
    'AllMinusPassOnly_MMR_FCP': 'All-Pass MMR',
    'AllMinusPassOnly_FacLoc_FCP': 'All-Pass F',
    'AllMinusPassOnly_Delta_FacLoc_MMR_FCP': 'All-Pass Delta',
    'selected_lambda_norm_mean': 'lambda* norm mean',
    'selected_lambda_norm_std': 'lambda* norm std',
    'near_optimal_fraction_mean': 'near-opt frac',
    'near_optimal_span_norm_mean': 'near-opt width',
}
TABLE_COL_WIDTHS = {
    'ShortExperiment': 10,
    'ShortDistribution': 8,
    'EmbeddingModel': 24,
    'DistributionCategory': 28,
    'TopFailureModes': 34,
    'PassOnlyShortExperiment': 10,
    'AllQueriesShortExperiment': 10,
    'strategy': 8,
}
DEFAULT_TABLE_COL_WIDTH = 14
INTEGER_TABLE_COLUMNS = frozenset({
    'k',
    'EmbeddingDimension',
    'GeometryQueries',
    'GeometryPassQueries',
    'Runs',
    'PassOnlyRuns',
    'AllQueryRuns',
    'n_selected',
    'distinct_lambda_count',
    'PassOnly_k',
    'AllQueries_k',
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
