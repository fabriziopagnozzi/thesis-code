from experiments.medical_dataset_gen.reports.latex_macros import (
    _distribution_budget_result_macros,
)


def _comparison_row(
    *,
    model: str,
    fcp_delta: float,
    mmr_near_miss_rate: float,
    facloc_near_miss_rate: float,
) -> dict[str, object]:
    return {
        'EmbeddingModel': model,
        'Distribution': 'balanced_reference',
        'k': 6,
        'Delta_FacLoc_MMR_FCP': fcp_delta,
        'MMR_NearMissDistractorRate': mmr_near_miss_rate,
        'FacLoc_NearMissDistractorRate': facloc_near_miss_rate,
    }


def test_distribution_macros_weight_embedding_models_equally() -> None:
    qwen = 'Qwen/Qwen3-Embedding-0.6B'
    medembed = 'abhinand/MedEmbed-large-v0.1'
    rows = [
        _comparison_row(
            model=qwen,
            fcp_delta=0.0,
            mmr_near_miss_rate=0.2,
            facloc_near_miss_rate=0.1,
        ),
        _comparison_row(
            model=qwen,
            fcp_delta=0.2,
            mmr_near_miss_rate=0.4,
            facloc_near_miss_rate=0.2,
        ),
        _comparison_row(
            model=medembed,
            fcp_delta=0.5,
            mmr_near_miss_rate=0.9,
            facloc_near_miss_rate=0.3,
        ),
    ]

    macros = _distribution_budget_result_macros(
        rows,
        embedding_models=(qwen, medembed),
    )

    prefix = 'ResultDistributionBalancedReferenceK6'
    assert macros['ResultDistributionPooledModels'] == '2'
    assert macros[f'{prefix}ModelCount'] == '2'
    assert macros[f'{prefix}FacLocMmrFcpMeanDelta'] == '+0.300'
    assert macros[f'{prefix}FacLocMmrFcpMinModelDelta'] == '+0.100'
    assert macros[f'{prefix}FacLocMmrFcpMaxModelDelta'] == '+0.500'
    assert macros[f'{prefix}FacLocMmrFcpPositiveModelCount'] == '2'
    assert macros[f'{prefix}MmrNearMissDistractorRate'] == r'60\%'
    assert macros[f'{prefix}FacLocNearMissDistractorRate'] == r'22.5\%'
    assert macros[f'{prefix}NearMissReductionPp'] == '37.5'


def test_distribution_macros_omit_incomplete_model_crossings() -> None:
    qwen = 'Qwen/Qwen3-Embedding-0.6B'
    medembed = 'abhinand/MedEmbed-large-v0.1'
    rows = [
        _comparison_row(
            model=qwen,
            fcp_delta=0.2,
            mmr_near_miss_rate=0.4,
            facloc_near_miss_rate=0.2,
        )
    ]

    macros = _distribution_budget_result_macros(
        rows,
        embedding_models=(qwen, medembed),
    )

    assert 'ResultDistributionBalancedReferenceK6FacLocMmrFcpMeanDelta' not in macros
    assert macros['ResultQwenSmallDistributionBalancedReferenceK6FacLocMmrFcpMeanDelta'] == (
        '+0.200'
    )
