from experiments.medical_dataset_gen.reports.report_config import LOW_BUDGET_K
from experiments.medical_dataset_gen.reports.summaries import budget_category_rows_from_comparisons


def _comparison_row(*, experiment: str, k: int) -> dict[str, object]:
    return {
        'Experiment': experiment,
        'k': k,
        'TopK_FCP': 0.5,
        'MMR_FCP': 0.6,
        'FacLoc_FCP': 0.7,
    }


def test_low_budget_uses_the_global_k_value() -> None:
    rows = [
        _comparison_row(experiment='complete_grid', k=4),
        _comparison_row(experiment='complete_grid', k=LOW_BUDGET_K),
        _comparison_row(experiment='complete_grid', k=10),
        _comparison_row(experiment='isolated_budget_run', k=LOW_BUDGET_K + 6),
    ]

    budget_rows = budget_category_rows_from_comparisons(rows)
    low_rows = [row for row in budget_rows if row['BudgetCategory'] == 'low_budget']

    assert len(low_rows) == 1
    assert low_rows[0]['Experiment'] == 'complete_grid'
    assert low_rows[0]['k'] == LOW_BUDGET_K
    assert low_rows[0]['LowBudgetRule'] == f'fixed global k={LOW_BUDGET_K}'
