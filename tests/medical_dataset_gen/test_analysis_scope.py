from collections.abc import Mapping

from experiments.medical_dataset_gen.reports.analysis_scope import (
    interaction_rows,
    primary_rows,
)
from experiments.medical_dataset_gen.reports.helpers import interaction_distribution_label


def test_interaction_scope_partitions_report_rows() -> None:
    rows: list[Mapping[str, object]] = [
        {'ExperimentFamily': 'dominance'},
        {'ExperimentFamily': 'interaction'},
        {'ExperimentFamily': 'sparse_niche'},
    ]

    assert primary_rows(rows) == [rows[0], rows[2]]
    assert interaction_rows(rows) == [
        {'ExperimentFamily': 'interaction', 'ExperimentFamilyLabel': 'Interaction experiments'}
    ]


def test_interaction_distribution_labels_are_self_describing() -> None:
    assert interaction_distribution_label('interaction_dom_high_far_32x1') == (
        'DOM-high \N{MULTIPLICATION SIGN} BG-32\N{MULTIPLICATION SIGN}1'
    )
    assert interaction_distribution_label('interaction_sparse_severe_h24') == (
        'SN-severe \N{MULTIPLICATION SIGN} NM-H24'
    )
