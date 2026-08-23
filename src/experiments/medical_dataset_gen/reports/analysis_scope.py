"""Classify report rows into primary and interaction analysis scopes."""

from collections.abc import Mapping, Sequence

INTERACTION_FAMILY_ID = 'interaction'
INTERACTION_FAMILY_LABEL = 'Interaction experiments'


def is_interaction_row(row: Mapping[str, object]) -> bool:
    """Return whether a row belongs to a deliberately crossed interaction cell."""
    if row.get('ExperimentFamily') == INTERACTION_FAMILY_ID:
        return True
    if row.get('AnalysisTier') == 'interaction':
        return True
    tags = row.get('SuiteTags')
    return isinstance(tags, str) and 'interaction' in tags.split('|')


def interaction_rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return [
        {
            **row,
            'ExperimentFamily': INTERACTION_FAMILY_ID,
            'ExperimentFamilyLabel': INTERACTION_FAMILY_LABEL,
        }
        for row in rows
        if is_interaction_row(row)
    ]


def primary_rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    """Exclude crossed interaction cells from main-effect reporting."""
    return [row for row in rows if not is_interaction_row(row)]
