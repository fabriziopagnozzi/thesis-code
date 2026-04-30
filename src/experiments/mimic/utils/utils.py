import re

from experiments.mimic.embeddings.schemas_embeddings import EmbedJoinedRow
from experiments.mimic.queries.schemas_queries import QueryAspect
from experiments.mimic.utils.charlson import CHARLSON_LABELS_TO_STR


def modifier_to_snake_label(text: str) -> str:
    s = re.sub(r'\([^)]*\)', '', text)
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s.lower())
    s = re.sub(r'\b(?:the|a|an)\b', '', s)
    tokens = [t for t in s.split() if t]
    return '_'.join(tokens[:6])


def aspects_from_modifiers(modifiers_json: list[dict]) -> list[QueryAspect]:
    return [
        QueryAspect(facet_label=modifier_to_snake_label(m['text']), description=m['text'])
        for m in modifiers_json
    ]


def get_vec_col_name(model_name: str) -> str:
    safe = re.sub(r'[/\-.]', '_', model_name)
    return f'vector_{safe}'


def get_age_group(age: float | None) -> str:
    if age is None:
        return 'unknown age'
    if age < 30:
        return 'young adult'
    if age < 50:
        return 'middle-aged'
    if age < 65:
        return 'older adult'
    if age < 80:
        return 'elderly'
    return 'very elderly'


def get_charlson_conditions(meta_row: EmbedJoinedRow) -> list[str]:
    return [label for col, label in CHARLSON_LABELS_TO_STR.items() if meta_row.get(col) == 1]
