import re

import polars as pl

from experiments.mimic.embeddings.schemas_embeddings import EmbedJoinedRow
from experiments.mimic.global_configs import read_parquet
from experiments.mimic.utils.charlson import CHARLSON_LABELS_TO_STR


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


def load_filtered_queries(embedding_model: str) -> pl.DataFrame:
    bool_filter_for_model = f'filter_{get_vec_col_name(embedding_model)}'
    queries_df = read_parquet('queries')
    if bool_filter_for_model not in queries_df.columns:
        raise RuntimeError(
            f'You need to run the query filtering step before (expected column: {bool_filter_for_model!r}).'
        )

    return queries_df.filter(pl.col(bool_filter_for_model))
