from __future__ import annotations

from typing import Literal, TypedDict

type Strategy = Literal['top_k', 'mmr', 'fac_loc']


class RetrievalIndexMaps(TypedDict):
    chunk_id_to_idx: dict[str, int]
    query_id_to_idx: dict[str, int]
    chunk_by_id: dict[str, dict[str, object]]
    membership_by_query_chunk: dict[tuple[str, str], dict[str, object]]
    query_by_id: dict[str, dict[str, object]]
    chunks_by_source_query: dict[str, list[int]]
    chunks_by_condition: dict[str, list[int]]
