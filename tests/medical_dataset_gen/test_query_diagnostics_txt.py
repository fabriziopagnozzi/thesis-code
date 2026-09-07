from pathlib import Path
from typing import cast

import polars as pl

from experiments.medical_dataset_gen.scripts.query_diagnostics_txt import (
    _collect_query_table,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    SharedGenerationArtifactPaths,
)


def test_collect_query_table_honors_shared_suite_path(tmp_path: Path) -> None:
    shared_queries = tmp_path / 'shared' / 'queries.parquet'
    shared_queries.parent.mkdir()
    pl.DataFrame(
        {
            'query_id': ['q1', 'q2'],
            'query_text': ['first', 'second'],
        }
    ).write_parquet(shared_queries)
    paths = MedicalDatasetGenPaths(
        'suite-cell',
        artifact_root=tmp_path / 'attempt',
        shared_generation_artifact_paths=cast(
            SharedGenerationArtifactPaths,
            {'queries': shared_queries},
        ),
    )

    rows = _collect_query_table(paths, 'queries', ['q2'], ['query_id', 'query_text'])

    assert rows.to_dicts() == [{'query_id': 'q2', 'query_text': 'second'}]
