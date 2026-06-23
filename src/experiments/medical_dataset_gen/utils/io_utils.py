import json
from pathlib import Path
from typing import Any

import polars as pl

from experiments.medical_dataset_gen.global_config import (
    MedicalDatasetGenPaths,
    SyntheticMedicalDatasetTableName,
)


def json_loads(value: str | None, default: Any = None) -> Any:
    if value is None or value == '':
        return default
    return json.loads(value)


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def write_json(paths: MedicalDatasetGenPaths, name: str, payload: Any) -> Path:
    path = paths.experiment_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'[write] {path}')
    return path


def write_parquet(
    paths: MedicalDatasetGenPaths, table: SyntheticMedicalDatasetTableName, df: pl.DataFrame
) -> Path:
    path = paths.table_path(table)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    print(f'[write] {table}: {len(df):,} rows -> {path}')
    return path


def read_parquet(
    paths: MedicalDatasetGenPaths, table: SyntheticMedicalDatasetTableName
) -> pl.DataFrame:
    return pl.read_parquet(paths.table_path(table))


def read_parquet_if_exists_else_empty_df(
    paths: MedicalDatasetGenPaths, table: SyntheticMedicalDatasetTableName
) -> pl.DataFrame:
    if paths.table_path(table).exists():
        return read_parquet(paths, table)
    return pl.DataFrame()
