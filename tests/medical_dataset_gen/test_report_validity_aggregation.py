from __future__ import annotations

from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from experiments.medical_dataset_gen.evaluation.statistics import stats_aggregated_results_df
from experiments.medical_dataset_gen.reports.validity import (
    _GEOMETRY_POPULATIONS,
    _POPULATION_METRIC_COLUMNS,
    _POPULATION_PASS_FILTER_VALUE,
    _load_population_partition_stats,
    _stats_for_geometry_population,
)


def test_lazy_population_aggregation_matches_raw_results(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for split in ('validation', 'test'):
        for passes_filter in (False, True):
            for query_index in range(2):
                for strategy, lam in (('top_k', None), ('mmr', 0.5), ('fac_loc', 0.5)):
                    base = 0.1 + 0.2 * int(passes_filter) + 0.01 * query_index
                    row: dict[str, object] = {
                        'query_id': f'{split}-{passes_filter}-{query_index}',
                        'split': split,
                        'strategy': strategy,
                        'k': 6,
                        'lam': lam,
                        'passes_geometry_filter': passes_filter,
                    }
                    row.update({source: base for source, _target in _POPULATION_METRIC_COLUMNS})
                    rows.append(row)
    raw = pl.DataFrame(rows)
    results_path = tmp_path / 'evaluation_results.parquet'
    raw.write_parquet(results_path)

    partitioned = _load_population_partition_stats(
        results_path,
        geometry_path=tmp_path / 'unused_geometry.parquet',
    )
    for population in _GEOMETRY_POPULATIONS:
        pass_value = _POPULATION_PASS_FILTER_VALUE[population]
        source = (
            raw
            if pass_value is None
            else raw.filter(pl.col('passes_geometry_filter') == pass_value)
        )
        actual_all = _stats_for_geometry_population(partitioned, population)
        for split in ('validation', 'test'):
            expected = stats_aggregated_results_df(source.filter(pl.col('split') == split))
            actual = actual_all.filter(pl.col('split') == split).drop('split')
            common_columns = [column for column in expected.columns if column in actual.columns]
            assert_frame_equal(
                actual.select(common_columns).sort('k', 'strategy', 'lam'),
                expected.select(common_columns).sort('k', 'strategy', 'lam'),
                check_dtypes=False,
            )
