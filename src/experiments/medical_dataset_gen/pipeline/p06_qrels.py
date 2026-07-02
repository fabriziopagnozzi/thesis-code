"""Derive relevance labels from the synthetic chunk-to-facet mapping.

This module exists to convert the generated chunk metadata into qrels that the
retrieval evaluation stage can consume directly. It uses a simple label
derivation rule over the hidden gold/distractor structure so relevance stays
fully aligned with the benchmark design.
"""

from __future__ import annotations

import polars as pl

from experiments.medical_dataset_gen.schemas.global_config_schemas import (
    ExperimentCfg,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    load_config_from_cli,
    paths_for,
)


def run_make_qrels(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    _ = cfg
    qrels_path = paths.table_path('qrels')
    (
        pl
        .scan_parquet(paths.table_path('chunk_memberships'))
        .select(
            'query_id',
            'evidence_profile_id',
            'pool_id',
            'primary_axis',
            'secondary_axis',
            'calibrated_primary_facet_id',
            'chunk_id',
            'fact_id',
            'facet_id',
            'target_facet_id',
            'cluster_id',
            'cluster_role',
            'axis',
            'facet_priority',
            'is_gold',
            'distractor_type',
        )
        .with_columns(
            pl.when(pl.col('is_gold')).then(1).otherwise(0).alias('relevance_grade'),
            pl
            .when(pl.col('is_gold'))
            .then(pl.lit('positive'))
            .when(pl.col('cluster_role') == 'background_outlier')
            .then(pl.lit('background_outlier'))
            .otherwise(pl.lit('hard_negative'))
            .alias('support_type'),
        )
        .sink_parquet(qrels_path)
    )
    qrels = pl.read_parquet(qrels_path)
    print(f'[write] qrels: {len(qrels):,} rows -> {qrels_path}')
    return qrels


if __name__ == '__main__':
    from experiments.medical_dataset_gen.utils.global_utils import (
        setup_logging,
    )

    cfg = load_config_from_cli()
    paths = paths_for(cfg)
    setup_logging(paths)
    run_make_qrels(cfg, paths)
