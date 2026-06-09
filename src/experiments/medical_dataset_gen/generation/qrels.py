import polars as pl

from experiments.medical_dataset_gen.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
    read_parquet,
    write_parquet,
)


def run_make_qrels(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    _ = cfg
    chunks = read_parquet(paths, 'chunks')
    qrels = (
        chunks
        .select(
            'query_id',
            'chunk_id',
            'fact_id',
            'facet_id',
            'target_facet_id',
            'cluster_id',
            'cluster_role',
            'is_gold',
            'distractor_type',
        )
        .with_columns(
            pl.when(pl.col('is_gold')).then(1).otherwise(0).alias('relevance_grade'),
            pl.when(pl.col('is_gold')).then(pl.lit('positive')).otherwise(pl.lit('hard_negative')).alias('support_type'),
        )
    )
    write_parquet(paths, 'qrels', qrels)
    return qrels


if __name__ == '__main__':
    from experiments.medical_dataset_gen.global_configs import (
        dump_effective_config,
        load_config_from_cli,
        paths_for,
        setup_logging,
    )

    cfg = load_config_from_cli()
    paths = paths_for(cfg)
    setup_logging(paths)
    dump_effective_config(cfg, paths)
    run_make_qrels(cfg, paths)

