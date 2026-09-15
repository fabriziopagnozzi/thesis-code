"""Largest-to-smallest projection and safe artifact reuse for scale lineages."""

from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl

from experiments.medical_dataset_gen.dataset_generation.caches import chunk_embedding_signature
from experiments.medical_dataset_gen.embedding.artifacts import embedding_artifacts_ready
from experiments.medical_dataset_gen.pipeline.stages import PipelineStage
from experiments.medical_dataset_gen.suites.contracts import SuiteManifestCell
from experiments.medical_dataset_gen.suites.runtime import SuiteRuntime
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths
from experiments.medical_dataset_gen.utils.io_utils import read_parquet, write_parquet


def required_nested_scale_source(
    runtime: SuiteRuntime, cell: SuiteManifestCell
) -> SuiteManifestCell | None:
    """Return the terminal, largest cell that must construct this lineage."""
    runtime.cell(cell.cell_id)
    current = cell
    while True:
        children = [
            candidate
            for candidate in runtime.manifest.cells
            if candidate.nested_from == current.cell_id
            and candidate.run_profile_id == cell.run_profile_id
        ]
        if not children:
            return current if current.cell_id != cell.cell_id else None
        if len(children) != 1:
            raise ValueError(
                f'{cell.cell_id}: nested scale lineage must have one successor, found '
                f'{[candidate.cell_id for candidate in children]}'
            )
        current = children[0]


def project_nested_scale_parents(
    runtime: SuiteRuntime,
    source_cell: SuiteManifestCell,
    source_cfg: ExperimentCfg,
) -> list[str]:
    """Project every smaller support from a completed terminal support."""
    if source_cell.nested_from is None:
        return []
    if any(
        candidate.nested_from == source_cell.cell_id
        and candidate.run_profile_id == source_cell.run_profile_id
        for candidate in runtime.manifest.cells
    ):
        return []
    if source_cell.origin != 'native':
        return []
    source_paths = runtime.paths(source_cell, source_cfg)
    required = (
        'query_plans',
        'clinical_facts',
        'chunk_documents',
        'chunk_memberships',
        'qrels',
        'queries',
        'gold_answers',
    )
    if any(not source_paths.table_path(name).is_file() for name in required):
        return []
    source_facts = read_parquet(source_paths, 'clinical_facts')
    source_memberships = read_parquet(source_paths, 'chunk_memberships')
    source_qrels = read_parquet(source_paths, 'qrels')
    projected: list[str] = []
    parent_id = source_cell.nested_from
    while parent_id is not None:
        parent = runtime.cells.get(parent_id)
        if parent is None:
            raise ValueError(f'{source_cell.cell_id}: missing nested parent {parent_id!r}')
        if parent.origin != 'native':
            raise ValueError(f'{source_cell.cell_id}: cannot project a derived parent')
        parent_cfg = runtime.load_config(parent)
        _project_one_nested_scale_cell(
            source_paths=source_paths,
            source_facts=source_facts,
            source_memberships=source_memberships,
            source_qrels=source_qrels,
            target_paths=runtime.paths(parent, parent_cfg),
            target_cfg=parent_cfg,
            source_cell=source_cell,
            target_cell=parent,
        )
        projected.append(parent.cell_id)
        parent_id = parent.nested_from
    return projected


def reuse_nested_scale_chunk_embeddings(
    *,
    runtime: SuiteRuntime,
    cell: SuiteManifestCell,
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    requested: set[PipelineStage],
) -> bool:
    if 'embed' not in requested:
        return False
    source = required_nested_scale_source(runtime, cell)
    if source is None or source.cell_id == cell.cell_id:
        return False
    source_cfg = runtime.load_config(source)
    if chunk_embedding_signature(source_cfg) != chunk_embedding_signature(cfg):
        return False
    source_paths = runtime.paths(source, source_cfg)
    if not embedding_artifacts_ready(source_paths):
        return False
    if not os.path.samefile(
        source_paths.table_path('chunk_documents'), paths.table_path('chunk_documents')
    ):
        raise RuntimeError(
            f'{cell.cell_id}: nested scale does not share its source chunk documents; '
            'refusing to reuse source embedding vectors'
        )
    for artifact in ('chunk_vectors', 'chunk_ids'):
        source_path = source_paths.embeddings_paths(artifact)
        target_path = paths.embeddings_paths(artifact)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            if not os.path.samefile(source_path, target_path):
                raise RuntimeError(
                    f'{cell.cell_id}: existing {artifact} is not the verified nested source'
                )
            continue
        if source_path.stat().st_dev != target_path.parent.stat().st_dev:
            raise RuntimeError(
                f'{cell.cell_id}: nested embedding reuse requires one filesystem: '
                f'{source_path} -> {target_path}'
            )
        os.link(source_path, target_path)
    print(f'[pipeline] reusing verified chunk embeddings from {source.cell_id}')
    return True


def _project_one_nested_scale_cell(
    *,
    source_paths: MedicalDatasetGenPaths,
    source_facts: pl.DataFrame,
    source_memberships: pl.DataFrame,
    source_qrels: pl.DataFrame,
    target_paths: MedicalDatasetGenPaths,
    target_cfg: ExperimentCfg,
    source_cell: SuiteManifestCell,
    target_cell: SuiteManifestCell,
) -> None:
    base_files = (
        target_paths.table_path('query_plans'),
        target_paths.table_path('clinical_facts'),
    )
    chunk_files = (
        target_paths.table_path('chunk_documents'),
        target_paths.table_path('chunk_memberships'),
        target_paths.table_path('qrels'),
    )
    query_files = (
        target_paths.table_path('queries'),
        target_paths.table_path('gold_answers'),
    )
    base_existing = [path.exists() for path in base_files]
    chunk_existing = [path.exists() for path in chunk_files]
    query_existing = [path.exists() for path in query_files]
    if all(chunk_existing) and all(query_existing):
        if not all(base_existing):
            raise RuntimeError(
                f'{target_cell.cell_id}: completed nested surface is missing shared base data'
            )
        return
    _refuse_partial(target_cell, 'chunk', chunk_existing)
    _refuse_partial(target_cell, 'query', query_existing)
    _refuse_partial(target_cell, 'shared base', base_existing)
    if all(chunk_existing) and not all(base_existing):
        raise RuntimeError(
            f'{target_cell.cell_id}: completed chunk surface is missing shared base data'
        )
    if not all(base_existing):
        _hard_link(source_paths.table_path('query_plans'), target_paths.table_path('query_plans'))
    if not all(chunk_existing):
        projected_facts = _select_nested_scale_facts(source_facts, target_cfg)
        selected_chunk_keys = projected_facts.select(
            [
                'query_id',
                pl.concat_str([pl.lit('chunk_'), pl.col('chunk_reuse_key')]).alias('chunk_id'),
            ]
        )
        projected_memberships = source_memberships.join(
            selected_chunk_keys, on=['query_id', 'chunk_id'], how='inner'
        )
        projected_qrels = source_qrels.join(
            selected_chunk_keys, on=['query_id', 'chunk_id'], how='inner'
        )
        if projected_memberships.height != projected_qrels.height:
            raise RuntimeError(
                f'{target_cell.cell_id}: source memberships/qrels disagree during projection'
            )
        _hard_link(
            source_paths.table_path('chunk_documents'),
            target_paths.table_path('chunk_documents'),
        )
        write_parquet(target_paths, 'chunk_memberships', projected_memberships)
        write_parquet(target_paths, 'qrels', projected_qrels)
        target_paths.table_path('qrels').with_suffix('.projection.json').write_text(
            json.dumps(
                {
                    'layout_version': 5,
                    'projection': 'nested_scale_subset',
                    'source_cell_id': source_cell.cell_id,
                    'target_cell_id': target_cell.cell_id,
                    'source_document_path': str(source_paths.table_path('chunk_documents')),
                    'candidate_rows': projected_qrels.height,
                },
                indent=2,
                sort_keys=True,
            )
            + '\n'
        )
        if not all(base_existing):
            write_parquet(target_paths, 'clinical_facts', projected_facts)
    if not all(query_existing):
        _hard_link(source_paths.table_path('queries'), target_paths.table_path('queries'))
        _hard_link(source_paths.table_path('gold_answers'), target_paths.table_path('gold_answers'))


def _refuse_partial(cell: SuiteManifestCell, label: str, existing: list[bool]) -> None:
    if any(existing) and not all(existing):
        raise RuntimeError(
            f'{cell.cell_id}: nested projection found a partial {label} data tree; '
            'inspect the incomplete target manually'
        )


def _select_nested_scale_facts(
    source_facts: pl.DataFrame, target_cfg: ExperimentCfg
) -> pl.DataFrame:
    pools = target_cfg.generation.chunk_pools
    gold_per_role = {
        'dominant_primary_gold': int(pools.dominant_primary.size or 0),
        'primary_gold': int(pools.other_primary.size or 0),
        'secondary_gold': int(pools.secondary.size or 0),
        'niche_gold': int(pools.niche.size or 0),
    }
    gold = source_facts.filter(pl.col('is_gold')).with_columns(
        pl.col('fact_id').rank('ordinal').over(['query_id', 'facet_id']).alias('_rank')
    )
    gold_limit = pl.lit(0)
    for role, count in gold_per_role.items():
        gold_limit = pl.when(pl.col('cluster_role') == role).then(count).otherwise(gold_limit)
    selected_gold = gold.filter(pl.col('_rank') <= gold_limit).drop('_rank')
    non_gold = source_facts.filter(~pl.col('is_gold')).with_columns(
        pl.col('fact_id').rank('ordinal').over(['query_id', 'cluster_id']).alias('_rank')
    )
    cluster_limit = pl.lit(0)
    for index, spec in enumerate(target_cfg.generation.near_miss_specs, start=1):
        cluster_limit = (
            pl.when(pl.col('cluster_id').str.contains(f'_v5_nm_s{index:02d}_'))
            .then(int(spec.chunks_per_cluster or 0))
            .otherwise(cluster_limit)
        )
    for index, spec in enumerate(pools.background_outliers, start=1):
        cluster_limit = (
            pl.when(pl.col('cluster_id').str.contains(f'_bg_s{index:02d}_'))
            .then(int(spec.chunks_per_cluster or 0))
            .otherwise(cluster_limit)
        )
    selected_non_gold = non_gold.filter(pl.col('_rank') <= cluster_limit).drop('_rank')
    selected = pl.concat([selected_gold, selected_non_gold], how='vertical_relaxed').sort(
        ['query_id', 'fact_id']
    )
    expected = (
        target_cfg.generation.total_gold_chunks() + target_cfg.generation.total_distractor_chunks()
    )
    invalid = selected.group_by('query_id').len().filter(pl.col('len') != expected)
    if invalid.height:
        raise RuntimeError(
            'nested projection did not produce the target candidate mass; '
            f'expected={expected}, examples={invalid.head(5).to_dicts()}'
        )
    return selected


def _hard_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if os.path.samefile(source, target):
            return
        raise FileExistsError(f'nested projection refuses to overwrite {target}')
    if source.stat().st_dev != target.parent.stat().st_dev:
        raise RuntimeError(f'nested projection requires same filesystem: {source} -> {target}')
    os.link(source, target)
