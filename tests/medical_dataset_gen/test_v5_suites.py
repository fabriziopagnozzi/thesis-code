from __future__ import annotations

import hashlib
import json
import os
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from experiments.medical_dataset_gen.scripts.migrate_shared_chunk_embeddings import (
    migrate_suite_chunk_embeddings,
)
from experiments.medical_dataset_gen.scripts.migrate_v4_to_v5 import (
    execute_migration,
    inventory_v4_artifacts,
    rollback_migration,
)

from experiments.medical_dataset_gen.dataset_generation.chunk_materialization import (
    _write_normalized_chunks,
)
from experiments.medical_dataset_gen.dataset_generation.facts import run_make_facts
from experiments.medical_dataset_gen.dataset_generation.planning import run_make_query_plans
from experiments.medical_dataset_gen.dataset_generation.qrels import run_make_qrels
from experiments.medical_dataset_gen.dataset_generation.queries_answers import (
    run_make_queries_answers,
)
from experiments.medical_dataset_gen.embedding import stage as embedding_stage
from experiments.medical_dataset_gen.embedding.artifacts import (
    chunk_embedding_artifacts_ready,
    embedding_artifacts_ready,
)
from experiments.medical_dataset_gen.embedding.cleanup import run_cleanup
from experiments.medical_dataset_gen.evaluation import eval_worker_handler
from experiments.medical_dataset_gen.pipeline.cli import build_parser, selected_stage_names
from experiments.medical_dataset_gen.pipeline.suite import reuse_nested_scale_chunk_embeddings
from experiments.medical_dataset_gen.query_geometry import geom_worker_handler
from experiments.medical_dataset_gen.reports.discovery import discover_suite_experiments
from experiments.medical_dataset_gen.reports.suite_analysis import (
    analysis_series_rows,
    matched_contrast_rows,
    write_suite_factor_figures,
)
from experiments.medical_dataset_gen.suites.core import (
    SuiteSpec,
    SuiteTransform,
    _declared_composition,
    load_suite_manifest,
    load_suite_spec,
    materialize_suite,
    validate_suite,
)
from experiments.medical_dataset_gen.suites.runtime import (
    load_cell_config,
    project_nested_scale_parents,
    resolve_manifest_cell,
    suite_paths_for_cell,
    validate_materialized_suite,
)
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths
from experiments.medical_dataset_gen.utils.io_utils import read_parquet, write_parquet


def test_native_thesis_suite_expands_to_explicit_qwen_cells(tmp_path: Path) -> None:
    spec = load_suite_spec('thesis_v5')
    validation = validate_suite(spec)
    assert len(spec.distributions) == 41
    assert len(validation.resolved_configs) == 164
    assert all('qwen_' in cell_id for cell_id in validation.resolved_configs)
    assert not any(
        'NIC_M02' in cell_id or 'NIC_L02' in cell_id for cell_id in validation.resolved_configs
    )
    assert {distribution.family_id for distribution in spec.distributions.values()} == {
        'balanced_clean',
        'dominance',
        'sparse_niche',
        'near_miss_heavy',
        'background_variant',
        'interaction',
    }

    manifest = materialize_suite(spec, results_dir=tmp_path)
    assert manifest.manifest_version == 2
    assert len(manifest.distributions) == 41
    assert len(manifest.cells) == 164
    assert len(manifest.evaluations) == 164
    assert manifest.analysis_series == []
    assert all(cell.dataset_schema_version == 5 for cell in manifest.cells)
    small = next(
        cell
        for cell in manifest.cells
        if cell.cell_id == 'scale_balanced_small__qwen_unbiased_simple'
    )
    medium = next(
        cell
        for cell in manifest.cells
        if cell.cell_id == 'balanced_reference__qwen_unbiased_simple'
    )
    assert medium.nested_from == small.cell_id
    for anchor in (
        ('scale_balanced_small', 'balanced_reference', 'scale_balanced_large'),
        ('scale_dominance_small', 'dominance_high', 'scale_dominance_large'),
        ('scale_sparse_one_small', 'sparse_one_moderate', 'scale_sparse_one_large'),
        ('scale_near_miss_small', 'scale_near_miss_medium', 'scale_near_miss_large'),
        (
            'scale_compact_background_small',
            'scale_compact_background_medium',
            'scale_compact_background_large',
        ),
    ):
        masses = []
        for distribution_id in anchor:
            cfg = ExperimentCfg.model_validate(
                validation.resolved_configs[f'{distribution_id}__qwen_unbiased_simple']
            )
            masses.append(
                cfg.generation.total_gold_chunks() + cfg.generation.total_distractor_chunks()
            )
        assert masses == [64, 128, 256]


def test_suite_embeddings_are_shared_by_independent_surface(tmp_path: Path) -> None:
    manifest = materialize_suite(load_suite_spec('thesis_v5'), results_dir=tmp_path)
    root = tmp_path / 'v5' / 'suites' / manifest.suite_id
    cells = {cell.cell_id: cell for cell in manifest.cells}

    biased_simple = cells['dominance_mild__qwen_biased_simple']
    unbiased_simple = cells['dominance_mild__qwen_unbiased_simple']
    biased_hardened = cells['dominance_mild__qwen_biased_hardened']
    simple_paths = suite_paths_for_cell(
        root=root,
        cell=biased_simple,
        cfg=load_cell_config(root, biased_simple),
    )
    unbiased_paths = suite_paths_for_cell(
        root=root,
        cell=unbiased_simple,
        cfg=load_cell_config(root, unbiased_simple),
    )
    hardened_paths = suite_paths_for_cell(
        root=root,
        cell=biased_hardened,
        cfg=load_cell_config(root, biased_hardened),
    )

    assert simple_paths.embeddings_paths('chunk_vectors') == unbiased_paths.embeddings_paths(
        'chunk_vectors'
    )
    assert simple_paths.embeddings_paths('chunk_ids') == unbiased_paths.embeddings_paths(
        'chunk_ids'
    )
    assert simple_paths.embeddings_paths('chunk_vectors') != hardened_paths.embeddings_paths(
        'chunk_vectors'
    )
    assert simple_paths.embeddings_paths('query_vectors') == hardened_paths.embeddings_paths(
        'query_vectors'
    )
    assert simple_paths.embeddings_paths('query_ids') == hardened_paths.embeddings_paths(
        'query_ids'
    )
    assert simple_paths.embeddings_paths('query_vectors') != unbiased_paths.embeddings_paths(
        'query_vectors'
    )


def test_shared_chunk_embedding_migration_moves_and_deduplicates(tmp_path: Path) -> None:
    manifest = materialize_suite(load_suite_spec('thesis_v5'), results_dir=tmp_path)
    root = tmp_path / 'v5' / 'suites' / manifest.suite_id
    cells = {cell.cell_id: cell for cell in manifest.cells}
    selected = [
        cells['dominance_mild__qwen_biased_simple'],
        cells['dominance_mild__qwen_unbiased_simple'],
    ]
    vectors = np.arange(6, dtype=np.float32).reshape(2, 3)
    chunk_ids = np.asarray(['c1', 'c2'])
    legacy_paths: list[tuple[Path, Path]] = []
    metadata_paths: list[Path] = []
    for cell in selected:
        attempt_root = root / cell.attempt_root
        attempt_root.mkdir(parents=True, exist_ok=True)
        vector_path = attempt_root / 'embeddings_chunk_vectors.npy'
        ids_path = attempt_root / 'embeddings_chunk_ids.npy'
        metadata_path = attempt_root / 'embeddings_metadata.json'
        np.save(vector_path, vectors)
        np.save(ids_path, chunk_ids)
        metadata_path.write_text(json.dumps({'n_chunks': 2, 'n_queries': 1}))
        legacy_paths.append((vector_path, ids_path))
        metadata_paths.append(metadata_path)

    reports = migrate_suite_chunk_embeddings(
        results_dir=tmp_path,
        suite_id=manifest.suite_id,
        execute=True,
    )

    shared_paths = suite_paths_for_cell(
        root=root,
        cell=selected[0],
        cfg=load_cell_config(root, selected[0]),
    )
    shared_vectors = shared_paths.embeddings_paths('chunk_vectors')
    shared_ids = shared_paths.embeddings_paths('chunk_ids')
    assert len(reports) == 1
    assert reports[0].matching_attempts == 2
    assert reports[0].removed_files == 2
    np.testing.assert_array_equal(np.load(shared_vectors), vectors)
    np.testing.assert_array_equal(np.load(shared_ids), chunk_ids)
    assert all(not path.exists() for pair in legacy_paths for path in pair)
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        assert metadata['chunk_vectors_file'] == str(shared_vectors)
        assert metadata['chunk_ids_file'] == str(shared_ids)


def test_shared_query_embedding_migration_moves_and_deduplicates(tmp_path: Path) -> None:
    manifest = materialize_suite(load_suite_spec('thesis_v5'), results_dir=tmp_path)
    root = tmp_path / 'v5' / 'suites' / manifest.suite_id
    cells = {cell.cell_id: cell for cell in manifest.cells}
    selected = [
        cells['dominance_mild__qwen_biased_simple'],
        cells['dominance_mild__qwen_biased_hardened'],
    ]
    vectors = np.arange(12, dtype=np.float32).reshape(4, 3)
    query_ids = np.asarray(['q1', 'q2', 'q3', 'q4'])
    legacy_paths: list[tuple[Path, Path]] = []
    metadata_paths: list[Path] = []
    for cell in selected:
        attempt_root = root / cell.attempt_root
        attempt_root.mkdir(parents=True, exist_ok=True)
        vector_path = attempt_root / 'embeddings_query_vectors.npy'
        ids_path = attempt_root / 'embeddings_query_ids.npy'
        metadata_path = attempt_root / 'embeddings_metadata.json'
        np.save(vector_path, vectors)
        np.save(ids_path, query_ids)
        metadata_path.write_text(json.dumps({'n_chunks': 1, 'n_queries': 4}))
        legacy_paths.append((vector_path, ids_path))
        metadata_paths.append(metadata_path)

    reports = migrate_suite_chunk_embeddings(
        results_dir=tmp_path,
        suite_id=manifest.suite_id,
        execute=True,
    )

    shared_paths = suite_paths_for_cell(
        root=root,
        cell=selected[0],
        cfg=load_cell_config(root, selected[0]),
    )
    shared_vectors = shared_paths.embeddings_paths('query_vectors')
    shared_ids = shared_paths.embeddings_paths('query_ids')
    assert len(reports) == 1
    assert reports[0].artifact_kind == 'queries'
    assert reports[0].matching_attempts == 2
    assert reports[0].removed_files == 2
    np.testing.assert_array_equal(np.load(shared_vectors), vectors)
    np.testing.assert_array_equal(np.load(shared_ids), query_ids)
    assert all(not path.exists() for pair in legacy_paths for path in pair)
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        assert metadata['query_vectors_file'] == str(shared_vectors)
        assert metadata['query_ids_file'] == str(shared_ids)


@pytest.mark.parametrize(
    ('chunks_ready', 'queries_ready', 'expected_mode'),
    [
        (True, True, 'all_arrays'),
        (True, False, 'queries_only'),
        (False, True, 'chunks_only'),
        (False, False, 'full'),
    ],
)
def test_embed_selects_only_the_missing_shared_side(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chunks_ready: bool,
    queries_ready: bool,
    expected_mode: str,
) -> None:
    validation = validate_suite(load_suite_spec('thesis_v5'))
    cfg = ExperimentCfg.model_validate(
        validation.resolved_configs['dominance_mild__qwen_biased_simple']
    )
    paths = MedicalDatasetGenPaths('branch_fixture', artifact_root=tmp_path)
    vectors = np.zeros((1, 2), dtype=np.float32)
    called: list[str] = []

    def result(mode: str) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        called.append(mode)
        return vectors, vectors, {'n_chunks': 1, 'n_queries': 1}

    monkeypatch.setattr(embedding_stage, 'embedding_artifacts_ready', lambda _: False)
    monkeypatch.setattr(embedding_stage, 'chunk_embedding_artifacts_ready', lambda _: chunks_ready)
    monkeypatch.setattr(embedding_stage, 'query_embedding_artifacts_ready', lambda _: queries_ready)
    monkeypatch.setattr(
        embedding_stage,
        '_reuse_embedding_arrays',
        lambda _cfg, _paths: result('all_arrays'),
    )
    monkeypatch.setattr(
        embedding_stage,
        '_embed_queries_only',
        lambda _cfg, _paths: result('queries_only'),
    )
    monkeypatch.setattr(
        embedding_stage,
        '_embed_chunks_only',
        lambda _cfg, _paths: result('chunks_only'),
    )
    monkeypatch.setattr(
        embedding_stage,
        '_embed_sentence_transformers_streaming',
        lambda _cfg, _paths: result('full'),
    )

    embedding_stage.run_embed(cfg, paths)

    assert called == [expected_mode]


def test_cleanup_removes_only_chunk_embedding_arrays(tmp_path: Path) -> None:
    validation = validate_suite(load_suite_spec('thesis_v5'))
    cfg = ExperimentCfg.model_validate(
        validation.resolved_configs['dominance_mild__qwen_biased_simple']
    )
    paths = MedicalDatasetGenPaths('cleanup_fixture', artifact_root=tmp_path)
    paths.ensure_dirs()
    for artifact in ('chunk_vectors', 'chunk_ids', 'query_vectors', 'query_ids', 'metadata'):
        path = paths.embeddings_paths(artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact == 'metadata':
            path.write_text(json.dumps({'model_name': cfg.embeddings.model_name}))
        else:
            np.save(path, np.zeros((1, 2), dtype=np.float32))

    run_cleanup(cfg, paths)

    assert not paths.embeddings_paths('chunk_vectors').exists()
    assert not paths.embeddings_paths('chunk_ids').exists()
    assert paths.embeddings_paths('query_vectors').is_file()
    assert paths.embeddings_paths('query_ids').is_file()
    assert paths.embeddings_paths('metadata').is_file()


def test_cleanup_stage_is_explicit_only() -> None:
    parser = build_parser()

    default_args = parser.parse_args(['--exp', 'fixture'])
    assert 'cleanup' not in selected_stage_names(parser, default_args)

    open_ended_args = parser.parse_args(['--exp', 'fixture', '--from', 'embed'])
    assert 'cleanup' not in selected_stage_names(parser, open_ended_args)

    explicit_args = parser.parse_args(['--exp', 'fixture', '--stages', 'cleanup'])
    assert selected_stage_names(parser, explicit_args) == ['cleanup']

    bounded_args = parser.parse_args(['--exp', 'fixture', '--to', 'cleanup'])
    assert selected_stage_names(parser, bounded_args)[-1] == 'cleanup'


def test_native_v5_rejects_multiple_gold_regions_per_facet() -> None:
    raw = load_suite_spec('thesis_v5').model_dump(mode='python')
    raw['distributions']['balanced_reference']['config']['generation']['chunk_pools'][
        'dominant_primary'
    ]['num_clusters'] = 2
    with pytest.raises(ValueError, match='one materialized gold region'):
        validate_suite(SuiteSpec.model_validate(raw))


def test_v5_evaluation_worker_keeps_resolved_suite_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spawned evaluators must not reconstruct a legacy experiment path."""
    manifest = materialize_suite(load_suite_spec('thesis_v5'), results_dir=tmp_path)
    cell = next(
        item
        for item in manifest.cells
        if item.cell_id == 'scale_balanced_large__qwen_biased_simple'
    )
    root, resolved_cell = resolve_manifest_cell(
        results_dir=tmp_path,
        suite_id=manifest.suite_id,
        cell_id=cell.cell_id,
    )
    cfg = load_cell_config(root, resolved_cell)
    paths = suite_paths_for_cell(root=root, cell=resolved_cell, cfg=cfg)
    expected = paths.table_path('chunk_documents')
    assert pickle.loads(pickle.dumps(paths)).table_path('chunk_documents') == expected

    class ExpectedPathReached(Exception):
        pass

    def stop_after_path_check(
        supplied_paths: MedicalDatasetGenPaths,
        table: str,
        columns: object,
        optional_columns: object = (),
    ) -> pl.DataFrame:
        assert supplied_paths.table_path('chunk_documents') == expected
        assert table == 'chunk_documents'
        raise ExpectedPathReached

    monkeypatch.setattr(eval_worker_handler, 'load_selected_parquet_columns', stop_after_path_check)
    with pytest.raises(ExpectedPathReached):
        eval_worker_handler.init_evaluation_worker(cfg, paths)


def test_v5_geometry_plot_worker_keeps_resolved_suite_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Geometry plot workers must use suite paths under multiprocessing too."""
    manifest = materialize_suite(load_suite_spec('thesis_v5'), results_dir=tmp_path)
    cell = next(
        item
        for item in manifest.cells
        if item.cell_id == 'scale_balanced_large__qwen_biased_simple'
    )
    root, resolved_cell = resolve_manifest_cell(
        results_dir=tmp_path,
        suite_id=manifest.suite_id,
        cell_id=cell.cell_id,
    )
    cfg = load_cell_config(root, resolved_cell)
    paths = suite_paths_for_cell(root=root, cell=resolved_cell, cfg=cfg)
    expected = paths.table_path('chunk_documents')

    class ExpectedPathReached(Exception):
        pass

    def stop_after_path_check(
        supplied_paths: MedicalDatasetGenPaths,
        table: str,
        columns: object,
        optional_columns: object = (),
    ) -> pl.DataFrame:
        assert supplied_paths.table_path('chunk_documents') == expected
        assert table == 'chunk_documents'
        raise ExpectedPathReached

    monkeypatch.setattr(geom_worker_handler, 'load_selected_parquet_columns', stop_after_path_check)
    with pytest.raises(ExpectedPathReached):
        geom_worker_handler.init_query_geometry_worker(
            cfg=cfg,
            paths=paths,
            out_dir=str(tmp_path / 'figures'),
            query_group_by_id={},
            query_dir_name_by_id={},
            selected_plot_names=None,
        )


def test_v5_scale_support_has_exact_nested_candidate_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 0.5/1/2 support levels retain every smaller v5 document ID."""
    monkeypatch.setattr(MedicalDatasetGenPaths, 'results_dir', tmp_path / 'results')
    spec = load_suite_spec('thesis_v5')
    resolved = validate_suite(spec).resolved_configs
    ids_by_level: dict[str, set[str]] = {}
    counts_by_level: dict[str, set[int]] = {}

    distribution_by_level = {
        'small': 'scale_balanced_small',
        'medium': 'balanced_reference',
        'large': 'scale_balanced_large',
    }
    for level, distribution_id in distribution_by_level.items():
        cell_id = f'{distribution_id}__qwen_unbiased_simple'
        raw = resolved[cell_id]
        raw['generation']['query_limit'] = 2
        cfg = ExperimentCfg.model_validate(raw)
        paths = MedicalDatasetGenPaths(f'scale_nesting_{level}')
        paths.ensure_dirs()
        run_make_query_plans(cfg, paths)
        run_make_facts(cfg, paths)

        # Rendering is orthogonal to ID allocation.  Feed the structured facts
        # through the normal document/membership materializer with a stable,
        # non-colliding surface so this test exercises the exact qrel path
        # without launching a process pool three times.
        fact_rows = read_parquet(paths, 'clinical_facts').with_columns(
            pl.col('chunk_reuse_key').alias('text'),
            pl.lit('unassigned').alias('chunk_id'),
        )
        _write_normalized_chunks(paths, fact_rows, stable_document_ids=True)
        qrels = run_make_qrels(cfg, paths)
        ids_by_level[level] = set(str(value) for value in qrels['chunk_id'].to_list())
        counts_by_level[level] = set(
            int(value) for value in qrels.group_by('query_id').len()['len'].to_list()
        )

    assert counts_by_level == {'small': {64}, 'medium': {128}, 'large': {256}}
    assert ids_by_level['small'] <= ids_by_level['medium'] <= ids_by_level['large']


def test_terminal_scale_cell_projects_smaller_supports_without_rendering_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / 'results'
    monkeypatch.setattr(MedicalDatasetGenPaths, 'results_dir', results)
    manifest = materialize_suite(load_suite_spec('thesis_v5'), results_dir=results)
    root, source_cell = resolve_manifest_cell(
        results_dir=results,
        suite_id=manifest.suite_id,
        cell_id='scale_balanced_large__qwen_unbiased_simple',
    )
    source_config_path = root / source_cell.resolved_config_path
    source_raw = yaml.safe_load(source_config_path.read_text())
    assert isinstance(source_raw, dict)
    source_raw['generation']['query_limit'] = 2
    source_config_path.write_text(yaml.safe_dump(source_raw, sort_keys=False))
    source_cfg = load_cell_config(root, source_cell)
    source_paths = suite_paths_for_cell(root=root, cell=source_cell, cfg=source_cfg)
    source_paths.ensure_dirs()
    run_make_query_plans(source_cfg, source_paths)
    run_make_facts(source_cfg, source_paths)
    # Fact IDs retain a human-readable random suffix and can collide across
    # clusters.  Force that real-world edge case here: projection must use the
    # query/cluster/fact identity, rather than multiplying rows on ``fact_id``.
    source_facts = read_parquet(source_paths, 'clinical_facts')
    q1_clusters = (
        source_facts.filter(pl.col('query_id') == 'q1')
        .select(['cluster_id', 'fact_id'])
        .unique(maintain_order=True)
        .head(2)
    )
    assert q1_clusters.height == 2
    retained_fact_id = str(q1_clusters['fact_id'][0])
    colliding_cluster_id = str(q1_clusters['cluster_id'][1])
    source_facts = source_facts.with_columns(
        pl.when((pl.col('query_id') == 'q1') & (pl.col('cluster_id') == colliding_cluster_id))
        .then(pl.lit(retained_fact_id))
        .otherwise(pl.col('fact_id'))
        .alias('fact_id')
    )
    write_parquet(source_paths, 'clinical_facts', source_facts)
    source_fact_rows = source_facts.with_columns(
        pl.col('chunk_reuse_key').alias('text'),
        pl.lit('unassigned').alias('chunk_id'),
    )
    _write_normalized_chunks(source_paths, source_fact_rows, stable_document_ids=True)
    run_make_qrels(source_cfg, source_paths)
    run_make_queries_answers(source_cfg, source_paths)

    # A different run profile can already have materialized the shared query
    # surface while this profile still needs its chunk surface projected.
    _, existing_query_cell = resolve_manifest_cell(
        results_dir=results,
        suite_id=manifest.suite_id,
        cell_id='balanced_reference__qwen_unbiased_simple',
    )
    existing_query_cfg = load_cell_config(root, existing_query_cell)
    existing_query_paths = suite_paths_for_cell(
        root=root, cell=existing_query_cell, cfg=existing_query_cfg
    )
    for table in ('queries', 'gold_answers'):
        source_path = source_paths.table_path(table)
        target_path = existing_query_paths.table_path(table)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        os.link(source_path, target_path)

    projected = project_nested_scale_parents(
        root=root,
        source_cell=source_cell,
        source_cfg=source_cfg,
    )
    assert projected == [
        'balanced_reference__qwen_unbiased_simple',
        'scale_balanced_small__qwen_unbiased_simple',
    ]

    source_ids = set(read_parquet(source_paths, 'qrels')['chunk_id'].to_list())
    for distribution_id, expected_count in (
        ('balanced_reference', 128),
        ('scale_balanced_small', 64),
    ):
        _, target_cell = resolve_manifest_cell(
            results_dir=results,
            suite_id=manifest.suite_id,
            cell_id=f'{distribution_id}__qwen_unbiased_simple',
        )
        target_cfg = load_cell_config(root, target_cell)
        target_paths = suite_paths_for_cell(root=root, cell=target_cell, cfg=target_cfg)
        qrels = read_parquet(target_paths, 'qrels')
        assert set(qrels.group_by('query_id').len()['len'].to_list()) == {expected_count}
        assert set(qrels['chunk_id'].to_list()) <= source_ids
        assert os.path.samefile(
            source_paths.table_path('chunk_documents'),
            target_paths.table_path('chunk_documents'),
        )
        assert os.path.samefile(
            source_paths.table_path('queries'), target_paths.table_path('queries')
        )


def test_nested_scale_reuses_verified_source_chunk_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / 'results'
    monkeypatch.setattr(MedicalDatasetGenPaths, 'results_dir', results)
    manifest = materialize_suite(load_suite_spec('thesis_v5'), results_dir=results)
    root, source_cell = resolve_manifest_cell(
        results_dir=results,
        suite_id=manifest.suite_id,
        cell_id='scale_balanced_large__qwen_unbiased_simple',
    )
    source_config_path = root / source_cell.resolved_config_path
    source_raw = yaml.safe_load(source_config_path.read_text())
    assert isinstance(source_raw, dict)
    source_raw['generation']['query_limit'] = 2
    source_config_path.write_text(yaml.safe_dump(source_raw, sort_keys=False))
    source_cfg = load_cell_config(root, source_cell)
    source_paths = suite_paths_for_cell(root=root, cell=source_cell, cfg=source_cfg)
    source_paths.ensure_dirs()
    run_make_query_plans(source_cfg, source_paths)
    run_make_facts(source_cfg, source_paths)
    source_fact_rows = read_parquet(source_paths, 'clinical_facts').with_columns(
        pl.col('chunk_reuse_key').alias('text'),
        pl.lit('unassigned').alias('chunk_id'),
    )
    _write_normalized_chunks(source_paths, source_fact_rows, stable_document_ids=True)
    run_make_qrels(source_cfg, source_paths)
    run_make_queries_answers(source_cfg, source_paths)
    project_nested_scale_parents(root=root, source_cell=source_cell, source_cfg=source_cfg)

    chunks = read_parquet(source_paths, 'chunk_documents')['chunk_id'].to_list()
    queries = read_parquet(source_paths, 'queries')['query_id'].to_list()
    chunk_id_dtype = f'U{max(len(str(chunk_id)) for chunk_id in chunks)}'
    query_id_dtype = f'U{max(len(str(query_id)) for query_id in queries)}'
    source_paths.embeddings_paths('chunk_vectors').parent.mkdir(parents=True, exist_ok=True)
    np.save(
        source_paths.embeddings_paths('chunk_vectors'), np.zeros((len(chunks), 2), dtype=np.float32)
    )
    np.save(source_paths.embeddings_paths('chunk_ids'), np.asarray(chunks, dtype=chunk_id_dtype))
    source_paths.embeddings_paths('query_vectors').parent.mkdir(parents=True, exist_ok=True)
    np.save(
        source_paths.embeddings_paths('query_vectors'),
        np.zeros((len(queries), 2), dtype=np.float32),
    )
    np.save(source_paths.embeddings_paths('query_ids'), np.asarray(queries, dtype=query_id_dtype))
    source_paths.embeddings_paths('metadata').write_text(
        json.dumps({'n_chunks': len(chunks), 'n_queries': len(queries)})
    )

    _, target_cell = resolve_manifest_cell(
        results_dir=results,
        suite_id=manifest.suite_id,
        cell_id='balanced_reference__qwen_unbiased_simple',
    )
    target_cfg = load_cell_config(root, target_cell)
    target_paths = suite_paths_for_cell(root=root, cell=target_cell, cfg=target_cfg)
    target_paths.ensure_dirs()
    assert reuse_nested_scale_chunk_embeddings(
        root=root,
        cell=target_cell,
        cfg=target_cfg,
        paths=target_paths,
        requested={'embed'},
    )
    assert os.path.samefile(
        source_paths.embeddings_paths('chunk_vectors'),
        target_paths.embeddings_paths('chunk_vectors'),
    )


def test_embedding_artifact_readiness_rejects_truncated_chunk_ids(tmp_path: Path) -> None:
    paths = MedicalDatasetGenPaths('long_chunk_ids', artifact_root=tmp_path)
    paths.ensure_dirs()
    long_chunk_id = 'chunk_' + ('a' * 64)
    write_parquet(
        paths,
        'chunk_documents',
        pl.DataFrame({'chunk_id': [long_chunk_id], 'text': ['clinical evidence']}),
    )
    write_parquet(paths, 'queries', pl.DataFrame({'query_id': ['q1'], 'query_text': ['query']}))
    np.save(paths.embeddings_paths('chunk_vectors'), np.zeros((1, 2), dtype=np.float32))
    np.save(paths.embeddings_paths('query_vectors'), np.zeros((1, 2), dtype=np.float32))
    np.save(paths.embeddings_paths('chunk_ids'), np.asarray([long_chunk_id], dtype='U32'))
    np.save(paths.embeddings_paths('query_ids'), np.asarray(['q1'], dtype='U2'))
    paths.embeddings_paths('metadata').write_text(json.dumps({'n_chunks': 1, 'n_queries': 1}))

    assert not chunk_embedding_artifacts_ready(paths)
    assert not embedding_artifacts_ready(paths)


def test_scale_support_rejects_fractional_cluster_support() -> None:
    spec = load_suite_spec('thesis_v5')
    raw = spec.model_dump(mode='python')
    raw['suite_id'] = 'fractional_scale'
    raw['distributions']['balanced_reference']['nested_from'] = None
    raw['distributions']['balanced_reference']['transforms'] = [
        SuiteTransform(op='scale_support', multiplier=0.3).model_dump(mode='python')
    ]
    raw['evaluations'] = [
        {
            'evaluation_id': 'one',
            'distribution_ids': ['balanced_reference'],
            'run_profile_ids': ['qwen_unbiased_simple'],
        }
    ]
    raw['comparison_groups'] = []
    raw['analysis_series'] = []
    with pytest.raises(ValueError, match='integral chunks_per_cluster'):
        validate_suite(SuiteSpec.model_validate(raw))


def test_native_v5_compositions_match_declared_factorial_design() -> None:
    validation = validate_suite(load_suite_spec('thesis_v5'))

    def composition(distribution_id: str) -> dict[str, object]:
        cfg = ExperimentCfg.model_validate(
            validation.resolved_configs[f'{distribution_id}__qwen_unbiased_simple']
        )
        return _declared_composition(cfg)

    assert composition('dominance_high')['gold_mass_vector'] == [54, 14, 14, 14]
    assert composition('sparse_one_severe')['gold_mass_vector'] == [30, 30, 30, 6]
    assert composition('sparse_two_severe')['gold_mass_vector'] == [40, 40, 8, 8]
    assert composition('sparse_two_severe')['niche_count'] == 2

    two_change = composition('near_miss_h48_two_change')['near_miss_topology']
    mixed = composition('near_miss_h48')['near_miss_topology']
    one_change = composition('near_miss_h48_one_change')['near_miss_topology']
    assert [item['num_clusters'] for item in two_change] == [2, 2]
    assert [item['num_clusters'] for item in mixed] == [1, 1, 1, 1]
    assert [item['num_clusters'] for item in one_change] == [2, 2]
    assert all(item['chunks_per_cluster'] == 12 for item in two_change + mixed + one_change)

    near_miss = composition('near_miss_h24')
    dilution = composition('dilution_far_b40')
    assert near_miss['pool_mass'] == dilution['pool_mass'] == 136
    assert near_miss['near_miss_mass'] == 24
    assert dilution['near_miss_mass'] == 0
    assert dilution['background_mass'] == 40

    grid = [
        composition(f'background_far_{topology}') for topology in ('32x1', '16x2', '8x4', '4x8')
    ]
    assert all(item['background_mass'] == 32 for item in grid)
    assert {item['background_topology_components'][0]['num_clusters'] for item in grid} == {
        4,
        8,
        16,
        32,
    }


def test_native_v5_rejects_declared_composition_factor_drift() -> None:
    raw = load_suite_spec('thesis_v5').model_dump(mode='python')
    raw['distributions']['dominance_high']['factors']['dominance_share'] = 0.5
    with pytest.raises(ValueError, match='declared factor dominance_share'):
        validate_suite(SuiteSpec.model_validate(raw))


def test_native_v5_rejects_incomplete_background_topology_ladder() -> None:
    raw = load_suite_spec('thesis_v5').model_dump(mode='python')
    group = next(
        candidate
        for candidate in raw['comparison_groups']
        if candidate['comparison_id'] == 'background_topology'
    )
    group['distribution_ids'].remove('background_far_4x8')
    with pytest.raises(ValueError, match='declared factor_levels differ'):
        validate_suite(SuiteSpec.model_validate(raw))


def test_analysis_series_select_each_manifest_declared_budget(tmp_path: Path) -> None:
    raw = load_suite_spec('thesis_v5').model_dump(mode='python')
    raw['analysis_series'] = [
        {
            'series_id': 'declared_budget_fixture',
            'analysis_block': 'scale',
            'reference_point_id': 'medium',
            'lambda_source_k': 6,
            'points': [
                {
                    'point_id': 'small',
                    'distribution_id': 'scale_balanced_small',
                    'run_profile_id': 'qwen_unbiased_simple',
                    'k': 4,
                    'factors': {'scale': 'small', 'pool_mass': 64},
                },
                {
                    'point_id': 'medium',
                    'distribution_id': 'balanced_reference',
                    'run_profile_id': 'qwen_unbiased_simple',
                    'k': 6,
                    'factors': {'scale': 'medium', 'pool_mass': 128},
                },
                {
                    'point_id': 'large',
                    'distribution_id': 'scale_balanced_large',
                    'run_profile_id': 'qwen_unbiased_simple',
                    'k': 10,
                    'factors': {'scale': 'large', 'pool_mass': 256},
                },
            ],
        }
    ]
    manifest = materialize_suite(SuiteSpec.model_validate(raw), results_dir=tmp_path)
    series = manifest.analysis_series[0]
    cells = {cell.cell_id: cell for cell in manifest.cells}
    rows = []
    for point in series.points:
        cell = cells[f'{point.distribution_id}__{point.run_profile_id}']
        rows.append(
            {
                'Experiment': cell.name,
                'k': point.k,
                'IncludeInCausalSummaries': True,
                'Delta_FacLoc_MMR_FCP': 0.0,
            }
        )
    output = analysis_series_rows(
        manifest=manifest,
        comparison_rows=rows,
        enforce_strict=True,
        scope_cell_ids={
            f'{point.distribution_id}__{point.run_profile_id}' for point in series.points
        },
    )
    assert [(row['PointId'], row['k']) for row in output] == [
        ('small', 4),
        ('medium', 6),
        ('large', 10),
    ]


def test_replace_planned_refuses_any_native_artifact(tmp_path: Path) -> None:
    spec = load_suite_spec('thesis_v5')
    materialize_suite(spec, results_dir=tmp_path)
    root = tmp_path / 'v5' / 'suites' / 'thesis_v5'
    marker = root / 'distributions' / 'balanced_reference' / 'data' / 'schema-v5' / 'qrels.parquet'
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b'non-empty artifact')
    with pytest.raises(ValueError, match='artifacts'):
        materialize_suite(spec, results_dir=tmp_path, replace_planned=True)


def test_prune_planned_removes_only_obsolete_metadata_and_preserves_data(tmp_path: Path) -> None:
    spec = load_suite_spec('thesis_v5')
    extended_raw = deepcopy(spec.model_dump(mode='python'))
    obsolete = deepcopy(extended_raw['distributions']['dilution_far_b40'])
    obsolete['factors'] = {
        'scale': 'medium',
        'gold_mass_vector': [24, 24, 24, 24],
        'niche_count': 0,
        'near_miss_mass': 16,
        'background_mass': 8,
        'background_topology': '2x4',
        'background_shell': 'far',
    }
    obsolete['transforms'] = [
        {'op': 'set_background_mass', 'mass': 8, 'num_clusters': 2},
    ]
    extended_raw['distributions']['obsolete_planned'] = obsolete
    extended_raw['evaluations'][0]['distribution_ids'].append('obsolete_planned')
    materialize_suite(SuiteSpec.model_validate(extended_raw), results_dir=tmp_path)
    root = tmp_path / 'v5' / 'suites' / 'thesis_v5'
    data_marker = (
        root / 'distributions' / 'balanced_reference' / 'data' / 'schema-v5' / 'marker.txt'
    )
    data_marker.parent.mkdir(parents=True)
    data_marker.write_text('keep generated smoke data\n')

    manifest = materialize_suite(
        spec,
        results_dir=tmp_path,
        prune_planned=True,
    )
    assert len(manifest.distributions) == 41
    assert len(manifest.cells) == 164
    assert data_marker.read_text() == 'keep generated smoke data\n'
    assert not (root / 'distributions' / 'obsolete_planned').exists()


def test_refresh_planned_execution_preserves_data_but_rejects_attempt_artifacts(
    tmp_path: Path,
) -> None:
    spec = load_suite_spec('thesis_v5')
    manifest = materialize_suite(spec, results_dir=tmp_path)
    root = tmp_path / 'v5' / 'suites' / manifest.suite_id
    data_marker = (
        root / 'distributions' / 'balanced_reference' / 'data' / 'schema-v5' / 'marker.txt'
    )
    data_marker.parent.mkdir(parents=True)
    data_marker.write_text('generated data stay in place\n')
    raw = spec.model_dump(mode='python')
    raw['run_profiles']['qwen_unbiased_simple']['config']['embeddings']['device'] = 'cpu'
    refreshed = materialize_suite(
        SuiteSpec.model_validate(raw),
        results_dir=tmp_path,
        refresh_planned_execution=True,
    )
    assert data_marker.read_text() == 'generated data stay in place\n'
    assert refreshed.cells[0].status == 'planned'

    attempt = (
        root
        / 'distributions'
        / 'balanced_reference'
        / 'runs'
        / 'qwen_unbiased_simple'
        / 'attempts'
        / 'initial'
    )
    attempt.mkdir(parents=True)
    (attempt / 'embeddings_query_ids.npy').write_bytes(b'incomplete')
    with pytest.raises(ValueError, match='attempt artifacts'):
        materialize_suite(
            SuiteSpec.model_validate(raw),
            results_dir=tmp_path,
            refresh_planned_execution=True,
        )


def test_strict_suite_validation_rejects_undeclared_factor_drift() -> None:
    spec = load_suite_spec('thesis_v5')
    raw = spec.model_dump(mode='python')
    raw['distributions']['dominance_mild']['config']['generation']['chunk_pools'][
        'background_outliers'
    ] = [{'num_clusters': 4, 'chunks_per_cluster': 8, 'changes': ['condition', 'subgroup', 'axis']}]
    with pytest.raises(ValueError, match='undeclared configuration drift'):
        validate_suite(SuiteSpec.model_validate(raw))


def test_strict_matched_report_rejects_partial_declared_cross(tmp_path: Path) -> None:
    manifest = materialize_suite(load_suite_spec('thesis_v5'), results_dir=tmp_path)
    cells = {
        cell.cell_id: cell
        for cell in manifest.cells
        if cell.run_profile_id == 'qwen_unbiased_simple'
        and cell.distribution_id in {'scale_balanced_small', 'balanced_reference'}
    }
    rows = [
        {
            'Experiment': cell.name,
            'EmbeddingModel': 'Qwen/Qwen3-Embedding-0.6B',
            'k': 6,
            'IncludeInCausalSummaries': True,
            'Delta_FacLoc_MMR_FCP': 0.0,
        }
        for cell in cells.values()
    ]
    with pytest.raises(ValueError, match='scale_balanced: missing cells'):
        matched_contrast_rows(manifest=manifest, comparison_rows=rows, enforce_strict=True)

    # An endpoint-only smoke request deliberately excludes the medium cell;
    # strict mode must not mislabel that out-of-scope member as an artifact
    # failure.  By contrast, the assertion above still catches a missing cell
    # when the whole declared group is in scope.
    scoped = matched_contrast_rows(
        manifest=manifest,
        comparison_rows=rows,
        enforce_strict=True,
        scope_cell_ids={cell.cell_id for cell in cells.values()},
    )
    assert scoped == []


def test_suite_factor_figures_use_declared_factor_columns(tmp_path: Path) -> None:
    rows = []
    for level, value in (('small', -0.1), ('medium', 0.0), ('large', 0.1)):
        row: dict[str, object] = {
            'Comparison': 'scale_balanced',
            'AnalysisBlock': 'scale',
            'RunProfile': 'qwen_unbiased_simple',
            'EmbeddingModel': 'Qwen/Qwen3-Embedding-0.6B',
            'k': 6,
            'Factor_scale': level,
            'FactorOrder_scale': '["small", "medium", "large"]',
        }
        for metric in (
            'Delta_FacLoc_MMR_FCP',
            'Delta_FacLoc_MMR_FacetCoverage',
            'Delta_FacLoc_MMR_AllFacetCoverageRate',
            'Delta_FacLoc_MMR_AllFacetCleanRate',
            'Delta_FacLoc_MMR_Precision',
            'Delta_FacLoc_MMR_alpha_nDCG',
        ):
            row[metric] = value
        rows.append(row)
    written = write_suite_factor_figures(output_dir=tmp_path, contrast_rows=rows)
    assert {path.name for path in written} == {
        'scale_by_dataset_size.png',
        'scale_by_dataset_size.pdf',
    }
    assert all(path.is_file() for path in written)


def test_migration_moves_byte_identical_v4_artifacts_and_rolls_back(tmp_path: Path) -> None:
    results = tmp_path / 'results'
    source_stats = _make_legacy_fixture(results)
    source_hash = _sha256(source_stats)
    v2_marker = results / 'BAL_TEST' / 'v2' / 'legacy_marker.txt'
    v2_marker.parent.mkdir(parents=True)
    v2_marker.write_text('must remain untouched\n')
    inventory = inventory_v4_artifacts(results)
    assert len(inventory) == 1
    assert len(inventory[0].runs) == 1

    execute_migration(
        results_dir=results,
        suite_id='fixture',
        inventory=inventory,
        include_cache=False,
    )
    target_stats = (
        results / 'v5/suites/fixture/distributions/BAL_TEST/runs/'
        'unbiased_q_natural_f_simple_c_qwen3_06/attempts/migrated-v4/evaluation_stats.parquet'
    )
    assert target_stats.is_file()
    assert _sha256(target_stats) == source_hash
    assert not source_stats.exists()
    assert v2_marker.read_text() == 'must remain untouched\n'
    manifest = load_suite_manifest(results, 'fixture')
    assert manifest.origin == 'migrated_v4'
    assert manifest.cells[0].dataset_schema_version == 4
    validation = validate_materialized_suite(results_dir=results, suite_id='fixture')
    assert validation.errors == ()
    # The executable migration entry point is idempotent once the manifest
    # exists; no source tree is needed for a second invocation.
    execute_migration(results_dir=results, suite_id='fixture', include_cache=False)

    warnings: list[str] = []
    records = discover_suite_experiments(
        results,
        suite_id='fixture',
        where='origin=migrated_v4',
        warnings=warnings,
    )
    assert len(records) == 1
    assert records[0].paths.table_path('evaluation_stats') == target_stats

    rollback_migration(results_dir=results, suite_id='fixture')
    assert source_stats.is_file()
    assert _sha256(source_stats) == source_hash
    assert v2_marker.read_text() == 'must remain untouched\n'


def test_failed_migration_restores_sources(tmp_path: Path) -> None:
    results = tmp_path / 'results'
    source_stats = _make_legacy_fixture(results, include_clinical_facts=False)
    inventory = inventory_v4_artifacts(results)
    with pytest.raises(FileNotFoundError, match='clinical_facts'):
        execute_migration(
            results_dir=results,
            suite_id='broken',
            inventory=inventory,
            include_cache=False,
        )
    assert source_stats.is_file()
    assert (results / 'BAL_TEST/_shared_v4').is_dir()
    assert not (results / 'v5/suites/broken').exists()


def test_migration_adopts_only_valid_qwen_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / 'results'
    _make_legacy_fixture(results)
    cache_root = tmp_path / 'cache'
    monkeypatch.setattr(MedicalDatasetGenPaths, 'cache_dir', cache_root)
    signature = 'a' * 64
    qwen_shard = cache_root / 'chunk_embeddings' / signature / '00.parquet'
    _write_parquet(
        qwen_shard,
        pa.table(
            {
                'chunk_embedding_cache_key': ['key'],
                'chunk_id': ['chunk_1'],
                'text_sha256': ['text'],
                'embedding_signature': [signature],
                'embedding_signature_payload_json': ['{"model_name":"Qwen/Qwen3-Embedding-0.6B"}'],
                'dimension': [1024],
                'embedding': [[0.0] * 1024],
                'embedding_payload_sha256': ['payload'],
            }
        ),
    )
    legacy_shard = cache_root / 'chunk_embeddings' / ('b' * 64) / '00.parquet'
    _write_parquet(
        legacy_shard,
        pa.table(
            {
                'embedding_signature_payload_json': ['{"model_name":"BAAI/bge-m3"}'],
                'dimension': [1024],
            }
        ),
    )

    execute_migration(results_dir=results, suite_id='cache_fixture', include_cache=True)
    assert not qwen_shard.exists()
    assert (cache_root / 'v5/chunk_embeddings' / signature / '00.parquet').is_file()
    assert legacy_shard.is_file()
    rollback_migration(results_dir=results, suite_id='cache_fixture')
    assert qwen_shard.is_file()


def _make_legacy_fixture(results: Path, *, include_clinical_facts: bool = True) -> Path:
    parent = results / 'BAL_TEST'
    parent.mkdir(parents=True)
    (parent / '_config.yaml').write_text(
        """dataset_schema_version: 2
global: {seed: 42, conditions: 2}
generation:
  chunk_pools:
    dominant_primary: {size: 10, distractors: []}
    other_primary: {size: 10, distractors: []}
    secondary: {size: 8, distractors: []}
    niche: {size: 1, num_clusters_per_query: 0, distractors: []}
    background_outliers: []
retrieval: {pool_scope: query_local, candidate_pool_n: 1000, strategies: [top_k, mmr, fac_loc]}
"""
    )
    (parent / '_exp_family.yaml').write_text(
        'family_id: balanced_clean\nfamily_label: Balanced clean distributions\n'
    )
    shared = parent / '_shared_v4'
    _write_parquet(shared / 'base/query_plans.parquet', pa.table({'query_id': ['q1']}))
    if include_clinical_facts:
        _write_parquet(shared / 'base/clinical_facts.parquet', pa.table({'query_id': ['q1']}))
    _write_parquet(
        shared / 'chunks/simple_c/chunk_documents.parquet',
        pa.table({'chunk_id': ['c1'], 'text': ['example']}),
    )
    _write_parquet(
        shared / 'chunks/simple_c/chunk_memberships.parquet', pa.table({'chunk_id': ['c1']})
    )
    _write_parquet(
        shared / 'chunks/simple_c/qrels.parquet',
        pa.table(
            {
                'query_id': ['q1'],
                'cluster_role': ['dominant_primary_gold'],
                'facet_id': ['f1'],
                'is_gold': [True],
                'distractor_type': [None],
            }
        ),
    )
    _write_parquet(
        shared / 'queries/unbiased_q_natural_f/queries.parquet', pa.table({'query_id': ['q1']})
    )
    run = parent / 'unbiased_q_natural_f_simple_c_qwen3_06'
    run.mkdir()
    (run / '_subconfig.yaml').write_text(
        """generation: {focus_mode: natural, query_structure: balanced, chunk_text_style: ontology_explicit}
embeddings: {model_name: Qwen/Qwen3-Embedding-0.6B}
"""
    )
    stats = run / 'v4/evaluation_stats.parquet'
    _write_parquet(
        stats,
        pa.table(
            {
                'strategy': ['top_k'],
                'lam': [0.0],
                'k': [6],
                'n_queries': [1],
                'FacetCoveragePurity@k': [1.0],
                'lambda_selection_split': ['validation'],
                'report_split': ['test'],
                'lambda_selection_metric': ['FacetCoveragePurity@k'],
                'lambda_selection_metric_value': [1.0],
            }
        ),
    )
    _write_parquet(run / 'v4/evaluation_results.parquet', pa.table({'query_id': ['q1']}))
    return stats


def _write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
