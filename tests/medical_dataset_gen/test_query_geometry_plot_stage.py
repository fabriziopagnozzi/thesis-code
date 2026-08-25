from pathlib import Path
from typing import cast

import pytest

from experiments.medical_dataset_gen.query_geometry import plot_stage
from experiments.medical_dataset_gen.query_geometry.plot_stage import (
    _ensure_query_geometry_chunk_embeddings,
    _explicit_query_groups,
    _query_output_directory,
    parse_geom_plots_cli_args,
)
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths


def test_geom_plot_cli_parses_selective_thesis_export() -> None:
    options = parse_geom_plots_cli_args(
        [
            '--plots',
            'candidate_pool_umap,query_cosine_heatmap',
            '--query-ids',
            'q1,q1',
            '--output-dir',
            '/tmp/query-geometry-export',
            '--umap-neighbors',
            '25',
            '--umap-min-dist',
            '0.30',
        ]
    )

    assert options.selected_plots == {
        'candidate_pool_umap',
        'query_cosine_heatmap',
    }
    assert options.query_ids == ('q1',)
    assert options.output_dir == Path('/tmp/query-geometry-export')
    assert options.umap_neighbors == 25
    assert options.umap_min_dist == 0.30


def test_explicit_query_selection_requires_query_and_embedding() -> None:
    with pytest.raises(ValueError, match='missing from query embeddings: q1'):
        _explicit_query_groups(
            query_ids=('q1',),
            available_query_ids={'q1'},
            embedded_query_ids=set(),
        )


def test_explicit_query_selection_dispatches_requested_ids_in_order() -> None:
    groups = _explicit_query_groups(
        query_ids=('q2', 'q1'),
        available_query_ids={'q1', 'q2', 'q3'},
        embedded_query_ids={'q1', 'q2', 'q3'},
    )

    assert groups == {'manual': ['q2', 'q1']}


def test_explicit_query_export_uses_stable_flat_directory() -> None:
    output = _query_output_directory(
        out_dir=Path('/tmp/query-geometry-export/balanced_reference'),
        query_group='manual',
        query_dir_name='q1',
        flat_query_dirs=True,
    )

    assert output == Path('/tmp/query-geometry-export/balanced_reference/q1')


def test_geometry_plots_rematerialize_missing_chunk_embeddings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    readiness = iter((False, True))
    monkeypatch.setattr(
        plot_stage,
        'chunk_embedding_artifacts_ready',
        lambda _paths: next(readiness),
    )
    embed_calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        plot_stage,
        'run_embed',
        lambda cfg, paths: embed_calls.append((cfg, paths)),
    )

    cfg = cast(ExperimentCfg, object())
    paths = cast(MedicalDatasetGenPaths, object())
    _ensure_query_geometry_chunk_embeddings(cfg, paths)

    assert embed_calls == [(cfg, paths)]
    assert 'rematerializing them via the embed stage' in capsys.readouterr().out


def test_geometry_plots_reuse_valid_chunk_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plot_stage, 'chunk_embedding_artifacts_ready', lambda _paths: True)
    monkeypatch.setattr(
        plot_stage,
        'run_embed',
        lambda _cfg, _paths: pytest.fail('valid chunk embeddings must be reused'),
    )

    _ensure_query_geometry_chunk_embeddings(
        cast(ExperimentCfg, object()),
        cast(MedicalDatasetGenPaths, object()),
    )
