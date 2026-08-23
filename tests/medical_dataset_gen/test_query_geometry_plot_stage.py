from pathlib import Path

import pytest

from experiments.medical_dataset_gen.query_geometry.plot_stage import (
    _explicit_query_groups,
    _query_output_directory,
    parse_geom_plots_cli_args,
)


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
