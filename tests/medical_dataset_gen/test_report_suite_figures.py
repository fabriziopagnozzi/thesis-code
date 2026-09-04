from __future__ import annotations

import json
from pathlib import Path

from experiments.medical_dataset_gen.reports.suite_analysis import (
    _suite_line_label,
    write_suite_factor_figures,
)


def _interaction_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for comparison, x_factor, x_levels, line_factor, line_levels in (
        (
            'interaction_dominance_background',
            'background_topology',
            ('32x1', '4x8'),
            'dominance_level',
            ('control', 'high'),
        ),
        (
            'interaction_sparse_near_miss',
            'near_miss_mass',
            ('24', '96'),
            'sparse_level',
            ('control', 'severe'),
        ),
    ):
        for x_index, x_level in enumerate(x_levels):
            for line_index, line_level in enumerate(line_levels):
                row: dict[str, object] = {
                    'Comparison': comparison,
                    'RunProfile': 'qwen_unbiased_simple',
                    'EmbeddingModel': 'Qwen/Qwen3-Embedding-0.6B',
                    'k': 6,
                    f'Factor_{x_factor}': x_level,
                    f'FactorOrder_{x_factor}': json.dumps(list(x_levels)),
                    f'Factor_{line_factor}': line_level,
                }
                for metric in (
                    'Delta_FacLoc_MMR_FCP',
                    'Delta_FacLoc_MMR_FacetCoverage',
                    'Delta_FacLoc_MMR_AllFacetCoverageRate',
                    'Delta_FacLoc_MMR_AllFacetCleanRate',
                    'Delta_FacLoc_MMR_Precision',
                    'Delta_FacLoc_MMR_alpha_nDCG',
                ):
                    row[metric] = 0.1 * (x_index + line_index)
                rows.append(row)
    return rows


def test_suite_factor_figures_write_one_combined_interaction_as_png_and_pdf(
    tmp_path: Path,
) -> None:
    written = write_suite_factor_figures(output_dir=tmp_path, contrast_rows=_interaction_rows())

    assert {path.name for path in written} == {
        'stressor_interactions_by_objective.png',
        'stressor_interactions_by_objective.pdf',
    }
    assert all(path.is_file() for path in written)


def test_suite_factor_figures_can_regenerate_one_stem(tmp_path: Path) -> None:
    written = write_suite_factor_figures(
        output_dir=tmp_path,
        contrast_rows=_interaction_rows(),
        stems=('stressor_interactions_by_objective',),
    )

    assert {path.name for path in written} == {
        'stressor_interactions_by_objective.png',
        'stressor_interactions_by_objective.pdf',
    }


def test_suite_interaction_legends_describe_gold_support() -> None:
    assert (
        _suite_line_label(
            row={'Factor_dominance_level': 'control'}, line_key='dominance_level'
        )
        == 'Balanced support'
    )
    assert (
        _suite_line_label(row={'Factor_dominance_level': 'high'}, line_key='dominance_level')
        == 'High dominance'
    )
    assert (
        _suite_line_label(row={'Factor_sparse_level': 'control'}, line_key='sparse_level')
        == 'Balanced support'
    )
    assert (
        _suite_line_label(row={'Factor_sparse_level': 'severe'}, line_key='sparse_level')
        == 'One severe sparse facet'
    )
