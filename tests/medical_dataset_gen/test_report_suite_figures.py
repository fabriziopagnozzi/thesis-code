from __future__ import annotations

import json
from pathlib import Path

from experiments.medical_dataset_gen.reports.suite_analysis import write_suite_factor_figures


def test_suite_factor_figures_write_both_interactions_as_png_and_pdf(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for comparison, x_factor, x_levels, line_factor, line_levels in (
        (
            'interaction_dominance_background',
            'background_topology',
            ('32x1', '4x8'),
            'dominance_level',
            ('balanced', 'high'),
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

    written = write_suite_factor_figures(output_dir=tmp_path, contrast_rows=rows)

    assert {path.name for path in written} == {
        'dominance_background_interaction_by_objective.png',
        'dominance_background_interaction_by_objective.pdf',
        'sparse_near_miss_interaction_by_objective.png',
        'sparse_near_miss_interaction_by_objective.pdf',
    }
    assert all(path.is_file() for path in written)


def test_suite_factor_figures_can_regenerate_one_stem(tmp_path: Path) -> None:
    rows = [
        {
            'Comparison': 'interaction_sparse_near_miss',
            'RunProfile': 'qwen_unbiased_simple',
            'EmbeddingModel': 'Qwen/Qwen3-Embedding-0.6B',
            'k': 6,
            'Factor_near_miss_mass': level,
            'FactorOrder_near_miss_mass': json.dumps(['24', '96']),
            'Factor_sparse_level': 'control',
            **{
                metric: value
                for metric in (
                    'Delta_FacLoc_MMR_FCP',
                    'Delta_FacLoc_MMR_FacetCoverage',
                    'Delta_FacLoc_MMR_AllFacetCoverageRate',
                    'Delta_FacLoc_MMR_AllFacetCleanRate',
                    'Delta_FacLoc_MMR_Precision',
                    'Delta_FacLoc_MMR_alpha_nDCG',
                )
            },
        }
        for level, value in (('24', 0.1), ('96', 0.2))
    ]

    written = write_suite_factor_figures(
        output_dir=tmp_path,
        contrast_rows=rows,
        stems=('sparse_near_miss_interaction_by_objective',),
    )

    assert {path.name for path in written} == {
        'sparse_near_miss_interaction_by_objective.png',
        'sparse_near_miss_interaction_by_objective.pdf',
    }
