from experiments.medical_dataset_gen.query_geometry.filter_diagnostics import (
    strict_gate_failures,
)
from experiments.medical_dataset_gen.utils.global_schemas import GeometryFilterCfg


def test_strict_geometry_gate_retains_each_declared_failure() -> None:
    cfg = GeometryFilterCfg()

    missing_facet = strict_gate_failures(
        cfg,
        n_facets_present=3,
        n_facets=4,
        primary_axis_fraction=0.50,
        n_topk_retrieved_facets=2,
    )
    weak_primary_axis = strict_gate_failures(
        cfg,
        n_facets_present=4,
        n_facets=4,
        primary_axis_fraction=0.49,
        n_topk_retrieved_facets=2,
    )
    excess_coverage = strict_gate_failures(
        cfg,
        n_facets_present=4,
        n_facets=4,
        primary_axis_fraction=0.50,
        n_topk_retrieved_facets=3,
    )

    assert missing_facet == {
        'fail_missing_facet': True,
        'fail_weak_primary_axis_dominance': False,
        'fail_excess_stress_horizon_facet_coverage': False,
    }
    assert weak_primary_axis == {
        'fail_missing_facet': False,
        'fail_weak_primary_axis_dominance': True,
        'fail_excess_stress_horizon_facet_coverage': False,
    }
    assert excess_coverage == {
        'fail_missing_facet': False,
        'fail_weak_primary_axis_dominance': False,
        'fail_excess_stress_horizon_facet_coverage': True,
    }


def test_strict_geometry_gate_is_independent_of_background_metadata() -> None:
    failures = strict_gate_failures(
        GeometryFilterCfg(),
        n_facets_present=4,
        n_facets=4,
        primary_axis_fraction=0.50,
        n_topk_retrieved_facets=2,
    )

    assert not any(failures.values())
    assert all('background' not in name for name in failures)
