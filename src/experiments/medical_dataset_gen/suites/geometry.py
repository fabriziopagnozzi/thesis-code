"""Validation-frozen geometry strata for the native thesis suite."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

from experiments.medical_dataset_gen.suites.core import (
    _sha256_json,
)
from experiments.medical_dataset_gen.suites.runtime import SuiteRuntime
from experiments.medical_dataset_gen.utils.io_utils import read_parquet

_METRIC = 'in_minus_cross_similarity'


def freeze_separability_strata(*, results_dir: Path, suite_id: str) -> Path:
    """Freeze validation-only tertiles for balanced/unbiased Qwen geometry.

    The artifact records all source hashes.  It intentionally reads geometry
    and query metadata only: neither evaluation results nor selected strategy
    rows participate in threshold selection.
    """
    runtime = SuiteRuntime.load(results_dir=results_dir, suite_id=suite_id)
    root = runtime.root
    path = root / 'geometry' / 'separability_strata.json'
    if path.exists():
        return path
    calibrations = _calibration_frames(runtime)
    if not calibrations:
        raise FileNotFoundError('no completed balanced/unbiased geometry artifacts for calibration')
    payload_conditions: dict[str, object] = {}
    for surface, payload in calibrations.items():
        frame = payload['frame']
        assert isinstance(frame, pl.DataFrame)
        values = sorted(float(value) for value in frame[_METRIC].drop_nulls().to_list())
        if len(values) < 3:
            raise ValueError(f'{surface}: at least three validation evidence profiles are required')
        lower = _tertile(values, 1)
        upper = _tertile(values, 2)
        ids = sorted(str(value) for value in frame['evidence_profile_id'].to_list())
        payload_conditions[surface] = {
            'lower_threshold': lower,
            'upper_threshold': upper,
            'validation_profiles': len(ids),
            'validation_profile_ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
            'source_sha256': payload['source_sha256'],
            'model_signature': payload['model_signature'],
        }
    frozen = {
        'layout_version': 5,
        'kind': 'validation_frozen_separability_strata',
        'metric': _METRIC,
        'calibration': {
            'distribution_id': 'balanced_reference',
            'query_structure': 'unbiased',
            'split': 'validation',
            'deduplication': 'evidence_profile_id',
        },
        'document_surfaces': payload_conditions,
    }
    frozen['sha256'] = _sha256_json(frozen)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + '\n')
    return path


def apply_frozen_separability_strata(
    *, results_dir: Path, suite_id: str
) -> list[dict[str, object]]:
    """Assign test geometry to frozen strata without loading retrieval output."""
    runtime = SuiteRuntime.load(results_dir=results_dir, suite_id=suite_id)
    root = runtime.root
    frozen_path = root / 'geometry' / 'separability_strata.json'
    if not frozen_path.is_file():
        return []
    frozen = json.loads(frozen_path.read_text())
    manifest = runtime.manifest
    _validate_frozen_calibration(runtime=runtime, frozen=frozen)
    surfaces = frozen.get('document_surfaces', {})
    if not isinstance(surfaces, dict):
        raise ValueError(f'{frozen_path}: invalid document_surfaces')
    rows: list[dict[str, object]] = []
    for cell in manifest.cells:
        surface = cell.run_profile_factors.get('document_surface')
        limits = surfaces.get(surface)
        if not isinstance(limits, dict):
            continue
        try:
            cfg = runtime.load_config(cell)
            paths = runtime.paths(cell, cfg)
            geometry = read_parquet(paths, 'geometry_stats')
            queries = read_parquet(paths, 'queries').select(
                'query_id', 'evidence_profile_id', 'split'
            )
        except FileNotFoundError:
            continue
        if _METRIC not in geometry.columns:
            continue
        test = geometry.join(queries, on='query_id', how='inner').filter(pl.col('split') == 'test')
        lower, upper = float(limits['lower_threshold']), float(limits['upper_threshold'])
        for row in test.select('query_id', 'evidence_profile_id', _METRIC).iter_rows(named=True):
            value = float(row[_METRIC])
            rows.append(
                {
                    'Experiment': cell.name,
                    'CellId': cell.cell_id,
                    'Distribution': cell.distribution_id,
                    'RunProfile': cell.run_profile_id,
                    'query_id': row['query_id'],
                    'evidence_profile_id': row['evidence_profile_id'],
                    'SeparationMetric': _METRIC,
                    'SeparationValue': value,
                    'SeparationStratum': 'low'
                    if value <= lower
                    else 'medium'
                    if value <= upper
                    else 'high',
                    'FrozenStrataSha256': frozen.get('sha256'),
                }
            )
    return rows


def _validate_frozen_calibration(*, runtime: SuiteRuntime, frozen: object) -> None:
    """Fail closed when a geometry stratum no longer matches its calibration data."""
    if not isinstance(frozen, dict) or frozen.get('metric') != _METRIC:
        raise ValueError('invalid frozen separability-strata metric')
    expected_surfaces = frozen.get('document_surfaces')
    if not isinstance(expected_surfaces, dict):
        raise ValueError('invalid frozen separability-strata document surfaces')
    current = _calibration_frames(runtime)
    for surface, expected in expected_surfaces.items():
        if not isinstance(expected, dict) or surface not in current:
            raise ValueError(f'{surface}: frozen separability calibration source is unavailable')
        payload = current[surface]
        frame = payload['frame']
        assert isinstance(frame, pl.DataFrame)
        ids = sorted(str(value) for value in frame['evidence_profile_id'].to_list())
        ids_hash = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
        if expected.get('validation_profile_ids_sha256') != ids_hash:
            raise ValueError(f'{surface}: frozen separability validation IDs are stale')
        if expected.get('source_sha256') != payload['source_sha256']:
            raise ValueError(f'{surface}: frozen separability source hashes are stale')
        if expected.get('model_signature') != payload['model_signature']:
            raise ValueError(f'{surface}: frozen separability model signature is stale')


def _calibration_frames(runtime: SuiteRuntime) -> dict[str, dict[str, object]]:
    by_surface: dict[str, list[pl.DataFrame]] = {}
    source_hashes: dict[str, list[str]] = {}
    model_signatures: dict[str, set[str]] = {}
    for cell in runtime.manifest.cells:
        if cell.distribution_id != 'balanced_reference':
            continue
        if cell.run_profile_factors.get('query_structure') != 'unbiased':
            continue
        surface = cell.run_profile_factors.get('document_surface')
        if surface not in {'category-explicit', 'category-implicit'}:
            continue
        cfg = runtime.load_config(cell)
        paths = runtime.paths(cell, cfg)
        geometry_path, queries_path = (
            paths.table_path('geometry_stats'),
            paths.table_path('queries'),
        )
        if not geometry_path.is_file() or not queries_path.is_file():
            continue
        geometry = pl.read_parquet(geometry_path)
        if _METRIC not in geometry.columns:
            continue
        queries = pl.read_parquet(
            queries_path, columns=['query_id', 'evidence_profile_id', 'split']
        )
        validation = (
            geometry.select('query_id', _METRIC)
            .join(queries, on='query_id', how='inner')
            .filter(pl.col('split') == 'validation')
            .group_by('evidence_profile_id')
            .agg(pl.col(_METRIC).mean().alias(_METRIC))
        )
        if validation.is_empty():
            continue
        by_surface.setdefault(surface, []).append(validation)
        source_hashes.setdefault(surface, []).extend(
            [_sha256_file(geometry_path), _sha256_file(queries_path)]
        )
        model_signatures.setdefault(surface, set()).add(
            _sha256_json({'model': cfg.embeddings.model_name, 'config': cell.run_profile_sha256})
        )
    result: dict[str, dict[str, object]] = {}
    for surface, frames in by_surface.items():
        # Same profiles can be visible through multiple calibration cells only
        # if the suite later expands it; keep one value per evidence profile.
        frame = (
            pl.concat(frames)
            .group_by('evidence_profile_id')
            .agg(pl.col(_METRIC).mean().alias(_METRIC))
        )
        result[surface] = {
            'frame': frame,
            'source_sha256': hashlib.sha256(
                '\n'.join(sorted(source_hashes[surface])).encode()
            ).hexdigest(),
            'model_signature': sorted(model_signatures[surface]),
        }
    return result


def _tertile(values: list[float], index: int) -> float:
    position = (len(values) - 1) * index / 3
    low, high = int(position), min(int(position) + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
