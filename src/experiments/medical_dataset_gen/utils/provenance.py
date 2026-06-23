"""Lineage-aware artifact manifests for reproducible pipeline runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict, cast
from uuid import uuid4

from experiments.medical_dataset_gen.global_config import (
    SYNTH_MEDICAL_DATASET_TABLE_NAMES,
    ExperimentCfg,
    MedicalDatasetGenPaths,
    SyntheticMedicalDatasetTableName,
)


class ArtifactEntry(TypedDict):
    stage: str
    fingerprint: str
    row_count: int | None
    config_hash: str
    input_fingerprints: dict[str, str]
    created_at: str


STAGE_INPUTS: dict[str, tuple[str, ...]] = {
    'plans': (),
    'calibrate_plans': ('query_plans.parquet',),
    'facts': ('query_plans.parquet', 'query_plan_calibration.parquet'),
    'chunks': ('clinical_facts.parquet',),
    'queries_answers': ('query_plans.parquet', 'clinical_facts.parquet'),
    'qrels': ('chunk_memberships.parquet',),
    'embed': ('chunk_documents.parquet', 'queries.parquet'),
    'filter_queries': (
        'chunk_documents.parquet',
        'chunk_memberships.parquet',
        'queries.parquet',
        'qrels.parquet',
        'query_plan_calibration.parquet',
        'embeddings_chunk_vectors.npy',
        'embeddings_query_vectors.npy',
        'embeddings_chunk_ids.npy',
        'embeddings_query_ids.npy',
        'embeddings_metadata.json',
    ),
    'eval': (
        'chunk_documents.parquet',
        'chunk_memberships.parquet',
        'queries.parquet',
        'gold_answers.parquet',
        'qrels.parquet',
        'geometry_stats.parquet',
        'embeddings_chunk_vectors.npy',
        'embeddings_query_vectors.npy',
        'embeddings_chunk_ids.npy',
        'embeddings_query_ids.npy',
        'embeddings_metadata.json',
    ),
    'geom_plots': (
        'chunk_documents.parquet',
        'chunk_memberships.parquet',
        'queries.parquet',
        'qrels.parquet',
        'geometry_stats.parquet',
        'evaluation_results.parquet',
        'embeddings_chunk_vectors.npy',
        'embeddings_query_vectors.npy',
        'embeddings_chunk_ids.npy',
        'embeddings_query_ids.npy',
        'embeddings_metadata.json',
    ),
    'eval_plots': (
        'evaluation_results.parquet',
        'evaluation_stats.parquet',
        'lambda_pair_agreement.parquet',
    ),
}

STAGE_OUTPUTS: dict[str, tuple[str, ...]] = {
    'plans': ('query_plans.parquet',),
    'calibrate_plans': ('query_plans.parquet', 'query_plan_calibration.parquet'),
    'facts': ('clinical_facts.parquet',),
    'chunks': (
        'chunk_documents.parquet',
        'chunk_memberships.parquet',
        'generation_rejects.parquet',
    ),
    'queries_answers': ('queries.parquet', 'gold_answers.parquet'),
    'qrels': ('qrels.parquet',),
    'embed': (
        'embeddings_chunk_vectors.npy',
        'embeddings_query_vectors.npy',
        'embeddings_chunk_ids.npy',
        'embeddings_query_ids.npy',
        'embeddings_metadata.json',
    ),
    'filter_queries': (
        'geometry_stats.parquet',
        'geometry_slice_stats.parquet',
        'geometry_summary.json',
    ),
    'eval': (
        'evaluation_results.parquet',
        'evaluation_stats.parquet',
        'evaluation_slice_stats.parquet',
        'lambda_pair_agreement.parquet',
    ),
    'geom_plots': ('query_geometry_points.parquet', 'query_geometry_stats.parquet'),
    'eval_plots': (),
}


@dataclass
class PipelineProvenance:
    cfg: ExperimentCfg
    paths: MedicalDatasetGenPaths
    stages: list[str]

    def __post_init__(self) -> None:
        self.run_id = datetime.now(UTC).strftime('%Y%m%dT%H%M%S') + f'_{uuid4().hex[:8]}'
        raw_config_hash = (
            _fingerprint(self.paths.config_path)
            if self.paths.config_path.exists()
            else _hash_json(self.cfg.model_dump(mode='json', by_alias=True))
        )
        self.template_hash = _directory_fingerprint(self.paths.root / 'data_templates')
        self.config_hash = _hash_json(
            {
                'config_hash': raw_config_hash,
                'template_hash': self.template_hash,
                'dataset_schema_version': self.cfg.dataset_schema_version,
            }
        )
        self.manifest_path = self.paths.experiment_dir / '_artifact_manifest.json'
        self.run_path = self.paths.experiment_dir / '_runs' / f'{self.run_id}.json'
        self.run_path.parent.mkdir(parents=True, exist_ok=True)
        commit, dirty = _git_state()
        self.run_record: dict[str, object] = {
            'run_id': self.run_id,
            'started_at': datetime.now(UTC).isoformat(),
            'dataset_schema_version': self.cfg.dataset_schema_version,
            'experiment': self.paths.exp_name,
            'command': list(sys.argv),
            'config_hash': self.config_hash,
            'template_hash': self.template_hash,
            'git_commit': commit,
            'git_dirty': dirty,
            'stages': self.stages,
            'stage_records': [],
        }
        self._write_run_record()

    def before_stage(self, stage: str) -> dict[str, str]:
        inputs = self._paths(STAGE_INPUTS[stage])
        for path in inputs:
            if not path.exists():
                raise FileNotFoundError(f'missing pipeline input for {stage!r}: {path}')
        return {str(path): _fingerprint(path) for path in inputs}

    def after_stage(self, stage: str, input_fingerprints: dict[str, str]) -> None:
        manifest = self._load_manifest()
        artifacts = manifest.setdefault('artifacts', {})
        output_records: list[dict[str, object]] = []
        output_paths = self._paths(STAGE_OUTPUTS[stage])
        output_fingerprints = {
            str(path.resolve()): _fingerprint(path) for path in output_paths if path.exists()
        }
        for path in output_paths:
            if not path.exists():
                continue
            lineage = {
                input_path: fingerprint
                for input_path, fingerprint in input_fingerprints.items()
                if Path(input_path).resolve() != path.resolve()
            }
            if stage == 'calibrate_plans' and path.name == 'query_plan_calibration.parquet':
                plans_path = self.paths.experiment_dir / 'query_plans.parquet'
                lineage = {
                    str(plans_path.resolve()): output_fingerprints[str(plans_path.resolve())]
                }
            entry: ArtifactEntry = {
                'stage': stage,
                'fingerprint': _fingerprint(path),
                'row_count': _parquet_row_count(path),
                'config_hash': self.config_hash,
                'input_fingerprints': lineage,
                'created_at': datetime.now(UTC).isoformat(),
            }
            artifacts[str(path.resolve())] = entry  # type: ignore
            output_records.append({'path': str(path), **entry})
        manifest.update(
            {
                'dataset_schema_version': self.cfg.dataset_schema_version,
                'experiment': self.paths.exp_name,
                'config_hash': self.config_hash,
            }
        )
        _write_json(self.manifest_path, manifest)
        stage_records = self.run_record['stage_records']
        assert isinstance(stage_records, list)
        stage_records.append(
            {
                'stage': stage,
                'inputs': input_fingerprints,
                'outputs': output_records,
                'finished_at': datetime.now(UTC).isoformat(),
            }
        )
        self._write_run_record()

    def finish(self) -> None:
        self.run_record['finished_at'] = datetime.now(UTC).isoformat()
        self._write_run_record()

    def _paths(self, names: tuple[str, ...]) -> list[Path]:
        paths: list[Path] = []
        for name in names:
            relative = Path(name)
            if relative.suffix == '.parquet' and relative.stem in SYNTH_MEDICAL_DATASET_TABLE_NAMES:
                table = cast(SyntheticMedicalDatasetTableName, relative.stem)
                paths.append(self.paths.table_path(table))
            else:
                paths.append(self.paths.experiment_dir / relative)
        return paths

    def _load_manifest(self) -> dict[str, object]:
        if not self.manifest_path.exists():
            return {'artifacts': {}}
        with open(self.manifest_path) as file:
            return json.load(file)

    def _write_run_record(self) -> None:
        _write_json(self.run_path, self.run_record)


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _hash_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _directory_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(path.glob('*.yaml')):
        digest.update(file_path.name.encode())
        digest.update(_fingerprint(file_path).encode())
    return digest.hexdigest()


def _parquet_row_count(path: Path) -> int | None:
    if path.suffix != '.parquet':
        return None
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], capture_output=True, check=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ['git', 'status', '--porcelain'], capture_output=True, check=True, text=True
            ).stdout.strip()
        )
        return commit, dirty
    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None, None


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as file:
        json.dump(payload, file, indent=2, sort_keys=True)
