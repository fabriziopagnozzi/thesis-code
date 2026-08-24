"""Remove temporary chunk-embedding arrays after downstream stages finish."""

from __future__ import annotations

from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths

_GIB = 1024**3


def run_cleanup(_cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> None:
    """Unconditionally remove the resolved chunk vectors and IDs.

    Chunk arrays can be shared by multiple suite cells. Cleanup is deliberately
    explicit and caller-controlled, so selecting this stage means those shared
    paths may be unlinked immediately. Query arrays and embedding metadata stay
    available for reports and later inspection.
    """
    artifact_paths = (
        paths.embeddings_paths('chunk_vectors'),
        paths.embeddings_paths('chunk_ids'),
    )
    existing_paths = tuple(path for path in artifact_paths if path.is_file())
    logical_bytes = sum(path.stat().st_size for path in existing_paths)

    for path in artifact_paths:
        path.unlink(missing_ok=True)

    if not existing_paths:
        print('[cleanup] chunk vectors and IDs are already absent')
        return

    removed = ', '.join(str(path) for path in existing_paths)
    print(
        f'[cleanup] removed {len(existing_paths)} chunk embedding artifact(s) '
        f'({logical_bytes / _GIB:.2f} GiB logical): {removed}'
    )
