from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from experiments.medical_dataset_gen.analysis.analysis_constants import (
    DeltaMetricLabel,
    ExperimentFamilyId,
)
from experiments.medical_dataset_gen.schemas.global_config_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths

type PlotFormat = Literal['png', 'pdf', 'svg']
type BudgetCategory = Literal['low_budget', 'medium_budget', 'high_budget']


@dataclass(frozen=True)
class DeltaMetricPlotSpec:
    metric_label: DeltaMetricLabel
    title_label: str
    filename_token: str
    source_metric_name: str


@runtime_checkable
class ScalarItem(Protocol):
    def item(self) -> object: ...


@dataclass(frozen=True)
class CliArgs:
    results_dir: Path
    output_dir: Path
    include_scrapped: bool
    experiments: tuple[str, ...]
    max_table_rows: int
    tablefmt: str
    plots: bool
    plot_format: PlotFormat
    near_optimal_epsilon: float
    bootstrap_replicates: int
    bootstrap_seed: int


@dataclass(frozen=True)
class ExperimentRecord:
    name: str
    experiment_dir: Path
    distribution_id: str
    run_label: str
    is_subexperiment: bool
    cfg: ExperimentCfg | None
    paths: MedicalDatasetGenPaths
    config_error: str | None
    family_id: ExperimentFamilyId
    family_label: str

    @property
    def embedding_model(self) -> str:
        if self.cfg is None:
            return 'unknown'
        return str(self.cfg.embeddings.model_name)

    @property
    def only_pass_geometry(self) -> bool | None:
        if self.cfg is None:
            return None
        return bool(self.cfg.retrieval.only_pass_geometry)


@dataclass(frozen=True)
class ReportOutputs:
    output_dir: Path
    experiments_discovered: int
    experiments_loaded: int
    warnings_count: int
    figures_count: int
