from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from experiments.medical_dataset_gen.reports.analysis_constants import (
    DeltaMetricLabel,
    ExperimentFamilyId,
)
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths

type PlotFormat = Literal['png', 'pdf', 'svg']
type BudgetCategory = Literal['low_budget', 'medium_budget', 'high_budget']
type MainQueryScope = Literal['all', 'geometry_eligible']
type RefreshMode = Literal['plots', 'latex_macros']


@dataclass(frozen=True)
class DeltaMetricPlotSpec:
    metric_label: DeltaMetricLabel
    title_label: str
    filename_token: str
    source_metric_name: str
    higher_is_better: bool = True


@runtime_checkable
class ScalarItem(Protocol):
    def item(self) -> object: ...


@dataclass(frozen=True)
class CliArgs:
    results_dir: Path
    output_dir: Path
    include_scrapped: bool
    experiments: tuple[str, ...]
    experiment_regex: str | None
    exclude_experiment_regex: str | None
    embedding_models: tuple[str, ...]
    artifact_version: str | None
    max_table_rows: int
    tablefmt: str
    plots: bool
    plot_format: PlotFormat
    near_optimal_epsilon: float
    cross_query_chunk_modes: bool = False
    refresh_report_dir: Path | None = None
    refresh_mode: RefreshMode | None = None
    bootstrap_replicates: int = 1000
    bootstrap_seed: int = 20260712
    main_query_scope: MainQueryScope = 'all'
    lambda_analysis: bool = False
    global_lambda_analysis: bool = False
    lodo_analysis: bool = False
    paired_statistics: bool = False
    validity_analysis: bool = False
    full_report: bool = False
    suite_id: str | None = None
    suite_base_id: str | None = None
    suite_regex: str | None = None
    suite_where: str | None = None
    strict_suite: bool = False

    @property
    def run_lambda_analysis(self) -> bool:
        return self.lambda_analysis or self.full_report

    @property
    def run_global_lambda_analysis(self) -> bool:
        return self.global_lambda_analysis or self.full_report

    @property
    def run_lodo_analysis(self) -> bool:
        return self.lodo_analysis or self.full_report

    @property
    def run_paired_statistics(self) -> bool:
        return self.paired_statistics or self.full_report

    @property
    def run_validity_analysis(self) -> bool:
        return self.validity_analysis or self.full_report


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
    origin: str = 'legacy'
    dataset_schema_version: int | None = None
    evaluation_schema_version: int | None = None
    include_in_causal_summaries: bool = True
    include_in_family_summary: bool = True
    factors: dict[str, object] | None = None
    distribution_base_id: str | None = None
    tags: tuple[str, ...] = ()
    analysis_blocks: tuple[str, ...] = ()
    analysis_tier: str | None = None
    run_profile_factors: dict[str, str] | None = None

    @property
    def embedding_model(self) -> str:
        if self.cfg is None:
            return 'unknown'
        return str(self.cfg.embeddings.model_name)

    @property
    def only_pass_geometry(self) -> bool | None:
        # Retained for report-column compatibility with older artifacts. The
        # current evaluation contract always scores all queries and stores the
        # geometry pass flag per query.
        return None


@dataclass(frozen=True)
class ReportOutputs:
    output_dir: Path
    experiments_discovered: int
    experiments_loaded: int
    warnings_count: int
    figures_count: int
