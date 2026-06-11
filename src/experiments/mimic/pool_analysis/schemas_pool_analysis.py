from pathlib import Path

import yaml
from pydantic import BaseModel, PositiveFloat, PositiveInt

from experiments.mimic.global_configs import MimicPaths


class PoolAnalysisCfg(BaseModel):
    model_config = {'arbitrary_types_allowed': True}

    pool_n: PositiveInt = 2000

    umap_n_neighbors: PositiveInt = 30
    umap_min_dist: float = 0.05
    umap_metric: str = 'cosine'
    umap_dim_for_cluster: PositiveInt = 10
    umap_random_state: PositiveInt = 42

    hdbscan_min_cluster_size: PositiveInt = 30
    hdbscan_min_samples: PositiveInt | None = None

    lof_n_neighbors: PositiveInt = 20
    lof_contamination: PositiveFloat | str = 'auto'

    cv_n_splits: PositiveInt = 5
    n_figures: PositiveInt = 30

    commit_every: PositiveInt = 10
    limit: PositiveInt | None = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> PoolAnalysisCfg:
        cfg_path = Path(path) if path else MimicPaths.config_path
        with open(cfg_path) as f:
            data = yaml.safe_load(f) or {}
        block = data.get('pool_analysis') or {}
        return cls(**block)
