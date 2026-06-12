import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import polars as pl
import yaml
from pydantic import BaseModel, Field

from helpers.dir_paths import ROOT_DIR

type TableName = Literal[
    'query_plans',
    'clinical_facts',
    'chunks',
    'queries',
    'gold_answers',
    'qrels',
    'generation_rejects',
    'embeddings',
    'geometry_stats',
    'evaluation_results',
    'evaluation_stats',
    'embedding_geometry_points',
    'embedding_geometry_query_stats',
]


class GlobalCfg(BaseModel):
    seed: int = 42
    n_queries: int = 120
    conditions: int = 4
    output_experiment: str = 'mvp'


class GenerationCfg(BaseModel):
    ontology_path: str | None = None
    query_types: list[str] = Field(default_factory=lambda: ['subgroup_comparison'])
    gold_chunks_dominant: int = 18
    gold_chunks_complementary: int = 8
    distractors_per_query: int = 30
    chunk_min_words: int = 25
    chunk_max_words: int = 90
    chunk_word_tolerance: int = 2
    llm_name: str = 'gemma4-31b-text'
    llm_workers: int = 1
    use_llm_chunk_generation: bool = True
    use_llm_chunk_rewriting: bool = False
    use_llm_query_paraphrase: bool = False
    llm_chunk_max_attempts: int = 3
    llm_temperature: float = 0.1
    llm_num_ctx: int = 4096


class EmbeddingCfg(BaseModel):
    backend: Literal['tfidf', 'sentence_transformers'] = 'tfidf'
    model_name: str = 'multi-qa-mpnet-base-cos-v1'
    batch_size: int = 64
    device: Literal['cpu', 'cuda'] = 'cpu'
    query_prompt: str | None = None
    normalize: bool = True
    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = 2


class RetrievalCfg(BaseModel):
    pool_scope: Literal['query_local', 'same_condition', 'full_corpus'] = 'query_local'
    candidate_pool_n: int = 300
    k_values: list[int] = Field(default_factory=lambda: [5, 10, 20])
    lambda_values: list[float] = Field(default_factory=lambda: [0.3, 0.5, 0.7])
    strategies: list[Literal['top_k', 'mmr', 'fac_loc']] = Field(
        default_factory=lambda: ['top_k', 'mmr', 'fac_loc']
    )
    mmr_window: int | None = None
    only_pass_geometry: bool = True


class GeometryCfg(BaseModel):
    topk_dominance_k: int = 10
    min_topk_dominant_count: int = 5
    min_in_minus_cross_similarity: float = 0.03
    min_distractors_in_pool: int = 10


class EmbeddingGeometryCfg(BaseModel):
    n_queries: int = 6
    query_ids: list[str] = Field(default_factory=list)
    candidate_pool_n: int | None = None
    plot_k: int = 10
    reduction: Literal['umap', 'pca'] = 'umap'
    pca_dims: int | None = None
    umap_metric: Literal['cosine', 'euclidean'] = 'cosine'
    umap_neighbors: int = 15
    umap_min_dist: float = 0.08
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int | None = None
    random_state: int = 42


class ExperimentCfg(BaseModel):
    global_: GlobalCfg = Field(alias='global')
    generation: GenerationCfg = Field(default_factory=GenerationCfg)
    embeddings: EmbeddingCfg = Field(default_factory=EmbeddingCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    geometry: GeometryCfg = Field(default_factory=GeometryCfg)
    embedding_geometry: EmbeddingGeometryCfg = Field(default_factory=EmbeddingGeometryCfg)

    model_config = {'populate_by_name': True, 'extra': 'ignore'}


class MedicalDatasetGenPaths:
    root = ROOT_DIR / 'src' / 'experiments' / 'medical_dataset_gen'
    results_dir = root / '_results'
    default_ontology_path = root / 'ontology.yaml'

    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.experiment_dir = self.results_dir / exp_name
        self.logs_dir = self.experiment_dir / '_logs'
        self.figures_dir = self.experiment_dir / '_figures'
        self.config_path = self.experiment_dir / '_config.yaml'
        self.embeddings_npz_path = self.experiment_dir / 'embeddings.npz'
        self.embeddings_meta_path = self.experiment_dir / 'embeddings_metadata.json'

    def ensure_dirs(self) -> None:
        for path in [self.results_dir, self.experiment_dir, self.logs_dir, self.figures_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def table_path(
        self,
        table: TableName,
        ext: Literal['parquet', 'json', 'jsonl', 'csv'] = 'parquet',
    ) -> Path:
        return self.experiment_dir / f'{table}.{ext}'


def load_config(exp: str | None = None) -> ExperimentCfg:
    exp_name = exp or os.getenv('EXP') or os.getenv('EXP_NAME')
    if exp_name is None:
        raise ValueError(
            'missing experiment name; pass --exp or set EXP/EXP_NAME so '
            'the config can be loaded from _results/<exp>/_config.yaml'
        )

    cfg_path = MedicalDatasetGenPaths(exp_name).config_path
    if not cfg_path.exists():
        raise FileNotFoundError(
            f'missing experiment config: {cfg_path}. '
            'Create it manually before running the pipeline; '
            'src/experiments/medical_dataset_gen/_config.yaml is no longer used.'
        )
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    cfg = ExperimentCfg.model_validate(raw)
    cfg.global_.output_experiment = exp_name
    return cfg


def load_config_from_cli() -> ExperimentCfg:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--exp', type=str, default=os.getenv('EXP') or os.getenv('EXP_NAME'))
    parser.add_argument('--max-queries', type=int, default=None)
    parser.add_argument(
        '--embedding-backend', choices=['tfidf', 'sentence_transformers'], default=None
    )
    parser.add_argument('--embedding-model', type=str, default=None)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--llm-name', type=str, default=None)
    parser.add_argument('--llm-workers', type=int, default=None)
    parser.add_argument(
        '--pool-scope',
        choices=['query_local', 'same_condition', 'full_corpus'],
        default=None,
    )
    parser.add_argument('--embedding-geometry-queries', type=int, default=None)
    parser.add_argument('--embedding-geometry-k', type=int, default=None)
    parser.add_argument('--embedding-geometry-reduction', choices=['umap', 'pca'], default=None)
    llm_chunk_group = parser.add_mutually_exclusive_group()
    llm_chunk_group.add_argument('--llm-chunks', dest='llm_chunks', action='store_true')
    llm_chunk_group.add_argument('--no-llm-chunks', dest='llm_chunks', action='store_false')
    parser.set_defaults(llm_chunks=None)
    llm_chunk_rewrite_group = parser.add_mutually_exclusive_group()
    llm_chunk_rewrite_group.add_argument(
        '--llm-chunk-rewrite', dest='llm_chunk_rewrite', action='store_true'
    )
    llm_chunk_rewrite_group.add_argument(
        '--no-llm-chunk-rewrite', dest='llm_chunk_rewrite', action='store_false'
    )
    parser.set_defaults(llm_chunk_rewrite=None)
    args, unknown = parser.parse_known_args()
    if any(token == '--config' or token.startswith('--config=') for token in unknown):
        raise ValueError(
            '--config is no longer supported for medical_dataset_gen; '
            'place _config.yaml in the target experiment directory instead'
        )

    cfg = load_config(exp=args.exp)
    if args.max_queries is not None:
        cfg.global_.n_queries = args.max_queries
    if args.embedding_backend is not None:
        cfg.embeddings.backend = args.embedding_backend
    if args.embedding_model is not None:
        cfg.embeddings.model_name = args.embedding_model
    if args.device is not None:
        cfg.embeddings.device = args.device
    if args.batch_size is not None:
        cfg.embeddings.batch_size = args.batch_size
    if args.llm_name is not None:
        cfg.generation.llm_name = args.llm_name
    if args.llm_workers is not None:
        cfg.generation.llm_workers = args.llm_workers
    if args.pool_scope is not None:
        cfg.retrieval.pool_scope = args.pool_scope
    if args.embedding_geometry_queries is not None:
        cfg.embedding_geometry.n_queries = args.embedding_geometry_queries
    if args.embedding_geometry_k is not None:
        cfg.embedding_geometry.plot_k = args.embedding_geometry_k
    if args.embedding_geometry_reduction is not None:
        cfg.embedding_geometry.reduction = args.embedding_geometry_reduction
    if args.llm_chunks is not None:
        cfg.generation.use_llm_chunk_generation = args.llm_chunks
    if args.llm_chunk_rewrite is not None:
        cfg.generation.use_llm_chunk_rewriting = args.llm_chunk_rewrite
    return cfg


def paths_for(cfg: ExperimentCfg) -> MedicalDatasetGenPaths:
    paths = MedicalDatasetGenPaths(cfg.global_.output_experiment)
    paths.ensure_dirs()
    return paths


def dump_effective_config(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> None:
    # Intentionally do nothing: experiment configs are user-managed files stored
    # in _results/<exp>/_config.yaml and must never be overwritten here.
    paths.ensure_dirs()


def read_parquet(paths: MedicalDatasetGenPaths, table: TableName) -> pl.DataFrame:
    return pl.read_parquet(paths.table_path(table))


def write_parquet(paths: MedicalDatasetGenPaths, table: TableName, df: pl.DataFrame) -> Path:
    path = paths.table_path(table)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    print(f'[write] {table}: {len(df):,} rows -> {path}')
    return path


def write_json(paths: MedicalDatasetGenPaths, name: str, payload: Any) -> Path:
    path = paths.experiment_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'[write] {path}')
    return path


def setup_logging(paths: MedicalDatasetGenPaths) -> None:
    main = sys.modules['__main__']
    script_name = Path(main.__file__ if main.__file__ else f'unknown_script_{uuid4()}').stem
    log_path = paths.logs_dir / f'{script_name}.log'

    class _Tee(io.TextIOBase):
        def __init__(self, filepath: Path):
            self._terminal = sys.stdout
            self._file = open(filepath, 'a')  # noqa: SIM115

        def write(self, msg: str) -> int:
            self._terminal.write(msg)
            self._file.write(msg)
            return len(msg)

        def flush(self) -> None:
            self._terminal.flush()
            self._file.flush()

    sys.stdout = _Tee(log_path)


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def json_loads(value: str | None, default: Any = None) -> Any:
    if value is None or value == '':
        return default
    return json.loads(value)
