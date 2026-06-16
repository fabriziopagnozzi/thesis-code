# Configuration And Artifact Reference

This note explains the configuration model and the files written by the implemented pipeline. The source of truth is [global_configs.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/global_configs.py). The example config is [config_example.yaml](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/config_example.yaml), but actual runs read the per-experiment file under `_results/<exp>/_config.yaml`.

## Config Loading

`load_config_from_cli()` first parses common CLI overrides, then calls `load_config(exp)`. `load_config(exp)` reads:

```text
src/experiments/medical_dataset_gen/_results/<exp>/_config.yaml
```

The old `--config` flag is explicitly rejected. This is intentional: each experiment directory owns its config. `dump_effective_config()` only ensures directories exist; it does not rewrite `_config.yaml`.

The supported common CLI overrides are:

- `--exp`
- `--max-queries`
- `--embedding-model`
- `--device`
- `--batch-size`
- `--llm-name`
- `--llm-workers`
- `--pool-scope`
- `--embedding-geometry-queries`
- `--embedding-geometry-k`
- `--embedding-geometry-reduction`
- `--llm-chunks` / `--no-llm-chunks`
- `--llm-chunk-rewrite` / `--no-llm-chunk-rewrite`

`run_pipeline.py` also parses stage controls:

- `--start-at`
- `--stop-after`
- `--no-log-tee`

The valid stage names are exactly the names in `STAGES`: `plans`, `facts`, `chunks`, `queries_answers`, `qrels`, `embed`, `geom_filter`, `eval`, `geom_plots`, and `eval_plots`.

## `global`

`GlobalCfg` controls experiment size and deterministic seeding:

| field | type | default | meaning |
| --- | --- | --- | --- |
| `seed` | positive int | `42` | Base random seed for deterministic query and generation decisions. |
| `n_queries` | positive int | `120` | Maximum number of query plans to materialize. |
| `conditions` | positive int | `4` | Number of ontology conditions to use, selected from `ontology.yaml` in file order. |
| `output_experiment` | string | `mvp` | Experiment directory name; overwritten by `--exp` during load. |

The seed does not make every stage use a single global RNG stream. Query plans get their own `plan_seed`, facts use `Random(plan.plan_seed)`, and chunk reuse keys derive stable SHA-256 seeds. Deterministic chunk text is rendered from the reuse key so the same reusable chunk has one canonical text across queries.

## `generation`

`GenerationCfg` controls ontology use, dataset shape, chunk text generation, LLM behavior, and query paraphrasing:

| field | default | implementation role |
| --- | --- | --- |
| `ontology_path` | `null` | Optional custom YAML path; otherwise [ontology.yaml](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/ontology.yaml). |
| `query_types` | `['subgroup_comparison']` | Query templates instantiated for every condition/subgroup pair. Current code supports `subgroup_comparison` and `outcome_synthesis`. |
| `gold_chunks_dominant` | `25` | Target gold fact count for the dominant facet of each query. |
| `gold_chunks_complementary` | `14` | Target gold fact count for each non-dominant facet. |
| `distractors_per_query` | `30` | Number of near-miss hard negative facts generated per query. |
| `background_outlier_clusters_per_query` | `1` | Number of coherent non-gold clinical islands added to each query-local pool. |
| `background_outlier_cluster_size` | `12` | Number of chunks in each background outlier cluster. |
| `chunk_min_words` | `25` | Minimum accepted chunk length. |
| `chunk_max_words` | `90` | Maximum accepted chunk length. |
| `chunk_word_tolerance` | `2` | Tolerance applied to the min and max word-count checks. |
| `llm_name` | `gemma4-31b-text` | Ollama model passed to `helpers.ollama_client.generate()`. |
| `llm_workers` | `1` | Number of parallel LLM worker threads for grouped generation or rewrite. |
| `use_llm_chunk_generation` | `true` | If true, chunks are generated directly by the LLM. |
| `use_llm_chunk_rewriting` | `false` | If true and direct LLM generation is false, deterministic chunks are rewritten by the LLM. |
| `use_llm_query_paraphrase` | `false` | If true, rendered queries are paraphrased by the LLM if required labels remain present. |
| `llm_chunk_max_attempts` | `3` | Max generation/rewrite attempts before rejection behavior applies. |
| `llm_temperature` | `0.1` | Temperature passed to Ollama. |
| `llm_num_ctx` | `4096` | Context window passed to Ollama. |

The dominant/complementary counts define the intended top-k failure mode. If the dominant facet has many more gold chunks than the other facets, nearest-neighbor retrieval has a plausible way to over-select repeated evidence.

## `embeddings`

`EmbeddingCfg` controls [retrieval/embed.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/retrieval/embed.py):

| field | default | implementation role |
| --- | --- | --- |
| `model_name` | `multi-qa-mpnet-base-cos-v1` | SentenceTransformers model passed to `helpers.embedder.Embedder`. |
| `batch_size` | `64` | Embedder batch size; the parquet streaming bucket is `max(batch_size * 32, 32768)`. |
| `device` | `cuda` | Device passed to the embedder. |
| `query_prompt` | `null` | Optional query prompt for models that need query instructions. |
| `normalize` | `true` | If true, document and query vectors are normalized before cosine-style dot products. |

The rest of the retrieval code assumes dot product is the similarity score. With normalized vectors, this is cosine similarity.

## `retrieval`

`RetrievalCfg` controls candidate-pool construction and selection:

| field | default | implementation role |
| --- | --- | --- |
| `pool_scope` | `query_local` | Candidate universe before semantic top-N. |
| `candidate_pool_n` | `300` | Number of most query-similar candidates retained before strategy selection. |
| `k_values` | `[5, 10, 20]` | Selection budgets evaluated for every strategy. |
| `lambda_values` | `[0.3, 0.5, 0.7]` | Relevance/coverage tradeoff values for MMR and facility-location. |
| `strategies` | `['top_k', 'mmr', 'fac_loc']` | Retrieval strategies evaluated. |
| `mmr_window` | `null` | If set, MMR only penalizes similarity to the last `window` selected items. |
| `only_pass_geometry` | `true` | If true, evaluation skips queries failing `geometry_stats.passes_filter`. |

The three pool scopes are implemented in `candidate_pool_indices()`:

- `query_local`: candidates are chunk documents linked to the query through `chunk_memberships.parquet`.
- `same_condition`: candidates are chunks with the same `condition_id` as the query.
- `full_corpus`: candidates are all chunks.

## `geometry`

`GeometryCfg` controls [retrieval/filter_geometry.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/retrieval/filter_geometry.py):

| field | default | meaning |
| --- | --- | --- |
| `topk_dominance_k` | `10` | The top-k depth used to test whether one facet dominates nearest-neighbor retrieval. |
| `min_topk_dominant_count` | `5` | Minimum number of top-k chunks from the most frequent gold facet. |
| `min_in_minus_cross_similarity` | `0.03` | Required gap between mean same-facet and cross-facet gold similarity. |
| `min_distractors_in_pool` | `10` | Minimum number of hard distractors that must appear in the semantic candidate pool. |

A query passes the filter only if all planned facets are present in the candidate pool, top-k shows enough dominant-facet concentration, same-facet gold chunks are more similar than cross-facet gold chunks by the configured margin, and enough distractors are present.

## `embedding_geometry`

`EmbeddingGeometryCfg` controls the diagnostic visualization stage:

| field | default | meaning |
| --- | --- | --- |
| `n_queries` | `6` | Number of queries to visualize if `query_ids` is empty. |
| `query_ids` | `[]` | Explicit query ids to visualize. |
| `candidate_pool_n` | `null` | Candidate pool size for plots; `null` reuses `retrieval.candidate_pool_n`. |
| `plot_k` | `10` | Selection budget overlaid in geometry plots. |
| `reduction` | `umap` | 2D reduction method, `umap` or `pca`. |
| `pca_dims` | `null` | Optional PCA dimensionality before UMAP and HDBSCAN. |
| `umap_metric` | `cosine` | UMAP metric. |
| `umap_neighbors` | `15` | UMAP neighborhood size. |
| `umap_min_dist` | `0.08` | UMAP minimum distance. |
| `hdbscan_min_cluster_size` | `5` | Minimum HDBSCAN cluster size. |
| `hdbscan_min_samples` | `null` | Optional HDBSCAN `min_samples`. |
| `random_state` | `42` | PCA/UMAP reproducibility seed. |

## Experiment Paths

`MedicalDatasetGenPaths` centralizes all paths. For experiment `<exp>`, the root directory is:

```text
src/experiments/medical_dataset_gen/_results/<exp>
```

The standard directories are:

- `_logs/`
- `_figures/`

The standard table path is:

```text
_results/<exp>/<table>.parquet
```

`setup_logging()` tees stdout into `_logs/<script_name>.log` unless `--no-log-tee` is set.

## Persisted Tables

The main parquet tables are:

| table | produced by | purpose |
| --- | --- | --- |
| `query_plans.parquet` | `plans` | Hidden query/facet design. |
| `clinical_facts.parquet` | `facts` | Hidden atomic evidence and distractors. |
| `chunk_documents.parquet` | `chunks` | Unique rendered note chunks keyed by reusable document `chunk_id`. |
| `chunk_memberships.parquet` | `chunks` | Query-specific membership rows linking queries/facts/facets to chunk documents. |
| `generation_rejects.parquet` | `chunks` | Failed generation or validation rows. |
| `queries.parquet` | `queries_answers` | Natural-language query text and query metadata. |
| `gold_answers.parquet` | `queries_answers` | Canonical answer text and supporting fact ids. |
| `qrels.parquet` | `qrels` | Query-local chunk relevance labels derived from memberships. |
| `geometry_stats.parquet` | `geom_filter` | Per-query embedding geometry pass/fail diagnostics. |
| `evaluation_results.parquet` | `eval` | Per-query, per-strategy, per-k, per-lambda metrics. |
| `evaluation_stats.parquet` | `eval` | Grouped summary metrics. |
| `embedding_geometry_points.parquet` | `geom_plots` | Per-point 2D plot coordinates and labels. |
| `embedding_geometry_query_stats.parquet` | `geom_plots` | Per-visualized-query cluster and selection diagnostics. |

## Embedding Files

The current embedding stage writes memory-mappable arrays:

- `embeddings_chunk_vectors.npy`
- `embeddings_query_vectors.npy`
- `embeddings_chunk_ids.npy`
- `embeddings_query_ids.npy`
- `embeddings_metadata.json`

`load_embedding_arrays()` first tries this memmap format. It still supports the older `embeddings.npz` fallback for previous experiment artifacts.

## Chunk Caches

Chunk generation uses JSONL caches:

- Per-experiment direct generation cache: `_results/<exp>/chunk_generation_cache.jsonl`
- Shared direct generation cache: `src/experiments/medical_dataset_gen/_cache/chunk_generation_cache.jsonl`
- Shared rewrite cache: `src/experiments/medical_dataset_gen/_cache/chunk_rewrite_cache.jsonl`

Direct LLM generation uses `GENERATION_CACHE_VERSION = 9`. LLM rewrite uses `REWRITE_CACHE_VERSION = 1`. Cache entries are ignored if their version does not match the current code.
