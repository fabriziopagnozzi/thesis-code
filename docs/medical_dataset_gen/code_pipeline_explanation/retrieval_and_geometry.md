# Embeddings, Retrieval, And Geometry Filtering

This note explains the retrieval side of the implemented medical dataset pipeline: embedding text, constructing candidate pools, selecting chunks with top-k/MMR/facility-location, and filtering queries by embedding geometry.

Source files:

- [retrieval/embed.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/retrieval/embed.py)
- [retrieval/utils.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/retrieval/utils.py)
- [retrieval/filter_geometry.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/retrieval/filter_geometry.py)
- [helpers/query_algorithms.py](/home/pagnozzi/thesis/src/helpers/query_algorithms.py)
- [helpers/metrics.py](/home/pagnozzi/thesis/src/helpers/metrics.py)
- [embedding_geometry/run.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/embedding_geometry/run.py)
- [embedding_geometry/artifacts.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/embedding_geometry/artifacts.py)
- [embedding_geometry/reduction.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/embedding_geometry/reduction.py)
- [embedding_geometry/diagnostics.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/embedding_geometry/diagnostics.py)

## Stage 6: Embeddings

`run_embed()` writes chunk and query embedding arrays. The relevant config fields are:

- `embeddings.model_name`
- `embeddings.batch_size`
- `embeddings.device`
- `embeddings.devices`
- `embeddings.query_prompt`
- `embeddings.normalize`

The implementation uses `helpers.embedder.Embedder`, which wraps SentenceTransformers-style document and query embedding calls. Chunk text is read from `chunk_documents.parquet`; query text is read from `queries.parquet`.
If `embeddings.devices` contains more than one device, the embedder uses a SentenceTransformers multi-process pool to data-parallelize encoding across those devices. This affects both the final `embed` stage and the `calibrate_plans` probe embeddings.

The stage streams parquet batches instead of loading all text into one list. It computes:

```python
bucket_size = max(cfg.embeddings.batch_size * 32, 32768)
```

Each batch writes into `.npy` memmaps:

- `embeddings_chunk_vectors.npy`
- `embeddings_query_vectors.npy`
- `embeddings_chunk_ids.npy`
- `embeddings_query_ids.npy`

The metadata file records model name, vector dimension, normalization, and array paths in `embeddings_metadata.json`.

This stage is fundamental because every later retrieval score is a dot product. With `embeddings.normalize = true`, the dot product is cosine similarity:

```text
sim(query, chunk) = e_query dot e_chunk
```

## Shared Retrieval Indexes

`build_index_maps()` in [retrieval/utils.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/retrieval/utils.py) turns dataframes and embedding id arrays into lookup structures:

- `chunk_id_to_idx`: chunk id to embedding row index
- `query_id_to_idx`: query id to embedding row index
- `chunk_by_id`: chunk document id to metadata row
- `query_by_id`: query id to metadata row
- `membership_by_query_chunk`: `(query_id, chunk_id)` to query-specific membership row
- `chunks_by_source_query`: source query id to chunk-document embedding indices
- `chunks_by_condition`: condition id to chunk-document embedding indices

This is the bridge between hidden labels in parquet and vector operations in NumPy.

## Candidate-Pool Construction

`candidate_pool_indices()` implements the configured `retrieval.pool_scope`.

For `query_local`, it returns:

```text
{chunk index | chunk is linked to query_id in chunk_memberships.parquet}
```

The pool contains the generated local gold chunk documents and generated local distractors for that hidden query plan. Relevance labels still come from `qrels.parquet`, so the same chunk document can be interpreted relative to each query.

For `same_condition`, it returns:

```text
{chunk index | chunk.condition_id == query.condition_id}
```

For `full_corpus`, it returns every chunk index.

After the scope filter, `topn_by_query()` computes query similarity for every candidate:

```python
sims = chunk_vectors[candidate_indices] @ query_vector
```

It sorts candidates descending by similarity and retains at most `retrieval.candidate_pool_n`. All selection methods then operate only inside this top-N semantic pool. This two-stage design separates broad candidate generation from reranking or diversification.

## The Three Implemented Selectors

The configured strategies are evaluated through `select_indices()` in [retrieval/utils.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/retrieval/utils.py), which delegates to [helpers/query_algorithms.py](/home/pagnozzi/thesis/src/helpers/query_algorithms.py).

### Top-k

`top_k()` returns the `k` largest query similarities:

```text
S = argsort(sim_to_query, descending=True)[:k]
```

Top-k is the baseline. It has no notion of redundancy. If many near-identical chunks from the dominant facet are closest to the query, top-k can select them repeatedly.

### MMR

`mmr()` implements maximal marginal relevance:

```text
score(i) = lambda * sim(q, i) - (1 - lambda) * max_{j in W} sim(i, j)
```

`W` is either all selected items or the last `retrieval.mmr_window` selected items if a window is configured. The first selected item uses only query similarity.

MMR is a dispersion method. It penalizes candidates that are too similar to already selected chunks. That can help avoid exact redundancy, but it can also push the selection toward semantically distant distractors. This is the thesis-relevant distinction between dispersion and coverage: distance from selected chunks is not the same thing as covering the remaining answer facets.

### Facility Location

`fac_loc()` implements a lazy greedy facility-location selection objective:

```text
coverage(S) = (1 / n) * sum_i max_{j in S} sim(i, j)
score(candidate j) = lambda * sim(q, j) + (1 - lambda) * marginal_coverage_gain(j)
```

The running coverage vector is:

```text
m_i = max_{j in S} sim(i, j)
```

The marginal coverage gain for a new candidate is:

```text
gain(j) = (1 / n) * sum_i max(0, sim(i, j) - m_i)
```

The implementation uses a max heap with lazy recomputation. Initial gains are computed with the empty selected set. When a heap item reaches the top, its gain is recomputed against the current coverage vector unless it has already been updated in the current greedy step. This avoids recomputing all candidate gains on every step.

Facility location is the key coverage method in this benchmark. It rewards candidates that represent many not-yet-covered candidates in the pool. If facets form local clusters in embedding space, one good representative from each facet can have high marginal coverage.

The implemented selector already combines query relevance and coverage through `lambda * sim(q, j) + (1 - lambda) * marginal_coverage_gain(j)`. The current pipeline does not add a separate relevance-weighted coverage objective; low-query-similarity background clusters are instead measured explicitly through distractor and geometry diagnostics.

## Retrieval Diagnostics

`retrieval_diagnostics()` computes:

- `fac_cov_score`: mean facility-location coverage score of the selected set.
- `avg_cos`: mean query similarity of selected chunks.
- `jaccard_vs_topk`: Jaccard overlap against the top-k selected set, or `1.0` for top-k itself.

`fac_cov_score` is:

```text
fac(S) = (1 / |D|) * sum_i max_{j in S} cos(e_i, e_j)
```

It is useful as a geometry diagnostic, not as the main gold-label metric. A method can have high embedding coverage while still selecting distractors, so evaluation also computes gold precision and facet coverage.

## Stage 7: Geometry Filtering

`run_filter_geometry()` creates `geometry_stats.parquet`. The relevant config fields are:

- `retrieval.pool_scope`
- `retrieval.candidate_pool_n`
- `geometry.topk_dominance_k`
- `geometry.primary_topk_dominance_k`
- `geometry.min_topk_dominant_count`
- `geometry.max_topk_retrieved_facets`
- `geometry.min_in_minus_cross_similarity`
- `geometry.min_distractors_in_pool`

The geometry filter asks whether a query actually exhibits the intended coverage-sensitive structure in embedding space.

For each query:

1. Build the configured candidate pool.
2. Keep the top `retrieval.candidate_pool_n` candidates by query similarity.
3. Check whether every hidden gold facet has at least one chunk in the retained candidate pool.
4. Count how many of the top `geometry.primary_topk_dominance_k` candidates come from the planned dominant gold facet.
5. Count how many distinct gold facets are already represented by top-k at that same depth.
6. Count distractors in the retained candidate pool.
7. Separately count near-miss hard distractors and background outlier chunks.
8. Compute same-facet and cross-facet gold similarity.
9. Compute background outlier compactness, query-similarity margin, and first/median query-rank diagnostics.
10. Compute top-k versus facility-location diagnostics.

The same-facet and cross-facet computation uses only gold chunks for the query. It builds a gold-gold similarity matrix and labels each gold chunk by `facet_id`:

```text
mean_in_facet_similarity    = mean sim(i, j) where facet_i == facet_j and i != j
mean_cross_facet_similarity = mean sim(i, j) where facet_i != facet_j
```

A query passes if:

```text
all facets are present in the candidate pool
planned_topk_dominant_count >= min_topk_dominant_count
if max_topk_retrieved_facets is not null:
    n_topk_retrieved_facets <= max_topk_retrieved_facets
mean_in_facet_similarity - mean_cross_facet_similarity >= min_in_minus_cross_similarity
n_distractors_in_pool >= min_distractors_in_pool
background outlier clusters are present and complete when enabled
```

The `max_topk_retrieved_facets` cap is meant to reject queries where plain nearest-neighbor top-k already covers too many answer facets at the primary evaluation-aligned depth. Those queries may still be valid clinical questions, but they are weak tests of coverage-oriented reranking because the baseline is already too representative.

`geometry_stats.parquet` also stores per-k top-k diagnostics for the configured retrieval `k_values`, the legacy `geometry.topk_dominance_k`, and the primary dominance k. The old `topk_dominant_count` field remains as the most frequent gold-facet count, while `planned_topk_dominant_count` is the pass/fail field tied to the calibrated dominant facet.

`geometry.min_distractors_in_pool` is applied to the near-miss hard negatives, not to background outliers. Background outliers have their own diagnostics because they are meant to test whether a selector wastes budget on an irrelevant but coherent clinical island.

This filter is fundamental because the benchmark claim depends on geometry. If embeddings do not cluster redundant same-facet evidence and separate facets at least somewhat, the retrieval comparison becomes noise.

## How Evaluation Uses The Filter

`retrieval.only_pass_geometry` controls whether [evaluation/evaluate.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/evaluation/evaluate.py) skips failed queries. When it is true, evaluation only scores queries whose `geometry_stats.passes_filter` is true.

The evaluation stage also calls `_assert_pool_scope_match()` to verify that `geometry_stats.pool_scope` matches the current config. This prevents accidentally evaluating `query_local` metrics against geometry statistics produced under `same_condition` or `full_corpus`.

## Stage 9: Embedding Geometry Diagnostics And Plots

`run_embedding_geometry()` is the diagnostic visualization stage. It reads chunks, queries, qrels, geometry stats, evaluation stats, evaluation results, and embedding arrays. It also verifies that stored geometry and evaluation tables match the current `retrieval.pool_scope`.

The relevant config fields are all under `embedding_geometry`, plus the retrieval fields used for candidate pools and selectors.

### Query Selection For Plots

`choose_query_groups()` chooses explicit `embedding_geometry.query_ids` if provided. Those manual IDs are written under `embedding_geometry/manual/`. Otherwise it ranks queries by a score that prefers:

- passing the geometry filter
- high facility-location facet-coverage gain over top-k at `embedding_geometry.plot_k`
- high top-k dominant count
- high in-minus-cross similarity
- enough distractors in the pool

With `embedding_geometry.query_selection: mixed`, the visualization budget is split between high-scoring `good` queries, central-ranked `mid` queries, and low-scoring `bad` queries. With `query_selection: best`, all automatically selected queries are written under `embedding_geometry/good/`. This automatic choice is not used for the main evaluation. It is only for selecting representative visualization cases.

### Query Artifact Construction

`build_query_artifact()` constructs one analysis artifact per selected query:

1. Rebuild the configured candidate pool.
2. Keep the configured top-N candidates.
3. Compute candidate-candidate similarity and query-candidate similarity.
4. Label each candidate as a hidden facet, generated distractor type, off-query same-condition chunk, or off-query wrong-condition chunk.
5. Compute top-k, MMR, and facility-location selections for the plot budget.
6. Reduce candidate vectors plus the query vector to 2D.
7. Run HDBSCAN on the clustering feature space.

The artifact contains both original high-dimensional vectors and 2D coordinates. Metrics are computed from the high-dimensional similarities; the 2D map is only a visualization.

In `full_strategy_selection_overlay_k<K>.png`, the MMR and facility-location rows always include the first plotted lambda and then collapse consecutive lambda values whose selected candidate membership is unchanged. Rank-only reorderings do not create an extra subplot; a new subplot appears only when the retrieved set itself changes.

### Reduction And Clustering

`reduce_for_plot()` supports:

- direct UMAP
- optional PCA preprocessing followed by UMAP
- PCA fallback if UMAP fails
- trivial coordinates for fewer than three points

`cluster_features()` returns either raw vectors or PCA-preprocessed vectors. `hdbscan_labels()` runs HDBSCAN with `hdbscan_min_cluster_size` capped by the number of points. If HDBSCAN is unavailable or fails, all points are marked noise with label `-1`.

### Diagnostic Tables

`point_rows()` writes one row per plotted candidate plus one query point. It stores rank, 2D coordinates, query similarity, hidden label, cluster role, gold flag, facet id, distractor type, HDBSCAN label, and whether each strategy selected the point.

`query_stats()` writes per-query diagnostics:

- number of hidden labels
- number of gold and distractor points
- gold-facet silhouette with cosine distance
- mean in-facet and cross-facet similarity
- query-to-gold and query-to-distractor similarity means
- HDBSCAN cluster count, noise rate, ARI, and NMI against hidden labels
- per-strategy selected facet count, gold precision, distractor rate, and dominant fraction

These diagnostics are useful for explaining whether a retrieval result reflects the intended hidden cluster structure or an embedding failure.
