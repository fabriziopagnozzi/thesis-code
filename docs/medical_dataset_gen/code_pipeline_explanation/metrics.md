# Evaluation Metrics

The evaluation separates ordinary binary relevance from facet-aware coverage. This matters because the benchmark is designed to expose cases where top-k retrieves relevant chunks but repeats the same answer facet.

## Summary Metrics

### Precision@k

Precision at k. The fraction of selected chunks that are gold for any answer facet.

Formula: `gold_selected / k`.

### Recall@k

Recall at k over all gold chunks for the query. This treats every gold chunk as interchangeable.

Formula: `gold_selected / total_gold_chunks`.

### F1@k

Harmonic mean of `Precision@k` and `Recall@k`.

### MAP@k

Mean average precision at k. Per query, `AP@k` averages `Precision@rank` over ranks where the selected chunk is gold, then divides by `min(total_gold_chunks, k)`. The summary reports the mean over queries.

This is kept as a secondary conventional IR metric in `evaluation_stats.parquet`, but it is not used in the evaluation plots. It does not care which facets the gold chunks cover, so it can look good in exactly the redundant same-facet case this benchmark is meant to expose.

### FacetCoverage@k

Facet coverage at k. The fraction of answer facets with at least one selected gold chunk.

Formula: `facets_hit / total_facets`.

This is the main metric for the thesis claim: a method should recover distinct answer facets, not only many redundant gold chunks.

### FacetMRR@k

Mean reciprocal rank of the first gold hit for each facet. Missing facets contribute zero.

This answers: how early does the retrieved context expose each answer facet?

### alpha-nDCG@k

Diversified ranking metric using `facet_id` as the subtopic label. Repeated gold chunks from the same facet receive diminishing gain, controlled by `ALPHA_NDCG_REDUNDANCY`.

This rewards early relevance, early facet coverage, and less redundant same-facet evidence.

### DistractorRate

The fraction of selected chunks that are not gold for any facet.

Lower is better.

For runs with background outlier clusters, the raw result rows and summary table
also split this into `NearMissDistractorRate`, `BackgroundOutlierRate`, and
`AnyDistractorRate`. `DistractorRate` is kept as the compatibility name for
total non-gold selections.

### DominantFacetRate

The fraction of selected chunks whose `facet_id` equals the query's planned dominant facet.

Lower is usually better when high values indicate repeated evidence from the same dominant facet.

### RedundantGoldRate

The fraction of selected chunks that are gold but do not add a new facet beyond facets already hit.

Formula: `(gold_selected - facets_hit) / k`.

Lower is better for multi-facet coverage.

## Retrieval Diagnostics

### fac

Facility-location objective score for the selected set, computed from candidate-candidate similarities.

### avg_cos

Average query cosine similarity of the selected chunks.

### jac

Jaccard overlap between a method's selected set and the top-k selected set. For top-k itself this is `1.0`.

## Raw Result Columns

`evaluation_results.parquet` also keeps machine-friendly raw names:

- `gold_precision`, `gold_recall`, `gold_f1`
- `average_precision_at_k`
- `facet_coverage`
- `weighted_facet_coverage` is retained as a table diagnostic, but the plot grids do not use it.
- `facet_mrr_at_k`
- `alpha_ndcg`
- `distractor_rate`, `dominant_facet_rate`, `redundant_gold_rate`
- `facet_hit_density` and `unique_facet_rate`, which are diagnostic aliases for `facets_hit / k`

The old raw names `aspect_precision`, `aspect_f1`, `facet_mrr`, and `dominant_cluster_concentration` are retained only for compatibility. They should not be used as headline thesis labels.
