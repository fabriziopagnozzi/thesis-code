# CHANGELOG

---

## After [2026/07/03 meeting](file:/home/pagnozzi/thesis/notes/call_2026-07-03.md)
### _subconfig.yaml and subexperiments support for more explainability

### Keep fine tuning the medical ontology to produce meaningful queries.
    - Added sixth axis diagnostic_evidence_type
    - Changed some wording to avoid leakage into other axes

### Add embeddings global caching mechanism
    - for deterministic chunks to save lots of time and compute when doing experiments, avoiding expensive embedding recomputation of the same rendered chunks.

### Data distribution formal categorization and properties; further variation

## Use the "split" property in the query records to distinguish validation data and test data and tune lambda for held-out evaluation on the test queries
    * add "mode: exploring | testing" under the "evaluation:" key in the global config schema
    * what is k in our final benchmark? Do we need to consider multiple values or focus on a small enough value (restricted budget)?
        - maximize the `FacetFacLocPurity@k` separately for each `k` and obtain `lambda_strategy*(k)`
        - in the end, we can focus on a single k value --> the LOWEST one. This is already used by the experiment analysis module and is called "baseline"

## Fix everywhere the geometry filter configuration
    * We're focusing on restricted budgets: the geometry filter should encode for top-k failures when budget is restricted, NOT when it's inflated!
        - Fix all the geometry_filter.top_k values in configs.


---

## After [2026/06/30 meeting](file:/home/pagnozzi/thesis/notes/call_2026-06-30.md)
* Improved an finalized query design

* Granurarily configurable distractors/outliers both for the facet-attached distractors and the background outliers.

* Fix ontology and avoid ambiguous wording that made some clinical axis (especially care_intensity) blend into others. Strengthened overall the ontology adding, for each primary condition, allowlists for the comorbidity positive vs. negative & positive vs. positive scenarios.

* Design two kinds of new metrics:
    - `FacetCoveragePurity@k` = `FacetCoverage@k` * `Precision@k`, combining our two goals: retrieving more query facets while avoiding distractors/outliers.
    - `AllFacetCleanRate@k` = Percentage of queries that have `FacetCoverage@k` == 1 && `Precision@k` >= clean_rate_threshold (0.80)

* In `lambda_selection.py` and the related global evaluation configs: deprecate the weighted average of metrics. Always use the `FacetCoveragePurity@k` metric as the maximizing metric, because it combines everything we want to achieve in the benchmark.

* Prepared a bunch of experiments to run.

---