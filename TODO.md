## Primary
* Figure out if the current `FacetCoveragePurity@k` is the target evaluation metric for our benchmark, or if we have to include even more evaluation metrics inside of it.
* Explain more formally and mathematically why coverage works only with very aggressive lambda values, < 0.40



## Secondary
* Compare embedding_calibrarion with rotating.



## Completed

### After 2026/06/30 call
* Improved an finalized query design

* Granurarily configurable distractors/outliers both for the facet-attached distractors and the background outliers.

* Design two kinds of new metrics:
    - `FacetCoveragePurity@k` = `FacetCoverage@k` * `Precision@k`, combining our two goals: retrieving more query facets while avoiding distractors/outliers.
    - `AllFacetCleanRate@k` = Percentage of queries that have `FacetCoverage@k` == 1 && `Precision@k` >= clean_rate_threshold (0.80)

* Fix ontology and avoid ambiguous wording that made some clinical axis (especially care_intensity) blend into others. Strengthened overall the ontology adding, for each primary condition, allowlists for the comorbidity positive vs. negative & positive vs. positive scenarios.

* In `lambda_selection.py` and the related global evaluation configs: deprecate the weighted average of metrics. Always use the `FacetCoveragePurity@k` metric as the maximizing metric, because it combines everything we want to achieve in the benchmark.


--- 


## Ideas about file://home/pagnozzi/thesis/_calls/call_2026-06-30.md
* `FacetCoveragePurity@k`:
  - How should different values of `k` be handled? Should we maximize the metric separately for each `k` and obtain `lambda*(k)`, or maximize it overall across all our `k` values?

* Many works that use “restricted budgets” focus directly on tokens:
  - I use the number of chunks, but Emilia would prefer something more granular because the context is very tight.
  - Does it make sense to do this now?
