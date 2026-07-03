# Primary
* Keep fine tuning the medical ontology to produce meaningful queries.

* Split validation/test data and tune hyperparameter lambda and see results
    - add "mode: exploring | testing" under "evaluation:" key
    - what is k in our final benchmark? Do we need to consider multiple values or focus on a small enough value (restricted budget)?
        - Should we maximize the `FacetCoveragePurity@k` separately for each `k` and obtain `lambda_strategy*(k)`, or maximize it overall across all our `k` values and obtain `lambda_strategy*`?

* "pool_mode: global" to scale up the dataset like Martinenghi wants

* how to make it all more realistic?
    - Draw inspiration from MIMIC
        - Get some aggregated stats about conditions, demographics, comorbidities
        - Try to identify patterns in the data
        - This process will turn out useful also later if we want to go back to the original MIMIC pipeline and make it work.

    - Fine tune the medical ontology somehow
    - use LLMs to rewrite chunks into a more natural clinical prose while retaining all the meaning --> drawback: expensive and time-consuming

* Data distribution
    - Should we formalize better and categorize different kinds of distributions and present results for each of these?


---------

# Secondary
* Figure out if the current `FacetCoveragePurity@k` is the target evaluation metric for our benchmark, or if we have to include even more evaluation metrics inside of it.
* Explain more formally and mathematically why coverage works only with very aggressive lambda values, < 0.40
* Compare embedding_calibrarion with rotating.

### ([From 30/06 call](file:/home/pagnozzi/thesis/_calls/call_2026-06-30.md))
* More granular budget based on token rather than num. documents 
* Lexical overlap with the answer: see what you can do to improve the AnswerROUGE metrics.


---------

# DONE

## After 2026/07/03 call
* _subconfig.yaml and subexperiments support for more explainability

## After 2026/06/30 call
* Improved an finalized query design

* Fix ontology and avoid ambiguous wording that made some clinical axis (especially care_intensity) blend into others. Strengthened overall the ontology adding, for each primary condition, allowlists for the comorbidity positive vs. negative & positive vs. positive scenarios.

* Granurarily configurable distractors/outliers both for the facet-attached distractors and the background outliers.

* Design two kinds of new metrics:
    - `FacetCoveragePurity@k` = `FacetCoverage@k` * `Precision@k`, combining our two goals: retrieving more query facets while avoiding distractors/outliers.
    - `AllFacetCleanRate@k` = Percentage of queries that have `FacetCoverage@k` == 1 && `Precision@k` >= clean_rate_threshold (0.80)

* In `lambda_selection.py` and the related global evaluation configs: deprecate the weighted average of metrics. Always use the `FacetCoveragePurity@k` metric as the maximizing metric, because it combines everything we want to achieve in the benchmark.

* Made lots of experiments, 21 through 48.