## Primary
<!-- * Aumentare significativamente i distrattori, rendere il dataset più realistico. -->
* Capire se e come architettare un'altro tipo di metrica di evaluation aggregata.

* positive-vs-positive comorbidity contrasts. --> capire

* Matematicamente spiegare perché coverage funziona solo con lambda molto aggressivi, minori di 0.40

## Secondary
* Understand whether having a larger number of ontology bins is detrimental, although it does not seem to be.
* Compare embedding_calibrarion with rotating.



## Ideas about file://home/pagnozzi/thesis/_calls/call_2026-06-30.md
* `FacetCoveragePurity@k`:
  - How should different values of `k` be handled? Should we maximize the metric separately for each `k` and obtain `lambda*(k)`, or maximize it overall across all our `k` values?

* Many works that use “restricted budgets” focus directly on tokens:
  - I use the number of chunks, but Emilia would prefer something more granular because the context is very tight.
  - Does it make sense to do this now?



## Completed

### After 2026/06/30 call
* Fix the outliers and make everything granurarily configurable both for the facet-attached outliers and the background outliers.

* Fix some `Primary Condition` / `Subgroup` / `Clinical Axis` entries in the legend of the `query_geometry` plots that are not being colored gray, even though they do not belong to the query.
  * Reduce the marker size in the `for_lambda` plots.

* In `lambda_selection.py` and the related global evaluation configs: deprecate the weighted average of metrics. Always use the `FacetCoveragePurity@k` metric as the maximizing metric, because it combines everything we want to achieve in the benchmark.


## Scrapped
* Revert the Delta chart with lambdas to the old bar-plot style. Also consider how to make it better.