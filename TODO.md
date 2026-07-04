# Code - Primary
* Data distribution
    - formalize better and categorize different kinds of distributions and present results for each of these
    - bottom line is that MMR behaves overall worse than FacLoc

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
    
---------

# Writing - Primary
* TASK: formulate current claims precisely, given the experiment results: write them down, half a dozen claims we're ready to do. 
    - First, frame the benchmark properties (structure of queries, chunks, facets)
    - then general settings used (model, k & lambda values...)
    - then evaluation and diagnostics metrics and the rationale behind them.
   
    * Claims so far:
        - FacLoc is overall a safer method across all lambda values - MMR can do worse than top-k because while it covers the space it can fall into distractors way more easily.
        - consistely better than top-k across all lambda values
            - claim that using validation/test data after tuning hyperparameter lambda FacLoc still seems to outperform MMR across all data distributions.
        - depending on dataset distribution we get results and the trend overall on the main evaluation metric is that FacLoc performs almost always equally or better than top-k 
        - MMR diversifies with balanced lambda values while FacLoc shows improvements only with low enough values (= high FacLoc weight)
        - ... (maybe include also other results drawn from diagnostics, maybe how the hidden variables like the model, the geometry etc. have an impact on this)

---------

# Code - Secondary
* Figure out if the current `FacetCoveragePurity@k` is the target evaluation metric for our benchmark, or if we have to include even more evaluation metrics inside of it.
* Explain more formally and mathematically why coverage works only with very aggressive lambda values, < 0.40
* Compare embedding_calibrarion with rotating.

### ([From 30/06 call](file:/home/pagnozzi/thesis/_calls/call_2026-06-30.md))
* More granular budget based on token rather than num. documents 
* Lexical overlap with the answer: see what you can do to improve the AnswerROUGE metrics.


---------
