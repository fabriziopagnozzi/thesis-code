# Code - Primary

## Data distribution
* formalize better and categorize different kinds of distributions and present results for each of these
* bottom line is that MMR behaves overall worse than FacLoc, no matter the distribution, given this dataset structure.


## Split validation/test data and tune hyperparameter lambda and see results
* add "mode: exploring | testing" under "evaluation:" key
* what is k in our final benchmark? Do we need to consider multiple values or focus on a small enough value (restricted budget)?
    - Should we maximize the `FacetCoveragePurity@k` separately for each `k` and obtain `lambda_strategy*(k)`, or maximize it overall across all our `k` values and obtain `lambda_strategy*`?


## Scale-up the dataset and retrieval
* "pool_mode: global" to scale up the dataset like Martinenghi wants
    - problem: when focusing on determining the validation/test set for a given query in "pool_mode: global" config, how to deal with facet distractors and background outliers coming from the other queries?
        - They can overlap with the 4 query facets, we should account for this and exclude all of these points from the test set for the query, otherwise the primary/secondary/niche cluster sizes are subject to these new golden chunks leaked from elsewhere.
    
    - add K_pre >> k hyperparameter for vanilla top-k prefiltering similarity sweep before applying the diversification method
        - One thing to study could be: how does FacLoc vs MMR change as K_pre increases and the prefiltered candidate pool becomes noisier?
            - That is likely where FacLoc’s advantage can become more meaningful.


## Make it realistic
* Draw inspiration from MIMIC
    - Get some aggregated stats about conditions, demographics, comorbidities
    - Try to identify patterns in the data
    - This process will turn out useful also later if we want to go back to the original MIMIC pipeline and make it work.
* Fine tune the medical ontology somehow
* use LLMs to rewrite chunks into a more natural clinical prose while retaining all the meaning --> drawback: expensive and time-consuming
* Eventually, go back to the MIMIC pipeline and find a way to make it work.
    

## Lambda stability across distinct medical datasets distributions
- Evaluate lambda stability across medical dataset realizations. If FacLoc shows lower variance or a wider stable lambda region, then test whether the result persists across multiple LLM-generated ontologies with the same abstract facet/contrast structure. (Point below, Cross-domain lambda stability)


## Cross-domain lambda stability
* An interesting thing to study could be the cross-semantic-domain stability of the FacLoc and MMR lambda values. To obtain a result, the idea would be to:
    - generate ontologies across different domains;
        - Problem: understand whether the code is generic enough to allow chunks to be rendered anonymously with respect to the current medical ontology.
    - evaluate the benchmark results separately on the different ontologies and understand which FacLoc and MMR lambda values perform best on each one;
    - observe how much the optimal lambda values vary.

- In a real-world setting where retrieval is performed over massive and heterogeneous data sources in terms of semantic content — medical, legal, financial, scientific literature, enterprise knowledge bases, ... — lambda stability is important. In a real pipeline, the steps would be:
    - pre-filtering with K_pre >> k;
    - running the diversification methods on the pre-filtered set.
            
- If one chooses to diversify, the issue of tuning the lambda parameter arises. This tuning is performed based on the content of the available knowledge base. However, the knowledge base may be dynamic, requiring continuous tuning and redeployment.
    - A method that performs better overall is more robust and easier to maintain, less dependent on the data distribution and on the semantic content of the documents it retrieves.
    - If we can demonstrate, at least on synthetic data across mutliple domains, that Coverage performs better than MMR and is more lambda-stable, that would be a very strong and “general” result.




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

## ([From 30/06 call](file:/home/pagnozzi/thesis/_calls/call_2026-06-30.md))
* More granular budget based on token rather than num. documents 
* Lexical overlap with the answer: see what you can do to improve the AnswerROUGE metrics.

---------
