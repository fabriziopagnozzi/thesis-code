# Code - Primary

## Add more report plots

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

---------

# Writing - Primary

## TASK: formulate half a dozen claims precisely, given the current experiment results.

### Claims so far
* on each distribution, FacLoc is consistely better than top-k in FCP@k across all lambda values, thus it's overall a safer method.
    - MMR, while covering well the facets, can fall into distractors and can be arbitrarily worse than top-k at very low lambda values.
    - 28/A shows that even for very low values of lambda, FacLoc does no worse than top-k as for FCP@k --> more stable method, more robust to distractor presence.
    - claim that using validation/test data after tuning hyperparameter lambda FacLoc still seems to outperform MMR across all data distributions


* across distinct distributions, the trend overall FCP@k is that FacLoc performs almost always equally or better than MMR and top-k, with some minor exceptions.
    - find exceptions, if any, and list them --> there shouldn't be.
    - categorize and formalize the properties of the data distributions we have experimented on
    - Present results:
        - S01_balanced_clean vs. S02_balanced_no_bg: when there is a limited number of near-miss distractors but no actual background outlier, MMR is far worse than FacLoc. 
        - ...


    - Given the differences between smaller and bigger models, we can conclude that this ontology produces different classes of queries in terms of difficulty: some are by design "harder" than others and require better models to capture their nuances.
        - even with stronger models, there is a small percentage of queries that show unseparable facet geometry.
            - this may be due to the semantical clinical aspects overlapping in the embedding space for particular combinations of Primary Condition / Subgroup / Clinical Axis which are not currently considered by the ontology
            - it may be due to poor wording in the rendered chunks

* considerations on alpha-nDCG: 
    - coverage tends to diversify earlier in the ranking, due to the its objective function encoding the broader dataset distribution.


* add "pool_scope: global" and discuss the results.


* ... (maybe include also other results drawn from diagnostics)


## DONE

### Introduction and pipeline presentation
* first, frame the benchmark properties (structure of queries, chunks, facets) -->
* then outline the main pipeline steps:
    - the general settings used (k & lambda values...)
    - medical ontology structure
        - very briefly: conditions, subgroups, clinical axes
    - deterministic template rendering
        - query types and constrasts: Primary Condition + Subgroup_1 vs. Subgroup_2 + Clinical_Axis_1 & Clinical_Axis_2
        - once we fix the query structure, we talk more deeply about the ontology and the reasons why it was designed like that:
            - allowlists for each condition to limit the combinatorial explosion and produce only meaningful queries as far as the clinical soundness is concerned
            - allowed cohort contrasts to compare in the query
            - allowed clinical axis pairs to include in the query
    - embedding
    - filtering based on the properties emerging from the embedding space
        - based on the highest k value set for retrieval, we enforce max. 2 facets retrieved out of 4.
    - evaluating, along with metrics and diagnostics definition and the rationale behind them

* also talk about AllFacetCleanRate@k
    - sometimes coverage shows a higher percentage of queries with perfect coverage (FacetCoverage@k == 100%) and very high precision (Precision@k >= 80%)
    - [results]: interpret the plots and check whether MMR can do better.

* MMR diversifies with balanced lambda values while FacLoc shows improvements only with low enough values (= high FacLoc weight)
    - due to mathematical properties of their scoring functions
    - show the greedy step update formulas and discuss 

### Claims

* the embedding model choice has an impact
    - smaller models, fewer dimensions (multi-qa-mpnet-base-cos-v1):
        - show a lower number of post-filtering queries compared to bigger models & dimensions (bge_m3, qwen3). 
            - This may be due to their less nuanced understanding of the semantics, yielding embedding vectors which mix up the content of the rendered clinical chunks. The chunks are semantically separable but pertain to the same domain, have similar template structure, etc.
            - Add table showing pass rate of the filter_queries step and show that bigger models do better 
        - even with fewer queries, their evaluation has roughly the same properties as the bigger models with more queries
        - due to the lower query pass rate, the difference in the evaluation between pass-only and all-queries are a bit more evident than bigger models. They can bump up the results of MMR, but coverage stays on top.

    - bigger models, more dimensions (bge_m3, qwen3[0.6B | 4B | 8B])
        - the evaluation results seem to NOT change too much whether we include or exclude the queries that do NOT pass the geometry_filter we impose
        - this may be due to the fact that the poorly-performing queries under low retrieval budgets are typically <= ~4%

    - [results]: see if the metrics/diagnostics change among distinct models for the same data distributions (and potentially different number of queries)
    - [results]: see if the metrics/diagnostics change among distinct models for the same data distributions and ALL of queries considered
    - [results]: check if the diagnostics of smaller models change over the diagnostics of bigger models    



---------

# Code - Secondary
* Figure out if the current `FacetCoveragePurity@k` is the target evaluation metric for our benchmark, or if we have to include even more evaluation metrics inside of it.
* Explain more formally and mathematically why FacLoc works only with very aggressive lambda values, < 0.40
* Compare embedding_calibrarion with rotating.
* Rerankers impact?

## ([From 30/06 call](file:/home/pagnozzi/thesis/notes/call_2026-06-30.md))
* More granular budget based on token rather than num. documents 
* Lexical overlap with the answer: see what you can do to improve the AnswerROUGE metrics.

---------

# Ideas

## Lambda stability across distinct medical datasets distributions
- Evaluate lambda stability across medical dataset realizations. If FacLoc shows lower variance or a wider stable lambda region, then test whether the result persists across multiple LLM-generated ontologies with the same abstract facet/contrast structure. (Point below, Cross-domain lambda stability)

## Cross-domain lambda stability
* An interesting thing to study could be the cross-semantic-domain stability of the FacLoc and MMR lambda values. To obtain a result, the idea would be to:
    - generate ontologies across different domains;
        - Problem: understand whether the code is generic enough to allow chunks to be rendered anonymously with respect to the current medical ontology.
    - evaluate the benchmark results separately on the different ontologies and understand which FacLoc and MMR lambda values perform best on each one;
    - observe how much the optimal lambda values vary.

- In a real-world setting where retrieval is performed over massive and heterogeneous data sources in terms of semantic content -- medical, legal, financial, scientific literature, enterprise knowledge bases, ... -- lambda stability is important. In a real pipeline, the steps would be:
    - pre-filtering with K_pre >> k;
    - running the diversification methods on the pre-filtered set.
            
- If one chooses to diversify, the issue of tuning the lambda parameter arises. This tuning is performed based on the content of the available knowledge base. However, the knowledge base may be dynamic, requiring continuous tuning and redeployment.
    - A method that performs better overall is more robust and easier to maintain, less dependent on the data distribution and on the semantic content of the documents it retrieves.
    - If we can demonstrate, at least on synthetic data across mutliple domains (or also WITHIN the same domain), that FacLoc performs better than MMR and is more lambda-stable, that would be a very strong and more “general” result.

