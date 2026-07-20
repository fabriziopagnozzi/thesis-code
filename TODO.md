# Code - Primary

## verifica che tutte le migrazioni siano giuste rirunnando una run vecchia e controllando i grafici

---------

# Writing - Primary

## Claims
* Given the differences between smaller and bigger models, we can conclude that this ontology produces different classes of queries in terms of difficulty: some are by design "harder" than others and require better models to capture their nuances.
    - even with stronger models, there is a small percentage of queries that show unseparable facet geometry.
        - this may be due to the semantic clinical aspects overlapping in the embedding space for particular combinations of Primary Condition / Subgroup / Clinical Axis which are not currently considered by the ontology
        - it may be due to poor wording in the rendered chunks


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

    - [results]: see if the metrics/diagnostics change among distinct models for the same data distributions
    - [results]: check if the diagnostics of smaller models change over the diagnostics of bigger models

---------

# Code - Secondary
* Figure out if the current `FacetCoveragePurity@k` is the target evaluation metric for our benchmark, or if we have to include even more evaluation metrics inside of it.
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

## Hyperparameter tuning may be overly favourable
Selecting a separate λ for every strategy, benchmark instance, and budget provides an upper estimate of tuned performance.
Include:
- globally tuned λ;
- one λ per embedding model;
- cross-family transfer, where λ is selected on some distributions and tested - others;

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
