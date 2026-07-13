## issues

### 1. Hyperparameter tuning and transferability

The current protocol selects the relevance–diversification parameter (\lambda) independently for each retrieval method, benchmark instance, and retrieval budget. This is appropriate for estimating the best achievable performance under validation-based tuning, but it may overstate how easily the methods would transfer to a new dataset or operating condition.

The final evaluation should therefore distinguish between:

* **instance-specific tuning**, representing an optimistic upper bound;
* **globally tuned parameters**, using one (\lambda) per method;
* **embedding-specific tuning**, using one value for each embedding model;
* **cross-family transfer**, tuning on some distribution families and testing on unseen families.

The exact validation grid, tie-breaking rule, and validation-set size should be reported. Performance curves across the entire (\lambda) grid are also important because they show robustness that a single tuned maximum cannot capture.

This is particularly relevant because the numerical scales of MMR and Facility-Location (\lambda) values are not directly comparable. The thesis already explains this mathematical distinction well, but the experimental protocol should make clear that each method receives an equally thorough and independent search over an appropriately chosen parameter range. 

---

### 5. Additional baselines and upper bounds are needed

Top-(k), MMR, and Facility-Location provide a clean core comparison, but they do not fully establish whether the observed gains are specific to Facility-Location or reflect a broader advantage of any structured diversification method.

At minimum, the experiments should include:

* **random selection**, providing a lower reference point;
* **an oracle facet-aware selector**, providing an upper bound under the available budget;
* **a simple clustering or medoid baseline**, testing whether explicit greedy facility-location optimization is necessary;
* ideally, one additional established diversification method such as xQuAD, IA-Select, a determinantal point process, or another submodular relevance-diversity objective.

The oracle is particularly valuable. At budget (k<|F_q|), complete facet coverage may be impossible; at larger budgets, some generated candidate pools may still contain facets that are hard to distinguish in the chosen embedding space. Oracle-normalized performance would show how much of the achievable coverage each method recovers.

Not every alternative discussed in the literature review must be implemented. The selected baselines should instead represent distinct mechanisms:

* relevance-only ranking;
* dispersion;
* density or cluster-based coverage;
* gold-aware upper-bound selection.

---

### 6. Template artifacts and lexical shortcuts require direct testing

Queries and chunks are rendered from structured ontology values and deterministic templates. This provides control, but it may also introduce repeated linguistic patterns that make facets, gold chunks, or distractor types unusually easy to identify.

The thesis should test whether the reported behaviour depends on these surface regularities. Useful checks include:

* evaluation on held-out rendering templates;
* paraphrased queries or chunks;
* reduced use of explicit subgroup and axis terminology;
* lexical-overlap statistics between queries and each document category;
* comparison of sparse lexical and dense semantic retrieval;
* a simple classifier predicting gold status or facet identity from bag-of-words features;
* inspection of nearest-neighbour examples for each embedding model.

A strong result would show that the relative method ordering persists even when surface forms change. A weak lexical baseline would also support the claim that the benchmark measures semantic set selection rather than template matching.

This does not require eliminating templates. It requires demonstrating that deterministic generation has not created unintended shortcuts.

---

### 7. Statistical analysis should account for repeated and correlated observations

The reported summaries count experiment–budget rows, but these rows are not fully independent. Results at different values of (k) may come from the same queries, several branches may share ontology structures, and all three retrieval strategies are evaluated on identical examples.

Win counts and average deltas are useful descriptive summaries, but they should not be interpreted as hundreds of independent replications.

The final analysis should include paired uncertainty estimates at the query level, such as:

* paired bootstrap confidence intervals;
* bootstrap resampling clustered by query or evidence profile;
* confidence intervals for mean method differences;
* effect sizes separated by experiment family and budget.

A hierarchical or mixed-effects analysis would be even stronger. Relevant grouping factors might include:

* condition;
* evidence profile;
* experiment family;
* embedding model;
* retrieval budget.

The main statistical unit should be stated explicitly. In most comparisons, the natural unit is the query, with method scores paired within each query.

Because the benchmark is synthetic and can contain large numbers of examples, practical effect sizes should receive at least as much emphasis as significance tests. The existing practical-tie threshold of (0.05) is useful, but it needs justification and should be accompanied by raw score differences and confidence intervals.


### 10. The task should consistently be framed as candidate-set selection for RAG

The benchmark evaluates retrieval or reranking over an already defined query-local pool. It does not yet evaluate the complete sequence of:

1. first-stage retrieval from a large shared corpus;
2. diversified subset selection;
3. context construction;
4. answer generation;
5. factuality and completeness of the generated answer.

This is a legitimate and useful experimental isolation, but the wording should remain precise. Terms such as “RAG benchmark” can imply end-to-end generation evaluation unless they are qualified.

Suitable descriptions include:

* “a multi-facet retrieval benchmark for RAG”;
* “a controlled candidate-set selection benchmark”;
* “a reranking and subset-selection evaluation for budget-constrained RAG.”

The thesis should avoid claiming direct improvements in answer quality unless a generator-based experiment is added. The current evidence supports improved retrieval coverage and coverage–purity trade-offs, from which downstream benefits are plausible but not guaranteed.

A limited end-to-end experiment would strengthen the work, but it is not essential if the retrieval scope is consistently maintained.

---

## Revised priority order

The most important remaining methodological improvements are:

1. evaluate global and transferable (\lambda) settings;
2. validate and contextualize FCP as a custom metric;
3. add oracle, random, and at least one additional structured baseline;
4. test for template and lexical shortcuts;
5. add paired, clustered uncertainty estimates;
6. clarify standard versus windowed MMR;
7. replace or supplement chunk-level recall;
8. report computational cost;
9. justify the practical-tie threshold;
10. consistently frame the contribution as retrieval and subset selection for RAG.

These issues do not undermine the benchmark’s central premise. They determine how broad and convincing the final empirical claims can be.
