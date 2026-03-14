# TESINA: Adding Diversity in Directional Query

- **Collaboration** with Prof. H V Jagadish (University of Michigan, USA)
- **Background:** Directional queries and diversified ranking in high-dimensional spaces.
- **Project objectives:**
  - Identify application scenarios where directional preferences and diversity requirements naturally arise.
  - Study the trade-offs between directional relevance and diversity in ranked result sets.
  - Explore modeling choices and algorithmic strategies for combining directional and diversity criteria in a unified query framework.
- **Examples of thesis outcomes:**
  - Formal definition and analysis of directional-diversity problems and their properties.
  - Design and implementation of exact and heuristic algorithms (e.g., improved greedy, branch-and-bound, "diverse neighbors" variants).
  - Experimental evaluation on synthetic/real datasets: trade-offs between proximity to the query, attribute balance, and diversity.
- **Key references:**
  - Ciaccia and Martinenghi - *Directional queries: Making top-k queries more effective in discovering relevant results*, Proc. ACM Management of Data, 2024
  - Guo et al. – *Finding diverse neighbors in high dimensional space*, ICDE 2018.


# Diversification: use-cases
## Non-RAG
- Dataset clusterizzato: quando i dati si addensano attorno alla preference line o alla curva di livello della funzione obiettivo, i top-k risultati collassano in punti quasi identici → il top-k diventa un top-1 di fatto
- RACCOMANDAZIONE PRODOTTI: un e-commerce che mostra i top-10 risultati per "scarpe da running" — se sono tutte Nike nere, l'utente non esplora alternative. Diversificare per brand, colore, fascia di prezzo aumenta la probabilità di acquisto
- RICERCA HOTEL/VOLI: l'esempio del paper di Martinenghi — se cerchi hotel bilanciando prezzo e distanza dal centro, il top-k lineare restituisce solo hotel di lusso in centro o budget in periferia, perdendo i compromessi intermedi
- PORTFOLIO SELECTION: selezionare k asset finanziari — vuoi massimizzare rendimento atteso ma anche diversificare per ridurre il rischio (settori diversi, geografie diverse)
- SENSOR PLACEMENT: posizionare k sensori in un'area — vuoi che coprano il territorio nel modo più uniforme possibile (coverage pura)
- SEARCH ENGINE RESULTS: Google stesso diversifica i risultati — per "apple" mostra sia l'azienda che il frutto, anche se l'azienda ha più pagine rilevanti
- HIRING / SELEZIONE DI CANDIDATI: da un pool di CV top-ranked, selezionare un team con competenze complementari, non 5 persone identiche
## RAG
- QUERY AMBIGUE O MULTI-ASPECT: "Java" (linguaggio, isola, caffè), "effetti del cambiamento climatico" (ambientali, economici, sanitari, geopolitici) → il retriever collassa su una sola interpretazione
- SUMMARIZATION: generare un riassunto su un argomento ampio richiede documenti da sotto-temi diversi, altrimenti il riassunto è parziale
- LONG-TAIL INFORMATION: in un corpus medico, "trattamenti per il diabete" restituisce N documenti su metformina (il più comune), trattamenti meno diffusi ma rilevanti finiscono fuori dal top-k.


# Glossary
- Directional queries: Top-k queries where the scoring function combines a traditional utility component (weighted sum of attribute values) with a "balance" component measuring closeness to a preference line in the attribute space. Instead of just ranking by a linear score, you also care that results are well-balanced across attributes. From Ciaccia & Martinenghi 2024.

- Preference line: A ray from the origin in attribute space defined by a weight vector w. Points close to this line are "balanced" — they satisfy the user's desired trade-off between attributes (e.g., price vs. distance for hotels). Directional queries penalize points far from it.

- Diversity in ranking: Given the top-k results, they might all be nearly identical (clustered). Diversity ensures the k selected items are spread out, so the user sees a range of options. Two main families:
	- Dispersion: Diversity measured as distance between selected points. You want the selected set to be spread apart. Example: Div(S) = min_{i≠j} dist(p_i, p_j) — maximize the minimum pairwise distance. MMR is a dispersion-flavored approach. The k-dispersion problem is NP-hard in general.
	- Coverage (facility-location): Diversity measured by how well the selected set represents the entire dataset. Every point in the full dataset D should have at least one "close representative" in the selected set S. Formally: g(S) = Σ_{i∈D} max_{j∈S} w_ij, where w_ij is similarity. This is monotone submodular — the greedy algorithm gives a (1-1/e) ≈ 0.632 approximation guarantee. This is the notion the thesis argues is underexplored.
	- NOTE: Why this distinction matters — a simple example:
		Imagine 100 points. 90 are clustered tightly in region A, 10 are spread across region B. You need to pick k=3.
		* Dispersion picks one point from each "extreme" — it maximizes spread. It might pick 1 from A, and 2 from the fringes of B. It doesn't care that 90% of the data is in A.
		* Coverage picks to minimize the maximum distance from any point to its nearest representative. It likely picks 1–2 from A (to cover the dense cluster) and 1 from B. It respects the data distribution.

- Submodularity: A set function property meaning "diminishing returns" — adding an element to a smaller set gives at least as much marginal gain as adding it to a larger set. Important because it guarantees the greedy algorithm is near-optimal (1 - 1/e factor). Facility-location coverage is submodular; dispersion is NOT.

- MMR (Maximal Marginal Relevance): A greedy algorithm (Carbonell & Goldstein 1998) that selects the next item by balancing relevance to the query against dissimilarity to already-selected items: argmax [α·relevance(i) - (1-α)·max_{j∈S} similarity(i,j)]. It's the de facto diversity method in RAG/IR.

- gMMR (geometric MMR): DF-RAG's variant that replaces cosine similarity with Euclidean distance between normalized embeddings: gMMR(c) = λ·cos(q,c) + (1-λ)·√(2-2cos(c, c_S)), where c_S is the CENTROID of already-selected chunks, and this is the MAIN difference. Empirically outperforms classical MMR.

- FPS (Farthest Point Sampling): Pure dispersion — iteratively select the point farthest from all already-selected points. No relevance component. From 3D computer vision. Slower than MMR (uses Euclidean distance vs cosine).

- Jaccard similarity: J(A,B) = |A∩B| / |A∪B|. Used in the draft to measure how much two selection strategies overlap. High Jaccard (~0.9) means they pick nearly the same chunks — diversity isn't changing anything. Low Jaccard (~0.5) means genuinely different selections.

- Pre-LLM recall (hit rate): Whether the correct answer appears in the selected chunks before they're sent to the LLM. This is the retrieval-time metric your thesis focuses on.
DIFFERENT from post-LLM recall (whether the LLM's generated answer is correct).

- Multi-hop query: A question requiring information from 2+ distinct documents to answer. E.g., "Which case was brought to court first, Miller v. California or Gates v. Collier?" — you need the date of each case from separate sources. This is where diversity in retrieval matters most.

- Aspect recall: Fraction of distinct information facets required by a query that appear in the selected chunks. Not a standard metric yet — this would be a thesis contribution.

# The Big Picture
Your thesis lives at the intersection of three areas. Let me walk you through them bottom-up, building the intuition layer by layer.

## Layer 1: The Retrieval Problem in RAG
A RAG pipeline has a simple structure: given a user query, retrieve relevant text chunks from a corpus, stuff them into an LLM's context window, and generate an answer.

The standard approach: embed the query, embed all chunks, rank by cosine similarity, take the top-k. This works well when the answer is in one place. But it has a fundamental failure mode: redundancy collapse. If the corpus has 50 chunks about metformin and 3 about insulin therapy, a query about "diabetes treatments" gets 50 metformin chunks at the top. The top-k is effectively a top-1.

This is the problem diversity tries to solve. You want the selected chunks to collectively provide useful information, not individually maximize similarity to the query.

## Layer 2: Two Philosophies of Diversity
This is the conceptual core of the thesis. There are two fundamentally different ways to think about "diverse":

Dispersion asks: "Are the selected items spread apart from each other?"
You look only at the selected set S. You want the items in S to be far from each other. The prototypical measure is the minimum pairwise distance: min_{i≠j ∈ S} dist(i,j). MMR is a greedy heuristic in this family. It's intuitive, widely used, and what everyone defaults to.

Coverage asks: "Does the selected set represent the whole dataset well?"
You look at the relationship between S and the full dataset D. Every item in D should have a close representative in S. The prototypical measure is facility-location: Σ_{i∈D} max_{j∈S} w_ij. It's less intuitive but has stronger theoretical properties.

## Layer 3: Why Doesn't It Work on Standard Benchmarks?
This is the critical question your thesis must address. Emilia's experiments show coverage ≈ baseline on TriviaQA/TREC-DL. Understanding why is more important than just finding a dataset where it works.

The root cause is query structure. Consider what TriviaQA looks like:

Query: "Which oil scandal hit the US in 1924?"
Answer: "Teapot Dome scandal"
The corpus has several chunks mentioning "Teapot Dome scandal." They're all similar to each other and all similar to the query. Whether you pick chunk #1 or chunk #3 doesn't matter — they contain the same answer. Coverage and baseline agree because there's nothing meaningfully different to cover. The information landscape is flat: one peak, one answer.

Now consider a multi-hop query from HotpotQA:
Query: "Which case was brought to court first, Miller v. California or Gates v. Collier?"
To answer this, you need:

Chunk A: "Miller v. California was decided in 1973..."
Chunk B: "Gates v. Collier was filed in 1970..."
These chunks are about different entities. The information landscape has two peaks. A pure-relevance retriever might grab 5 chunks about Miller v. California (more famous, more web pages) and miss Gates v. Collier entirely. Coverage would recognize that region B of the corpus (Gates v. Collier) is unrepresented and select from it.

The thesis claim, distilled: Coverage adds value when the information landscape has multiple relevant peaks that a relevance-only ranker would collapse into one.

## Layer 4: What You Need to Prove
Your thesis needs to establish a chain of reasoning:

Multi-aspect query structure
        ↓
  Clustered information landscape (multiple peaks)
        ↓
  Relevance-only selection collapses to one peak
        ↓
  Coverage-aware selection covers multiple peaks
        ↓
  Better retrieval quality (aspect recall, hit rate)
Each link needs evidence. The dataset/benchmark work is about creating controlled conditions where you can measure each link.

What You Need to Study Deeply
1. Submodular optimization — this is your algorithmic backbone.
Coverage (facility-location) is submodular. This gives you the greedy (1-1/e) guarantee, the branch-and-bound exact algorithm from the draft, and the theoretical foundation for why coverage is preferable to dispersion (which is NP-hard with no approximation guarantee for the greedy). Study:
What submodularity means (diminishing returns)
The Nemhauser-Wolsey-Fisher 1978 theorem (greedy guarantee for monotone submodular maximization under cardinality constraint)
How the marginal gain Δ(j|S) is computed efficiently for facility-location

2. The embedding space and similarity geometry.
Everything depends on how chunks are embedded. Cosine similarity, Euclidean distance on normalized vectors, and the centroid-based distance in gMMR all operate in this space. You need to understand:
Why dense embeddings (E5, multi-qa-mpnet, etc.) tend to cluster relevant chunks together
How PCA affects the geometry (the draft shows PCA hurts — it compresses the space and kills the diversity signal)
The relationship between embedding similarity and semantic similarity — they're correlated but not identical, and the gap matters

3. The RAG evaluation pipeline.
The mechanics of how experiments run — this is what you'll be implementing:
Chunking strategies (chunk size, stride/overlap) and how they affect the candidate pool
Candidate pool construction (how many documents, how many chunks per document)
The difference between pre-LLM recall (answer in selected text) and post-LLM recall (LLM extracts correct answer)
Why your thesis focuses on pre-LLM / retrieval-time metrics — isolating the retrieval component from LLM variability

4. Multi-hop QA datasets and their structure.
These are your experimental testbed. Understand how they're built:
HotpotQA: 2-hop, comes with "distractor" setting (2 relevant + 8 distractor documents) — good controlled setup
MuSiQue: 2–4 hops, harder, compositional
2WikiMultihopQA: 2-hop, derived from Wikidata, very structured
The key property: each query has annotated supporting facts from specific documents. This gives you ground-truth for aspect recall — you know exactly which documents need to be covered.

5. The facility-location vs. MMR comparison — what to measure.

This is the novel experimental contribution:
Same retrieval pipeline, same embeddings, same candidate pool
Swap only the selection strategy: top-k (baseline) vs. MMR/gMMR (dispersion) vs. facility-location coverage
Measure: hit rate, aspect recall (supporting facts recovered), facility-location score, Jaccard divergence from baseline
Analyze: under what query/corpus conditions does coverage diverge from MMR? When does that divergence help?


## The Narrative Arc of Your Thesis
- Diversity matters in RAG — established by Wang et al. and DF-RAG
- Everyone uses dispersion (MMR) — but there's an alternative: coverage
- Coverage has better theoretical properties — submodularity, approximation guarantees
- On standard benchmarks, coverage ≈ baseline — because the queries don't need it
- On multi-aspect/multi-hop queries, coverage diverges — HotpotQA signal from the draft
- Your contribution: characterize when and why coverage outperforms dispersion, using appropriate datasets and a controlled comparison framework


