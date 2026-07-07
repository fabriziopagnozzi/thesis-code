## Considerazioni generiche MIMIC
- What about different formulations for the coverage term, more suitable for high dimensional spaces?

- Nel retrieval quando sarà embeddato tutto il corpus, la coverage non rischia di aggiungere rumore da data points vicini all'embedding della ma non rappresentativi effettivamente degli aspetti specificati nella query? oppure la struttura stessa dello spazio di embedding già risolve questo problema?

- per le domande generate sono tantissimi i documenti, si può fare un topic modeling per vedere se il numero di argomenti che coprono le domande si ritrova --> questo viene dopo, non concentrartici adesso.

- variare la composizione del pool dell'evaluation --> injectare noise da altre conditions.

- implementare hybrid search (keyword + semantic) come relevance score

- le query per come sono adesso forse beneficiano dell'embedding del contextual prefix, non del contenuto stesso delle note: retrieval gerarchico prima sul prefix e poi sulle note associate rifacciamo il retrieval? strano


## General (old)
- 10 paragrafi su ogni Query: troppo pochi per valutare l'efficacia della coverage? i risultati sono statisticamente rilevanti?

- per stabilire se la differenza fra coverage e baseline è statisticamente significativa si usa qualche test statistico? t-student, z-score per trovare p-value?

- formulazioni alternative per coverage che funzionano meglio ad alte dim. rispetto alla query? visto paper di Jagadish che considera coni di angolo theta, magari avrebbe senso fare una "sector coverage" e misurare se la Jaccard è molto diversa rispetto alla coverage normale --> contro: altro iperparametro theta da settare, poco explainable

- provare su dataset: 
	- QA:
		* MIMIC-IV-note
		* HotpotQA distractor
		* 2WikiMultihopQA
			- per i due sopra: embeddare full corpus da wiki? così c'è più spazio per testare la coverage --> problema: performance. Necessità di farlo in step re-ranking
		* MuSiQue (ans)