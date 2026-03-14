## APPUNTI RAW CALL - EMILIA, MARTINENGHI - 06/03/2026
TESINA 2 - appunti della call:
- coniugare ranking top k + aspetto ortogonale al ranking, ovvero la diversificazione dei risultati *> trade off fra i due.
- come? COVERAGE: sel. punti rappresentativi a query time *> quanto bene coprono questi punti restituiti il dataset? minimizzare distanza cumulativa dagli altri punti del dataset. 
- altra possibilità: DISPERSIONE che vuole che gli oggetti selezionati siano piu lontani gli uni dagli altri
- Trade off fra utilità (similarità) e diversità (coverage, disperione).
- problema: lavoro di ricerca e creazione sui dataset
- Concetti spendibili anche nel contesto dei Modelli LLM, mentre Tesina 1 è focalizzata su ML classico
- Il principale focus di ricerca del prof sono le top k queries. Il paper nelle key references è scritto anche in collaborazione col prof (Martinenghi).
- In collaborazione col prof dell'università del Michigan

- corpus documenti molto grande, budget limitato in cui selezionare (per via di context window limitata e context rot)
- nuove tecniche e funzioni per ranking per il RAG in cui combinare similarità, coverage, dispersione.
- due task principali:
    - task 1 QA
    Questo è il task classico del RAG: hai una domanda, recuperi documenti rilevanti, li dai all'LLM, e valuti la risposta. La ground truth è la risposta corretta. Qui la diversificazione ha un senso intuitivo: se la domanda è complessa o multi-aspetto, documenti diversi possono fornire pezzi diversi della risposta.
    Benchmark noti: Natural Questions, HotpotQA, TriviaQA, ecc. — sono abbondanti e ben definiti.

    - task 2 document ranking
    Qui il task è diverso: dato un corpus e una query, ordinare i documenti per rilevanza. Non c'è un LLM che genera una risposta — si valuta direttamente la qualità del ranking prodotto.
    Similarità basata su un set di docs etichettati come importanti *> difficile trovare dati etichettati o esempi di doc importanti e ordinati per rilevanza. Oltre a benchmark pronti, ancora piu difficile è trovare dataset in cui può aver senso avere questo approccio, non è detto che io debba puntare alla diversità.
    La "similarità basata su un set di docs etichettati come importanti" significa che per valutare il ranking ti serve una ground truth di rilevanza: per ogni query, un insieme di documenti giudicati rilevanti (tipicamente con label tipo 0/1/2/3 per grado di rilevanza). Benchmark classici: TREC, MS MARCO, BEIR.


## APPUNTI RAW CALL - EMILIA -  11/03/2026
- Idea su cui si basa la direzione della ricerca: fairness + ranking QUINDI query direzionali + diversificazione.
ha una bozza di un paper da parte dei due prof, c'era un'idea originaria che però è stata un po' rivista in corso d'opera.

- in che contesto applicativo? quello del retrieval nel RAG, perché tutti ultimamente fanno ricerca su questo. 

concentriamoci diversificazione ranking, affrontato sotto diversi punti di vista per i recommender systems *> cosa nuova: molto spesso questi approcci usano la dispersione, quanto sono distanti fra loro i punti.
	- loro dicono no: in alcuni scenari è meglio la coverage. *> facility coverage. Ok in quale ambito dobbiamo ancora studiare la diversificazione? RAG! articoli un po' su Arxiv. 
	- riviste e conferenze peer-reviewed. Arxiv: chiunque può pubblicare un lavoro, ma non è stato ancora accettato e peer reviewed.
	- ha trovato un paio di articoli sulla diversificazione in ambito rag, partendo dalla fase di retrieval. 
	- ha cominciato a fare sperimenti con dataset dei paper, non si trova coi risultati del paper. quando mette assieme utility + coverage trova che i due approcci hanno delle minime differenze, non statisticamente signficiative. 
	- come fare? ragionare su caratteristiche dataset. ci serve un dataset, magari proviamo a farcelo anche noi (uno degli obiettivi della tesi)
  - lei questa settimana è incasinata
  - solo retrieval time per ora, gli esperimenti non vanno a generare una risposta via LLM ma si basano sul ranking dei chunk recuperati.

	Primo obiettivo:
		- creare dataset e benchmark per dimostrare che la coverage a volte funziona meglio
			* creati manualmente, non è detto che sia totalmente sintetico, si trova un corpus e su quel corpus (documenti) mi creo delle domande e delle risposte che rispettino determinate caratteristiche.
			* dovrebbe per ora capire il dataset che serve, adatto a lavorare con la coverage che abbiano diversi aspetti coperti in tutto il dataset.
			* task QA: confrontava il pezzo di doc recuperato e la risposta, ma la risposta era molto puntuale

			TASK
			* 2 paper: guardo i dataset, lei ha trovato un dataset da un paper per QA e un altro multi-hop, e poi altri dataset linkati dal prof del michigan, magari era meglio provare su altri perché come detto prima lei non si trova tanto coi suoi esperimenti con i risultati del prof.
			* problema: limiti dei dataset? ma anche in generale! Alcuni dataset erano enormi ma poi in realtà quelli etichettati in cui sapeva i docs importanti erano 40 query, quindi non c'è un benchmark affidabile

			* guarda anche le reference se qualcosa non è chiaro l'articolo. Negli appunti del paper finale ci sono le reference di Martinenghi e Jagadish da cui è originata questa idea, ma si sono allontanati un po' dall'idea originale.

      - altra idea: esplorare un dataset medico
      "In  tal caso, confermo che MIMIC potrebbe essere un buon punto di partenza, perché questo dataset di note: https://www.physionet.org/content/mimic-iv-note/2.2/ dovrebbe essere collegato (a quanto dicono) al dataset principale di MIMIC, che contiene anche informazioni sulle diagnosi accertate dei pazienti: https://physionet.org/content/mimiciv/3.1/"
        * coverage, associabile alla fairness
        * in ambito medico è un buon caso applicativo
        * si possono fare domande tipo: effetti per gruppo, per genere, possono venire naturalmente domande multi-aspetto. 
        * Task: inizia a vedere sta roba come ultima cosa, per avere l'accesso ai dataset richiede registrazione, del training con diverse domande etc. è un po' più rognoso.
