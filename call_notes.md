# APPUNTI CALL - EMILIA - 11/03/2026

## Direzione della ricerca

- Idea originaria: fairness + ranking → query direzionali + diversificazione
- Esiste una bozza di paper dei due prof (Martinenghi + Jagadish), l'idea è stata rivista in corso d'opera
- Si sono allontanati un po' dall'idea originale
- Contesto applicativo scelto: **retrieval nel RAG** (tema caldo in ricerca)
- Mi manda un po' di paper

## Tesi chiave: coverage > dispersione

- La diversificazione nel ranking è nota per i recommender systems, ma usa quasi sempre la **dispersione** (distanza tra punti)
- La loro proposta: in alcuni scenari è meglio la **coverage** (facility coverage)
- Ambito in cui la diversificazione è ancora poco studiata: **RAG**
- Nota: articoli trovati su arXiv, non ancora peer-reviewed

## Stato degli esperimenti (Emilia)

- Ha trovato un paio di articoli sulla diversificazione in RAG (fase di retrieval)
- Ha provato a replicare i risultati: **non si trova**. 
- Ha provato con solo utility, solo coverage, o con utility + coverage. Gli esperimenti hanno risultati con differenze minime, non statisticamente significative
- Possibile causa: caratteristiche dei dataset usati non adatte a far emergere il vantaggio della coverage

## Primo obiettivo della tesi: creare dataset e benchmark

- Dimostrare che la coverage a volte funziona meglio della pura utility
- Non necessariamente sintetico: si parte da un corpus reale e si creano domande/risposte con caratteristiche specifiche
- Serve capire che tipo di dataset è adatto: documenti con **diversi aspetti coperti** nel corpus
- Task QA precedente: confronto chunk recuperato vs risposta, ma la risposta era troppo puntuale
- **Solo retrieval time** per ora — non si genera risposta via LLM, si valuta il ranking dei chunk

## Problemi con dataset esistenti

- Emilia ha provato dataset da 2 paper (QA e multi-hop) + dataset dal prof del Michigan
- I risultati non corrispondono a quelli dei paper
- Alcuni dataset erano enormi ma con solo ~40 query etichettate → benchmark non affidabile
- Necessità di provare su dataset diversi

## Idea alternativa: dataset medico (MIMIC)

- MIMIC-IV Note (note cliniche): https://www.physionet.org/content/mimic-iv-note/2.2/
- MIMIC-IV (dati strutturati, diagnosi): https://physionet.org/content/mimiciv/3.1/
- Perché: coverage associabile a fairness, domande naturalmente multi-aspetto (per gruppo, per genere, ecc.)
- Buon caso applicativo per ambito medico
- **Da fare come ultima cosa**: l'accesso richiede registrazione + training obbligatorio

## Prossimi passi

- Leggere i paper che mi manda
- Guardare le reference dei paper per chiarire i concetti
- Capire che caratteristiche deve avere un dataset per far emergere il vantaggio della coverage
- Esplorare dataset MIMIC (bassa priorità)