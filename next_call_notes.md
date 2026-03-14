# Note per prossima call

## Paper "Diversity Enhances LLM's Performance in RAG" — critica uso distanza euclidea in FPS

- Distance concentration (Beyer et al., 1999): in alta dimensionalità le distanze euclidee convergono, "farthest point" perde significato
- Per embedding normalizzati (E5): euclidea e coseno danno lo stesso ranking, la distinzione del paper è fittizia
- Per embedding non normalizzati (SentenceBERT): euclidea cattura magnitudine, semanticamente irrilevante — coseno è meglio
- Coseno resiste meglio alla curse of dimensionality (proietta su sfera unitaria, ignora dimensioni rumorose)
- Il paper stesso lo conferma: MMR (coseno) batte FPS (euclidea), e applicano PCA per ridurre dimensionalità senza discutere il perché teorico
- FPS nasce per point clouds 3D dove euclidea ha senso fisico — trasferirlo a embedding 768-1024d senza giustificazione è una debolezza
