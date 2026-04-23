
import numpy as np
from numpy.typing import NDArray


class Embedder:
    def __init__(
        self, model_name: str, device: str = 'cpu', batch_size: int = 256
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name: str = model_name
        self.device: str = device
        self.batch_size: int = batch_size
        self._model: SentenceTransformer = SentenceTransformer(model_name, device=device)

    @property
    def dim(self) -> int:
        dim = self._model.get_sentence_embedding_dimension()
        assert dim is not None
        return dim

    def embed_corpus(self, texts: list[str], normalize: bool = True) -> NDArray[np.float32]:
        embs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return np.asarray(embs, dtype=np.float32)

    def embed_qa_record(
        self, query: str, chunk_texts: list[str], normalize: bool = True
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Encode query + chunks for a record.
        Returns (query_emb [dim], chunk_embs [n_chunks, dim]).
        """
        all_texts = [query, *chunk_texts]
        all_embs = self._model.encode(
            all_texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return all_embs[0], all_embs[1:]
