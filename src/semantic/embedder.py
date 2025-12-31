import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Wrapper around SentenceTransformer to generate embeddings."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        logger.info("Loaded embedding model %s (dim=%s)", self.model_name, self.dimension)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        # SentenceTransformer returns ndarray for list input.
        if isinstance(embeddings, np.ndarray):
            return [np.asarray(row, dtype=np.float32) for row in embeddings]
        return [np.asarray(vec, dtype=np.float32) for vec in embeddings]

    def embed_text(self, text: str) -> np.ndarray:
        emb = self.model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return np.asarray(emb[0], dtype=np.float32)
