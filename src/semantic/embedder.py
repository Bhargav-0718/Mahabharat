import logging
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wrapper around SentenceTransformer to generate embeddings.

    - Automatically uses GPU on Google Colab if available
    - Falls back to CPU otherwise
    - Safe for large batch embedding in Phase 2
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ) -> None:
        # 🔹 Auto-detect device if not explicitly provided
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name

        logger.info("Initializing embedding model")
        logger.info("Model: %s", model_name)
        logger.info("Device: %s", device)

        self.model = SentenceTransformer(
            model_name,
            device=device
        )

        self.dimension = int(self.model.get_sentence_embedding_dimension())

        logger.info(
            "Loaded embedding model %s (dim=%d) on %s",
            self.model_name,
            self.dimension,
            self.device,
        )

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """
        Embed a list of texts.

        Returns:
            List[np.ndarray] with dtype float32
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,  # ✅ Useful on Colab
            normalize_embeddings=False,
        )

        # SentenceTransformer returns ndarray for list input
        return [np.asarray(vec, dtype=np.float32) for vec in embeddings]

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text.
        """
        emb = self.model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return np.asarray(emb[0], dtype=np.float32)
