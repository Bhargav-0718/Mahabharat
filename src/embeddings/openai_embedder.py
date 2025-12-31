"""
OpenAI Embedder Module

Generates embeddings for Context Units using OpenAI API.
Uses text-embedding-3-large model.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class OpenAIEmbedder:
    """Generate embeddings using OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-large"):
        """
        Initialize OpenAI embedder.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name (default: text-embedding-3-large)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment or arguments")
        
        self.model = model
        self.embedding_dim = 3072 if model == "text-embedding-3-large" else 1536
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for API calls (max 2048)
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            
            # Sort by index to maintain order
            batch_embeddings = sorted(response.data, key=lambda x: x.index)
            embeddings.extend([item.embedding for item in batch_embeddings])
            
            logger.info(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} texts")
        
        return embeddings

    def embed_context_units(
        self,
        context_units: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Embed Context Units and return with embeddings.
        
        Args:
            context_units: List of Context Unit dictionaries
            
        Returns:
            Context Units with embedding vectors added
        """
        texts = [cu['text'] for cu in context_units]
        embeddings = self.embed_batch(texts)
        
        for cu, embedding in zip(context_units, embeddings):
            cu['embedding'] = embedding
        
        return context_units

    def embed_and_save(
        self,
        input_jsonl_path: str,
        output_jsonl_path: str
    ):
        """
        Embed Context Units from input JSONL and save to output.
        
        Args:
            input_jsonl_path: Path to input JSONL file with Context Units
            output_jsonl_path: Path to output JSONL file with embeddings
        """
        context_units = []
        with open(input_jsonl_path, 'r') as f:
            for line in f:
                context_units.append(json.loads(line))
        
        # Embed
        embedded_units = self.embed_context_units(context_units)
        
        # Save
        output_file = Path(output_jsonl_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for unit in embedded_units:
                f.write(json.dumps(unit) + '\n')
        
        logger.info(f"Saved {len(embedded_units)} embedded units to {output_jsonl_path}")


if __name__ == '__main__':
    # Example usage
    embedder = OpenAIEmbedder()
