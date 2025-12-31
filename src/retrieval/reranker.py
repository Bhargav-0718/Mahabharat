"""
Reranker Module

Re-ranks retrieved Context Units using semantic similarity.
Combines graph distance and embedding-based similarity.
"""

import logging
from typing import List, Dict, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


class SemanticReranker:
    """Re-rank Context Units using semantic similarity."""

    def __init__(self, embedding_dim: int = 3072):
        """
        Initialize reranker.
        
        Args:
            embedding_dim: Dimension of embeddings (default 3072 for text-embedding-3-large)
        """
        self.embedding_dim = embedding_dim

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def rerank(
        self,
        query_embedding: List[float],
        context_units: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Re-rank Context Units by semantic similarity to query.
        
        Args:
            query_embedding: Query embedding vector
            context_units: List of Context Unit dictionaries with embeddings
            top_k: Return top K results
            
        Returns:
            List of (unit_id, similarity_score) tuples, sorted by score
        """
        scored_units = []
        
        for unit in context_units:
            if 'embedding' not in unit or unit['embedding'] is None:
                logger.warning(f"No embedding for unit {unit['unit_id']}")
                continue
            
            similarity = self.cosine_similarity(
                query_embedding,
                unit['embedding']
            )
            scored_units.append((unit['unit_id'], similarity))
        
        # Sort by similarity descending
        scored_units.sort(key=lambda x: x[1], reverse=True)
        
        return scored_units[:top_k]

    def hybrid_rerank(
        self,
        query_embedding: List[float],
        context_units: List[Dict[str, Any]],
        graph_scores: Dict[str, float] = None,
        alpha: float = 0.7,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Re-rank using hybrid scoring: semantic similarity + graph distance.
        
        Args:
            query_embedding: Query embedding
            context_units: Context Units with embeddings
            graph_scores: Graph-based relevance scores (0-1)
            alpha: Weight for semantic similarity (1-alpha for graph)
            top_k: Return top K results
            
        Returns:
            List of (unit_id, hybrid_score) tuples
        """
        semantic_scores = self.rerank(query_embedding, context_units, top_k=len(context_units))
        
        # Normalize semantic scores to 0-1
        if semantic_scores:
            max_semantic = max(score for _, score in semantic_scores)
            semantic_dict = {
                uid: (score / max_semantic) if max_semantic > 0 else 0
                for uid, score in semantic_scores
            }
        else:
            semantic_dict = {}
        
        # Compute hybrid scores
        hybrid_scores = []
        for unit in context_units:
            unit_id = unit['unit_id']
            semantic = semantic_dict.get(unit_id, 0.0)
            graph = graph_scores.get(unit_id, 0.0) if graph_scores else 0.0
            
            hybrid = alpha * semantic + (1 - alpha) * graph
            hybrid_scores.append((unit_id, hybrid))
        
        # Sort and return top K
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return hybrid_scores[:top_k]


if __name__ == '__main__':
    # Example usage
    reranker = SemanticReranker()
