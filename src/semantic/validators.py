import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class ChunkValidator:
    REQUIRED_FIELDS = {
        "chunk_id",
        "parva_number",
        "parva_name",
        "section_number",
        "section_index",
        "chunk_index",
        "text",
        "token_count",
        "source",
        "language",
    }

    @staticmethod
    def validate_chunks(
        chunks: List[Dict],
        min_tokens: int,
        max_tokens: int,
    ) -> None:
        seen_ids = set()
        soft_min = int(0.7 * min_tokens)
        absolute_floor = 40
        soft_min_count = 0
        
        for chunk in chunks:
            missing = ChunkValidator.REQUIRED_FIELDS - set(chunk.keys())
            if missing:
                raise ValueError(f"Chunk {chunk.get('chunk_id')} missing fields: {missing}")

            cid = chunk.get("chunk_id")
            if cid in seen_ids:
                raise ValueError(f"Duplicate chunk_id detected: {cid}")
            seen_ids.add(cid)

            tokens = int(chunk.get("token_count", 0))
            
            # Soft minimum logic matching chunker
            if tokens < absolute_floor:
                raise ValueError(f"Chunk {cid} below absolute floor: {tokens} < {absolute_floor}")
            elif tokens < soft_min:
                logger.warning(f"Chunk {cid} below soft minimum: {tokens} < {soft_min}")
                soft_min_count += 1
            elif tokens < min_tokens:
                soft_min_count += 1
            
            if tokens > max_tokens:
                raise ValueError(f"Chunk {cid} exceeds maximum tokens: {tokens} > {max_tokens}")

            text = chunk.get("text", "").strip()
            if not text:
                raise ValueError(f"Chunk {cid} has empty text")

    @staticmethod
    def validate_embeddings(
        chunks: List[Dict],
        embeddings: List,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Embedding count {len(embeddings)} does not match chunk count {len(chunks)}"
            )

    @staticmethod
    def log_stats(chunks: List[Dict]) -> None:
        if not chunks:
            return
        token_counts = [c.get("token_count", 0) for c in chunks]
        logger.warning(
            "Chunk stats: count=%s, min=%s, max=%s, avg=%.1f",
            len(chunks),
            min(token_counts),
            max(token_counts),
            sum(token_counts) / len(token_counts),
        )
