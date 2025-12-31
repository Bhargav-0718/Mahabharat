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
        below_min_count = 0
        for chunk in chunks:
            missing = ChunkValidator.REQUIRED_FIELDS - set(chunk.keys())
            if missing:
                raise ValueError(f"Chunk {chunk.get('chunk_id')} missing fields: {missing}")

            cid = chunk.get("chunk_id")
            if cid in seen_ids:
                raise ValueError(f"Duplicate chunk_id detected: {cid}")
            seen_ids.add(cid)

            tokens = int(chunk.get("token_count", 0))
            if tokens < min_tokens:
                below_min_count += 1
                logger.warning(f"Chunk {cid} below minimum tokens: {tokens} < {min_tokens}")
            if tokens > max_tokens:
                raise ValueError(f"Chunk {cid} exceeds maximum tokens: {tokens} > {max_tokens}")

            text = chunk.get("text", "").strip()
            if not text:
                raise ValueError(f"Chunk {cid} has empty text")
        
        if below_min_count > 0:
            logger.info(f"Total chunks below minimum: {below_min_count} (allowed as edge cases)")

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
            logger.info("No chunks to report")
            return
        token_counts = [c.get("token_count", 0) for c in chunks]
        logger.info(
            "Chunk stats: count=%s, min=%s, max=%s, avg=%.1f",
            len(chunks),
            min(token_counts),
            max(token_counts),
            sum(token_counts) / len(token_counts),
        )
        per_parva = {}
        for c in chunks:
            name = c.get("parva_name", "Unknown")
            per_parva[name] = per_parva.get(name, 0) + 1
        logger.info("Chunks per parva: %s", per_parva)
