"""Phase 3 heuristic reranker.

Two-stage flow: after FAISS recall, apply lightweight heuristic scoring to
promote passages likely about death/defeat events for the detected entity.
"""

from typing import List, Dict, Any, Optional


class Reranker:
    """Heuristic reranker for event-focused queries."""

    def __init__(self) -> None:
        self.death_keywords = [
            "killed",
            "slew",
            "slain",
            "death",
            "fell",
            "struck",
            "beheaded",
            "cut off",
            "pierced",
            "shot",
        ]

    def _score_chunk(self, chunk: Dict[str, Any], entity: Optional[str], focus_event: bool) -> float:
        base = float(chunk.get("score", 0.0))
        text = chunk.get("text", "").lower()
        score = base

        if entity:
            name = entity.lower()
            if name in text:
                score *= 1.1  # soft boost if entity mentioned

        if focus_event:
            hits = sum(1 for kw in self.death_keywords if kw in text)
            if hits:
                score *= 1.0 + 0.05 * min(hits, 3)  # cap boost

        return score

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        entity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        scored: List[Dict[str, Any]] = []
        for c in candidates:
            new_score = self._score_chunk(c, entity, focus_event=entity is not None)
            item = dict(c)
            item["score"] = new_score
            scored.append(item)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
