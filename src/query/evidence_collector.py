"""Phase 5B (revised): Evidence Collector.

Collects events via GraphExecutor, retrieves semantic chunks (Phase 3 style),
and returns ranked evidence for downstream LLM synthesis.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from graph_executor import QueryExecutor

logger = logging.getLogger(__name__)

# Support execution as a script (python src/query/run_query.py) and make retrieval optional.
if __package__ is None or __package__ == "":
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.append(str(src_dir))

try:  # pragma: no cover - defensive import
    from retrieval.reranker import Reranker
    from retrieval.retriever import Retriever
    from retrieval.phase3_pipeline import build_expansions, detect_death_query, parva_boost_map
    RETRIEVAL_AVAILABLE = True
except Exception as exc:  # pragma: no cover - defensive guard when faiss or assets missing
    logger.warning("Phase 3 retrieval stack unavailable: %s", exc)
    Reranker = None
    Retriever = None
    build_expansions = detect_death_query = parva_boost_map = None
    RETRIEVAL_AVAILABLE = False


class EvidenceCollector:
    """Collect events and lightweight chunks for a query plan."""

    def __init__(
        self,
        executor: QueryExecutor,
        chunk_retriever: Optional[Retriever] = None,
        chunk_reranker: Optional[Reranker] = None,
    ) -> None:
        self.executor = executor
        if not RETRIEVAL_AVAILABLE:
            raise RuntimeError("Phase 3 retrieval stack not available (faiss/index/embeddings missing)")

        if chunk_retriever is not None:
            self.chunk_retriever = chunk_retriever
        elif RETRIEVAL_AVAILABLE and Retriever is not None:
            try:
                self.chunk_retriever = Retriever()
            except Exception as exc:  # pragma: no cover - defensive guard for missing assets
                raise RuntimeError(f"Retriever init failed: {exc}")
        else:
            raise RuntimeError("Retriever not available")

        if chunk_reranker is not None:
            self.chunk_reranker = chunk_reranker
        elif RETRIEVAL_AVAILABLE and Reranker is not None:
            try:
                self.chunk_reranker = Reranker()
            except Exception as exc:  # pragma: no cover - defensive guard
                raise RuntimeError(f"Reranker init failed: {exc}")
        else:
            raise RuntimeError("Reranker not available")

    def _retrieve_chunks(self, question: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """Retrieve semantic chunks via FAISS + heuristic rerank."""
        entity = detect_death_query(question)
        expansions = build_expansions(entity) if entity else None
        parva_boost = parva_boost_map(entity)

        stage1 = self.chunk_retriever.retrieve_expanded(
            query=question,
            expanded_queries=expansions,
            top_k_stage1=30,
            parva_boost=parva_boost,
        )

        if not stage1:
            return []

        reranked = self.chunk_reranker.rerank(
            query=question,
            candidates=stage1,
            top_k=max(top_k, 5),
            entity=entity,
        )
        return reranked[:top_k]

    def collect(self, query_plan: Any, question: str = "") -> Dict[str, Any]:
        """Execute plan, return events and chunks."""
        plan_dict = asdict(query_plan) if hasattr(query_plan, "__dataclass_fields__") else dict(query_plan)
        exec_result = self.executor.execute(plan_dict, question)

        events: List[Dict[str, Any]] = []
        seen_events = set()
        for ev in exec_result.matched_events:
            eid = ev.get("event_id") or ev.get("id")
            if not eid or eid in seen_events:
                continue
            seen_events.add(eid)
            events.append(
                {
                    "event_id": eid,
                    "event_type": ev.get("type") or ev.get("event_type"),
                    "sentence": ev.get("sentence", ""),
                    "participants": ev.get("participants", []),
                    "score": 1.0,  # Placeholder deterministic score
                }
            )

        chunks: List[Dict[str, Any]] = self._retrieve_chunks(question)

        return {
            "query_plan": plan_dict,
            "events": events,
            "chunks": chunks,
        }
