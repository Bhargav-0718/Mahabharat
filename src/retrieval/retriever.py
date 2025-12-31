import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

# Support execution as a script (python src/retrieval/phase3_pipeline.py)
if __package__ is None or __package__ == "":
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.append(str(src_dir))
    from semantic.embedder import Embedder
else:
    from semantic.embedder import Embedder
from .faiss_index import FaissIndex

logger = logging.getLogger(__name__)


class Retriever:
    """FAISS-backed retriever for Phase 3."""

    def __init__(
        self,
        chunks_path: Path = Path("data/semantic_chunks/chunks.jsonl"),
        manifest_path: Path = Path("data/semantic_chunks/embedding_manifest.json"),
        index_path: Path = Path("data/retrieval/faiss.index"),
        id_map_path: Path = Path("data/retrieval/id_mapping.json"),
        model_name: str = "jinaai/jina-embeddings-v2-base-en",
        batch_size: int = 32,
        force_rebuild_index: bool = False,
    ) -> None:
        self.chunks_path = chunks_path
        self.batch_size = batch_size
        self.embedder = Embedder(model_name=model_name)
        self.indexer = FaissIndex(manifest_path=manifest_path, index_path=index_path, id_map_path=id_map_path)
        self.index, self.id_map = self.indexer.ensure(force=force_rebuild_index)
        self.chunk_lookup = self._load_chunks(chunks_path)
        self.dimension = self.index.d

    def _load_chunks(self, path: Path) -> Dict[str, Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Chunks file not found: {path}")
        lookup: Dict[str, Dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                cid = row.get("chunk_id")
                if cid:
                    lookup[cid] = row
        return lookup

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self.embedder.embed_text(query)
        vec = np.asarray(vec, dtype=np.float32)
        faiss.normalize_L2(vec.reshape(1, -1))
        return vec

    def _filter_ids(self, candidate_ids: List[str], filters: Optional[Dict[str, Any]]) -> List[str]:
        if not filters:
            return candidate_ids
        filtered: List[str] = []
        for cid in candidate_ids:
            chunk = self.chunk_lookup.get(cid)
            if chunk is None:
                continue
            parva_ok = filters.get("parva_number") is None or chunk.get("parva_number") == filters.get("parva_number")
            section_ok = filters.get("section_number") is None or chunk.get("section_number") == filters.get("section_number")
            if parva_ok and section_ok:
                filtered.append(cid)
        return filtered

    def retrieve_expanded(
        self,
        query: str,
        expanded_queries: Optional[List[str]] = None,
        top_k_stage1: int = 30,
        filters: Optional[Dict[str, Any]] = None,
        parva_boost: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        if not query.strip():
            raise ValueError("Query must be non-empty")

        queries = expanded_queries or [query]
        merged: Dict[str, Dict[str, Any]] = {}

        for q in queries:
            qvec = self._embed_query(q)
            scores, idxs = self.index.search(qvec.reshape(1, -1), top_k_stage1)
            for score, idx in zip(scores.flatten().tolist(), idxs.flatten().tolist()):
                if idx < 0 or idx >= len(self.id_map):
                    continue
                cid = self.id_map[idx]
                chunk = self.chunk_lookup.get(cid)
                if chunk is None:
                    continue

                base_score = float(score)
                if parva_boost and chunk.get("parva_name") in parva_boost:
                    base_score *= parva_boost[chunk["parva_name"]]

                existing = merged.get(cid)
                if existing is None or base_score > existing["score"]:
                    merged[cid] = {
                        "chunk_id": cid,
                        "score": base_score,
                        "text": chunk.get("text", ""),
                        "parva_number": chunk.get("parva_number"),
                        "parva_name": chunk.get("parva_name"),
                        "section_number": chunk.get("section_number"),
                        "section_index": chunk.get("section_index"),
                    }

        results = list(merged.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        if filters:
            filtered_ids = self._filter_ids([r["chunk_id"] for r in results], filters)
            results = [r for r in results if r["chunk_id"] in filtered_ids]
        return results
