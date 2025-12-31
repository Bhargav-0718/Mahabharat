import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FaissIndex:
    """FAISS index builder/loader for Phase 3 retrieval."""

    def __init__(
        self,
        manifest_path: Path = Path("data/semantic_chunks/embedding_manifest.json"),
        index_path: Path = Path("data/retrieval/faiss.index"),
        id_map_path: Path = Path("data/retrieval/id_mapping.json"),
    ) -> None:
        self.manifest_path = manifest_path
        self.index_path = index_path
        self.id_map_path = id_map_path

    def _load_manifest(self) -> Tuple[np.ndarray, List[str], int]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Embedding manifest not found: {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        records = manifest.get("chunks") or []
        if not records:
            raise ValueError("Embedding manifest has no records")

        embeddings = np.array([rec["embedding"] for rec in records], dtype=np.float32)
        chunk_ids = [rec["chunk_id"] for rec in records]
        dim = int(manifest.get("dimension") or embeddings.shape[1])
        return embeddings, chunk_ids, dim

    def build(self, force: bool = False) -> Tuple[faiss.IndexFlatIP, List[str]]:
        if self.index_path.exists() and self.id_map_path.exists() and not force:
            logger.info("Index already exists; skipping rebuild")
            return self.load()

        embeddings, chunk_ids, dim = self._load_manifest()

        if embeddings.shape[1] != dim:
            raise ValueError(f"Embedding dimension mismatch: data={embeddings.shape[1]}, manifest={dim}")

        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        with self.id_map_path.open("w", encoding="utf-8") as f:
            json.dump({"ids": chunk_ids}, f, ensure_ascii=False, indent=2)

        logger.info("Built FAISS index with %d vectors (dim=%d)", len(chunk_ids), dim)
        return index, chunk_ids

    def load(self) -> Tuple[faiss.IndexFlatIP, List[str]]:
        if not self.index_path.exists() or not self.id_map_path.exists():
            raise FileNotFoundError("FAISS index or id mapping not found; build first")
        index = faiss.read_index(str(self.index_path))
        with self.id_map_path.open("r", encoding="utf-8") as f:
            mapping = json.load(f)
        chunk_ids = mapping.get("ids", [])
        if index.ntotal != len(chunk_ids):
            raise ValueError("Index size and id mapping length differ")
        return index, chunk_ids

    def ensure(self, force: bool = False) -> Tuple[faiss.IndexFlatIP, List[str]]:
        if self.index_path.exists() and self.id_map_path.exists() and not force:
            return self.load()
        return self.build(force=force)
