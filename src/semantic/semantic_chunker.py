import logging
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from .embedder import Embedder
from .paragraph_normalizer import ParagraphNormalizer

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Deterministic chunker that merges paragraphs with semantic awareness."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        embedder: Embedder,
        target_tokens: int = 450,
        min_tokens: int = 120,
        max_tokens: int = 800,
        similarity_threshold: float = 0.35,
    ) -> None:
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.similarity_threshold = similarity_threshold

    def _token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def chunk_parvas(self, parvas: List[Dict]) -> List[Dict]:
        chunks: List[Dict] = []
        for parva in tqdm(parvas, desc="Chunking Parvas", unit="parva"):
            chunks.extend(self._chunk_parva(parva))
        return chunks

    def _chunk_parva(self, parva: Dict) -> List[Dict]:
        parva_number = parva.get("parva_number")
        parva_name = parva.get("parva_name")
        sections = parva.get("sections", [])
        parva_chunks: List[Dict] = []

        for section_idx, section in enumerate(tqdm(sections, desc=f"  {parva_name}", leave=False, unit="section"), start=1):
            section_number = section.get("section_number")
            paragraphs = section.get("paragraphs", [])
            normalized = ParagraphNormalizer.normalize_paragraphs(paragraphs)
            if not normalized:
                continue

            section_chunks = self._chunk_section(
                parva_number=parva_number,
                parva_name=parva_name,
                section_number=section_number,
                section_index=section_idx,
                paragraphs=normalized,
            )
            parva_chunks.extend(section_chunks)

        return parva_chunks

    def _chunk_section(
        self,
        parva_number: int,
        parva_name: str,
        section_number: str,
        section_index: int,
        paragraphs: List[str],
    ) -> List[Dict]:
        chunks: List[Dict] = []
        current_paras: List[str] = []
        current_tokens = 0
        prev_emb: np.ndarray = None

        expanded_paras: List[str] = []
        for para in tqdm(paragraphs, desc="  Normalizing", leave=False, unit="para"):
            tokens = self._token_count(para)
            if tokens > self.max_tokens:
                expanded_paras.extend(self._split_long_paragraph(para))
            else:
                expanded_paras.append(para)

        for para in expanded_paras:
            para_tokens = self._token_count(para)
            if para_tokens == 0:
                continue

            para_emb = self.embedder.embed_text(para)
            similarity = self._cosine(prev_emb, para_emb) if prev_emb is not None else 1.0

            will_exceed_max = current_tokens + para_tokens > self.max_tokens
            reached_target = current_tokens >= self.target_tokens
            sharp_drop = similarity < self.similarity_threshold
            if will_exceed_max:
                if current_paras:
                    chunks.append(
                        self._finalize_chunk(
                            current_paras,
                            parva_number,
                            parva_name,
                            section_number,
                            section_index,
                            len(chunks) + 1,
                        )
                    )
                current_paras = [para]
                current_tokens = para_tokens
                prev_emb = para_emb
                continue

            should_split = reached_target and sharp_drop

            if should_split and current_tokens >= self.min_tokens:
                chunks.append(
                    self._finalize_chunk(
                        current_paras,
                        parva_number,
                        parva_name,
                        section_number,
                        section_index,
                        len(chunks) + 1,
                    )
                )
                current_paras = []
                current_tokens = 0
                prev_emb = None

            current_paras.append(para)
            current_tokens += para_tokens
            prev_emb = para_emb

        if current_paras:
            chunks.append(
                self._finalize_chunk(
                    current_paras,
                    parva_number,
                    parva_name,
                    section_number,
                    section_index,
                    len(chunks) + 1,
                )
            )

        chunks = self._merge_small_chunks(chunks)
        return chunks

    def _merge_small_chunks(self, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return chunks

        merged: List[Dict] = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            if chunk["token_count"] < self.min_tokens:
                # Try merge with previous if possible
                if merged:
                    prev = merged[-1]
                    combined_tokens = prev["token_count"] + chunk["token_count"]
                    if combined_tokens <= self.max_tokens:
                        prev["text"] = prev["text"] + "\n\n" + chunk["text"]
                        prev["token_count"] = combined_tokens
                        i += 1
                        continue
                # Else try merge forward
                if i + 1 < len(chunks):
                    nxt = chunks[i + 1]
                    combined_tokens = chunk["token_count"] + nxt["token_count"]
                    if combined_tokens <= self.max_tokens:
                        merged_chunk = chunk.copy()
                        merged_chunk["text"] = chunk["text"] + "\n\n" + nxt["text"]
                        merged_chunk["token_count"] = combined_tokens
                        merged.append(merged_chunk)
                        i += 2
                        continue
            merged.append(chunk)
            i += 1

        # Final check to ensure min_tokens constraint (warn but allow if unmerge-able)
        for chunk in merged:
            if chunk["token_count"] < self.min_tokens:
                logger.warning(
                    f"Chunk {chunk.get('chunk_id', 'unknown')} has {chunk['token_count']} tokens, "
                    f"below minimum {self.min_tokens} (could not merge without exceeding max)"
                )

        # Renumber chunk indices and ids deterministically after merges
        for idx, chunk in enumerate(merged, start=1):
            chunk["chunk_index"] = idx
            chunk["chunk_id"] = f"P{int(chunk['parva_number']):02d}-S{int(chunk['section_index']):03d}-C{idx:03d}"

        return merged

    def _finalize_chunk(
        self,
        paragraphs: List[str],
        parva_number: int,
        parva_name: str,
        section_number: str,
        section_index: int,
        chunk_index: int,
    ) -> Dict:
        text = "\n\n".join(paragraphs).strip()
        token_count = self._token_count(text)
        chunk_id = f"P{int(parva_number):02d}-S{int(section_index):03d}-C{int(chunk_index):03d}"
        return {
            "chunk_id": chunk_id,
            "parva_number": parva_number,
            "parva_name": parva_name,
            "section_number": section_number,
            "section_index": section_index,
            "chunk_index": chunk_index,
            "text": text,
            "token_count": token_count,
            "source": "KM Ganguly",
            "language": "English",
        }

    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        # Split by sentence boundaries to keep pieces under the hard max token limit.
        import re

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        splits: List[str] = []
        buffer: List[str] = []
        buffer_tokens = 0
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_tokens = self._token_count(sent)
            if sent_tokens > self.max_tokens:
                # As a last resort, hard cut the sentence to respect the ceiling.
                words = sent.split()
                current: List[str] = []
                current_tokens = 0
                for word in words:
                    word_tokens = self._token_count(word)
                    if current_tokens + word_tokens > self.max_tokens:
                        if current:
                            splits.append(" ".join(current))
                        current = [word]
                        current_tokens = word_tokens
                    else:
                        current.append(word)
                        current_tokens += word_tokens
                if current:
                    splits.append(" ".join(current))
                buffer = []
                buffer_tokens = 0
                continue

            if buffer_tokens + sent_tokens > self.max_tokens:
                if buffer:
                    splits.append(" ".join(buffer))
                buffer = [sent]
                buffer_tokens = sent_tokens
            else:
                buffer.append(sent)
                buffer_tokens += sent_tokens

        if buffer:
            splits.append(" ".join(buffer))

        return splits if splits else [paragraph]
