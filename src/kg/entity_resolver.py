"""Entity alias resolution via co-occurrence and string similarity clustering."""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Set, Tuple

from tqdm import tqdm

from .schemas import EntityMention

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Normalize for alias matching."""
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


def _string_similarity(a: str, b: str, threshold: float = 0.7) -> float:
    """Compute normalized string similarity."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def sentences_with_mentions(text: str, targets: Set[str]) -> List[Set[str]]:
    """Return list of mention sets per sentence for faster co-occurrence checks."""
    sentences = split_sentences(text)
    sent_sets: List[Set[str]] = []
    for sent in sentences:
        lowered = _normalize(sent)
        hits = {t for t in targets if t in lowered}
        if hits:
            sent_sets.append(hits)
    return sent_sets


class AliasResolver:
    """Resolve entity aliases via sentence co-occurrence and strict string similarity."""

    def __init__(
        self,
        similarity_threshold: float = 0.88,
        min_mention_frequency: int = 2,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.min_mention_frequency = min_mention_frequency
        self.mention_to_canonical: Dict[str, str] = {}
        self.canonical_mentions: Dict[str, List[str]] = defaultdict(list)
        self.mention_chunks: Dict[str, Set[str]] = defaultdict(set)
        self.mention_counts: Dict[str, int] = defaultdict(int)
        self.mention_sentences: Dict[str, Set[str]] = defaultdict(set)
        self.chunk_texts: Dict[str, str] = {}

    def add_mentions(self, mentions: List[EntityMention]) -> None:
        """Register mentions from chunks."""
        for m in mentions:
            norm = _normalize(m.text)
            self.mention_chunks[norm].add(m.chunk_id)
            self.mention_counts[norm] += 1

    def add_chunk_texts(self, chunks: List[Dict[str, object]], target_norms: Set[str]) -> None:
        """Store sentence-level mention sets for co-occurrence analysis."""
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text", "")
            if chunk_id and text:
                # Precompute sentences -> mention hits for only target_norms to speed co-occurrence
                self.chunk_texts[chunk_id] = sentences_with_mentions(text, target_norms)

    def _sentence_cooccurrence(self, mentions_to_test: List[str]) -> Dict[str, Set[str]]:
        """Find mentions that appear in the same pre-indexed sentence."""
        cooccurrence: Dict[str, Set[str]] = defaultdict(set)
        for mention in mentions_to_test:
            m_norm = _normalize(mention)
            chunks_for_mention = self.mention_chunks.get(m_norm, set())
            for chunk_id in chunks_for_mention:
                sent_sets = self.chunk_texts.get(chunk_id, [])
                for hits in sent_sets:
                    if m_norm in hits:
                        for other_norm in hits:
                            if other_norm == m_norm:
                                continue
                            # Map back to original mention strings (best effort: same norm)
                            # We'll just store by norm since canonical_map keys are norms.
                            cooccurrence[mention].add(other_norm)
        return cooccurrence

    def resolve(self, mentions: List[EntityMention], chunks: List[Dict[str, object]]) -> Dict[str, str]:
        """Resolve mentions to canonical forms per type, using strict matching."""
        by_type: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for m in mentions:
            norm = _normalize(m.text)
            by_type[m.type].append((m.text, norm))

        canonical_map: Dict[str, str] = {}
        for etype, type_mentions in tqdm(by_type.items(), desc="Resolving aliases", unit="type", ncols=80):
            unique_mentions = list(set(m[0] for m in type_mentions))
            if len(unique_mentions) == 1:
                canonical = unique_mentions[0]
                for _, norm in type_mentions:
                    canonical_map[norm] = canonical
                continue

            # Filter mentions below minimum frequency
            frequent_mentions = [m for m in unique_mentions if self.mention_counts[_normalize(m)] >= self.min_mention_frequency]
            if not frequent_mentions:
                # If no frequent mentions, keep single instances unmerged
                for m in unique_mentions:
                    canonical_map[_normalize(m)] = m
                continue

            # Build sentence co-occurrence graph
            # Pre-index sentences for the mentions in this type bucket only
            target_norms = {_normalize(m) for m in frequent_mentions}
            self.add_chunk_texts(chunks, target_norms)
            cooccurrence = self._sentence_cooccurrence(frequent_mentions)

            # High-similarity clustering with sentence co-occurrence validation
            clusters: List[List[str]] = []
            assigned = set()
            for i, text in enumerate(frequent_mentions):
                if _normalize(text) in assigned:
                    continue
                cluster = [text]
                for other in frequent_mentions[i + 1 :]:
                    if _normalize(other) in assigned:
                        continue
                    sim = _string_similarity(text, other, self.similarity_threshold)
                    # Require BOTH high similarity AND sentence co-occurrence
                    if sim >= self.similarity_threshold and other in cooccurrence.get(text, set()):
                        cluster.append(other)
                        assigned.add(_normalize(other))
                if len(cluster) > 1:
                    clusters.append(cluster)
                    assigned.add(_normalize(text))

            # Assign canonical to each cluster
            for cluster in clusters:
                canonical = max(cluster, key=lambda x: self.mention_counts[_normalize(x)])
                for mention in cluster:
                    canonical_map[_normalize(mention)] = canonical

            # Unmerged mentions stay as themselves
            unclustered = set(m for m in frequent_mentions if _normalize(m) not in assigned)
            for mention in unclustered:
                canonical_map[_normalize(mention)] = mention

        return canonical_map

    def build_entity_id(self, canonical_mention: str, etype: str) -> str:
        """Generate entity_id from canonical mention and type."""
        norm_form = re.sub(r"[^a-z0-9_]", "_", canonical_mention.lower()).strip("_")
        # Clean up multiple underscores
        norm_form = re.sub(r"_+", "_", norm_form)
        return f"{etype.upper()}_{norm_form}".replace(" ", "_")

