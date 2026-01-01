"""Entity recognition using spaCy NER."""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import spacy
from spacy.tokens import Doc

from .schemas import EntityMention

logger = logging.getLogger(__name__)

NLP_CACHE: Optional[object] = None


def get_nlp():
    global NLP_CACHE
    if NLP_CACHE is None:
        try:
            NLP_CACHE = spacy.load("en_core_web_sm")
        except OSError:
            logger.error("spaCy model en_core_web_sm not found. Install: python -m spacy download en_core_web_sm")
            raise
    return NLP_CACHE


class EntityRecognizer:
    """Extract entities using spaCy NER + heuristics."""

    def __init__(self) -> None:
        self.nlp = get_nlp()

    def _normalize(self, text: str) -> str:
        """Normalize for matching."""
        return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()

    def _capitalize_proper(self, text: str) -> str:
        """Title case for canonical names."""
        return " ".join(word.capitalize() for word in text.split())

    def _extract_spacy_entities(self, text: str, chunk_id: str) -> List[EntityMention]:
        """Use spaCy NER to extract entities."""
        if not text:
            return []
        doc = self.nlp(text)
        mentions: List[EntityMention] = []
        for ent in doc.ents:
            label = ent.label_
            etype = self._map_spacy_label(label)
            if not etype:
                continue
            mentions.append(
                EntityMention(
                    text=ent.text,
                    type=etype,
                    chunk_id=chunk_id,
                    start=ent.start_char,
                    end=ent.end_char,
                )
            )
        return mentions

    def _extract_proper_nouns(self, text: str, chunk_id: str, exclude_spans: List[Tuple[int, int]]) -> List[EntityMention]:
        """Fallback: extract capitalized proper nouns not already captured by spaCy.
        
        Only include if POS is PROPN (strict) and at least 3 chars.
        Note: This is a fallback; main extraction is spaCy-driven.
        """
        doc = self.nlp(text)
        mentions: List[EntityMention] = []
        for token in doc:
            # Strict: only include if marked PROPN by spaCy, not generic NOUN
            if token.pos_ != "PROPN":
                continue
            if len(token.text) < 3:
                continue
            overlap = any(token.idx < e and token.idx + len(token.text) > s for s, e in exclude_spans)
            if overlap:
                continue
            mentions.append(
                EntityMention(
                    text=token.text,
                    type="PERSON",
                    chunk_id=chunk_id,
                    start=token.idx,
                    end=token.idx + len(token.text),
                )
            )
        return mentions

    def _map_spacy_label(self, label: str) -> Optional[str]:
        """Map spaCy label to our entity type."""
        mapping = {
            "PERSON": "PERSON",
            "GPE": "PLACE",
            "ORG": "GROUP",
            "PRODUCT": "WEAPON",
            "EVENT": "TIME",
        }
        return mapping.get(label)

    def extract(self, chunk_id: str, text: str) -> List[EntityMention]:
        """Extract all entities from text."""
        spacy_mentions = self._extract_spacy_entities(text, chunk_id)
        exclude_spans = [(m.start or 0, m.end or 0) for m in spacy_mentions]
        proper_mentions = self._extract_proper_nouns(text, chunk_id, exclude_spans)
        all_mentions = spacy_mentions + proper_mentions
        # Deduplicate by text/type/position
        seen = set()
        uniq: List[EntityMention] = []
        for m in all_mentions:
            key = (m.text, m.type, m.start, m.end)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(m)
        return uniq
