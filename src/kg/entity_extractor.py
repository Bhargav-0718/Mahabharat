"""Rule-based entity extraction for Phase 4."""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

from .schemas import EntityMention


class EntityExtractor:
    def __init__(self, alias_map: Dict[str, Dict[str, object]]) -> None:
        self.alias_map = alias_map
        self.alias_lookup: List[Tuple[str, str]] = []  # (alias_lower, type)
        for entity_id, meta in alias_map.items():
            aliases = meta.get("aliases", []) or []
            etype = str(meta.get("type", "")).upper()
            for alias in aliases:
                alias_norm = alias.lower().strip()
                if alias_norm:
                    self.alias_lookup.append((alias_norm, etype))
        # Sort longer aliases first to prefer multi-word matches
        self.alias_lookup.sort(key=lambda x: len(x[0]), reverse=True)

    def _gazetteer_matches(self, chunk_id: str, text: str) -> List[EntityMention]:
        found: List[EntityMention] = []
        lower = text.lower()
        for alias, etype in self.alias_lookup:
            pattern = rf"\b{re.escape(alias)}\b"
            for m in re.finditer(pattern, lower):
                span_text = text[m.start() : m.end()]
                found.append(
                    EntityMention(
                        text=span_text,
                        type=etype,
                        chunk_id=chunk_id,
                        start=m.start(),
                        end=m.end(),
                    )
                )
        return found

    def _proper_noun_matches(self, chunk_id: str, text: str, already: Iterable[Tuple[int, int]]) -> List[EntityMention]:
        taken = list(already)
        results: List[EntityMention] = []
        for m in re.finditer(r"\b([A-Z][a-zA-Z']+(?:\s+[A-Z][a-zA-Z']+)*)\b", text):
            if any(m.start() < e and m.end() > s for s, e in taken):
                continue
            span = m.group(1)
            if len(span) < 3:
                continue
            results.append(
                EntityMention(
                    text=span,
                    type="PERSON",
                    chunk_id=chunk_id,
                    start=m.start(),
                    end=m.end(),
                )
            )
        return results

    def extract(self, chunk_id: str, text: str) -> List[EntityMention]:
        if not text:
            return []
        gaz = self._gazetteer_matches(chunk_id, text)
        taken_spans = [(m.start or 0, m.end or 0) for m in gaz]
        proper = self._proper_noun_matches(chunk_id, text, taken_spans)
        mentions = gaz + proper
        # Deduplicate identical spans/text/type
        seen = set()
        uniq: List[EntityMention] = []
        for m in mentions:
            key = (m.text.lower(), m.type, m.start, m.end)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(m)
        return uniq
