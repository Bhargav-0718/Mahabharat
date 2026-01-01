"""Alias resolution for Phase 4."""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .schemas import EntityMention, ResolvedEntity


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


class AliasResolver:
    def __init__(self, alias_map: Dict[str, Dict[str, object]]) -> None:
        self.alias_map = alias_map
        self.alias_to_entity: Dict[str, str] = {}
        self.entity_types: Dict[str, str] = {}
        for entity_id, meta in alias_map.items():
            etype = str(meta.get("type", "")).upper()
            self.entity_types[entity_id] = etype
            for alias in meta.get("aliases", []) or []:
                norm = _normalize(alias)
                if norm:
                    self.alias_to_entity[norm] = entity_id

    def resolve(self, mention_text: str) -> Optional[str]:
        norm = _normalize(mention_text)
        return self.alias_to_entity.get(norm)

    def resolve_mentions(self, mentions: List[EntityMention]) -> List[ResolvedEntity]:
        resolved: List[ResolvedEntity] = []
        for m in mentions:
            eid = self.resolve(m.text)
            if not eid:
                continue
            etype = self.entity_types.get(eid, m.type)
            resolved.append(
                ResolvedEntity(
                    mention=m.text,
                    entity_id=eid,
                    type=etype,
                    chunk_id=m.chunk_id,
                    start=m.start,
                    end=m.end,
                )
            )
        return resolved
