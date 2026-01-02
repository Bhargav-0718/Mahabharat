"""Alias resolution and normalization (EVENT-CENTRIC Phase 4).

Maps different names of the same entity to a canonical form.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

# Legacy import for backward compatibility
try:
    from .schemas import EntityMention, ResolvedEntity
    _has_schemas = True
except ImportError:
    _has_schemas = False


def normalize_name(text: str) -> str:
    """Normalize a name for matching.
    
    - lowercase
    - remove punctuation
    - collapse whitespace
    """
    text = re.sub(r"[^a-z0-9\s]", "", text.lower())
    text = " ".join(text.split())
    return text


def _normalize(text: str) -> str:
    """Backward compatibility wrapper."""
    return normalize_name(text)


# Hard-coded alias groups for major characters (EVENT-CENTRIC)
ALIAS_SEEDS = {
    # Pandavas
    "arjuna": ["arjuna", "partha", "dhananjaya", "vibhatsu", "kiritin", "phalguna"],
    "bhima": ["bhima", "bhimasena", "vrikodara"],
    "yudhishthira": ["yudhishthira", "dharma"],
    "nakula": ["nakula"],
    "sahadeva": ["sahadeva"],
    # Kauravas
    "duryodhana": ["duryodhana", "suyodhana"],
    "duhsasana": ["duhsasana"],
    # Elders
    "krishna": ["krishna", "keshava", "vasudeva", "janardana", "madhava", "achyuta"],
    "bhishma": ["bhishma"],
    "drona": ["drona"],
    "karna": ["karna", "radheya", "vasusena"],
    # Women
    "draupadi": ["draupadi", "panchali"],
    "kunti": ["kunti", "pritha"],
    "gandhari": ["gandhari"],
}


class AliasResolver:
    """Resolves aliases to canonical entity names."""

    def __init__(self, alias_map: Dict[str, Dict[str, object]] = None):
        """Initialize resolver.
        
        Can be initialized with:
        1. No args (uses ALIAS_SEEDS)
        2. External alias_map (for backward compatibility)
        """
        if alias_map is not None:
            # Backward compatibility: external alias map provided
            self.alias_map = alias_map
            self.alias_to_entity: Dict[str, str] = {}
            self.entity_types: Dict[str, str] = {}
            for entity_id, meta in alias_map.items():
                etype = str(meta.get("type", "")).upper()
                self.entity_types[entity_id] = etype
                for alias in meta.get("aliases", []) or []:
                    norm = normalize_name(alias)
                    if norm:
                        self.alias_to_entity[norm] = entity_id
        else:
            # New EVENT-CENTRIC initialization using ALIAS_SEEDS
            self.alias_map = alias_map or {}
            self.alias_to_entity: Dict[str, str] = {}
            self.entity_types: Dict[str, str] = {}
            self._init_seed_aliases()

    def _init_seed_aliases(self):
        """Initialize from seed aliases."""
        for canonical, aliases in ALIAS_SEEDS.items():
            canonical_norm = normalize_name(canonical)
            for alias in aliases:
                alias_norm = normalize_name(alias)
                self.alias_to_entity[alias_norm] = canonical_norm

    def resolve(self, mention_text: str) -> Optional[str]:
        """Resolve a name to canonical form.
        
        Args:
            mention_text: Raw text to resolve
            
        Returns:
            Canonical normalized form, or original normalized form if not in map
        """
        norm = normalize_name(mention_text)
        return self.alias_to_entity.get(norm, norm)

    def get_canonical_id(self, name: str, entity_type: str) -> str:
        """Get canonical entity ID from name.
        
        Format: {entity_type}_{canonical_normalized}
        
        Args:
            name: Raw name
            entity_type: PERSON, GROUP, PLACE, TIME
            
        Returns:
            Canonical entity ID
        """
        canonical = self.resolve(name)
        entity_id = f"{entity_type}_{canonical}".lower()
        # Ensure valid identifier
        entity_id = re.sub(r"[^a-z0-9_]", "_", entity_id)
        return entity_id

    # Backward compatibility methods
    def resolve_mentions(self, mentions: List) -> List:
        """Backward compatibility method for legacy code."""
        if not _has_schemas:
            return []
        
        resolved: List = []
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
