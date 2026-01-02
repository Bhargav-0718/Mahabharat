"""Entity registry with admission control.

Entities can ONLY be created if they participate in detected events.
No abstract entities, no mention-based creation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .alias_resolver import AliasResolver, normalize_name
from .event_extractor import ExtractedEvent, EventArgument

logger = logging.getLogger(__name__)

# Valid entity types
VALID_TYPES = {"PERSON", "GROUP", "PLACE", "TIME"}

# Entity type inference from context
TYPE_HINTS = {
    "PERSON": [
        "krishna", "arjuna", "bhima", "drona", "karna",
        "yudhishthira", "duryodhana", "bhishma", "draupadi"
    ],
    "GROUP": [
        "pandava", "kaurava", "warrior", "army", "clan", "tribe",
        "race", "kingdom"
    ],
    "PLACE": [
        "kurukshetra", "india", "hastinapura", "indraprastha",
        "forest", "city", "kingdom", "palace", "land"
    ],
    "TIME": [
        "morning", "evening", "night", "day", "year", "age"
    ],
}


@dataclass
class EntityRecord:
    """Record of an entity in the registry."""

    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: Set[str] = field(default_factory=set)
    event_ids: List[str] = field(default_factory=list)  # Events it participates in
    evidence: Dict[str, int] = field(default_factory=dict)  # chunk_id -> count


class EntityRegistry:
    """Central registry for entities.
    
    Implements admission control: entities can ONLY be created if they
    participate in detected events.
    """

    def __init__(self):
        self.alias_resolver = AliasResolver()
        self.entities: Dict[str, EntityRecord] = {}  # entity_id -> record
        self.canonical_to_id: Dict[str, str] = {}  # canonical_name -> entity_id

    def infer_type(self, text: str) -> str:
        """Infer entity type from text context.
        
        Uses heuristics:
        - If matches known person aliases -> PERSON
        - If contains group keywords -> GROUP
        - If contains place keywords -> PLACE
        - Default to PERSON for mentions
        """
        norm = normalize_name(text)

        # Check known aliases
        resolved = self.alias_resolver.resolve(text)
        if resolved and "_" in resolved:
            inferred_type = resolved.split("_")[0].upper()
            if inferred_type in VALID_TYPES:
                return inferred_type

        # Check keywords
        for entity_type, keywords in TYPE_HINTS.items():
            for keyword in keywords:
                if keyword in norm:
                    return entity_type

        # Default to PERSON (most common in Mahabharata)
        return "PERSON"

    def create_entity_from_argument(
        self, argument: EventArgument, event_id: str, chunk_id: str
    ) -> Optional[str]:
        """Create an entity from an event argument (admission control).
        
        Args:
            argument: EventArgument from event extraction
            event_id: ID of the event containing this argument
            chunk_id: Chunk where event occurred
            
        Returns:
            entity_id if created, None if rejected
        """
        if not argument.text or len(argument.text) < 2:
            logger.debug(f"Rejecting short text: {argument.text}")
            return None

        # Reject pure numbers, pronouns, common nouns
        if self._should_reject_text(argument.text):
            logger.debug(f"Rejecting text (noise): {argument.text}")
            return None

        # Infer type
        entity_type = self.infer_type(argument.text)

        # Get canonical form
        entity_id = self.alias_resolver.get_canonical_id(argument.text, entity_type)

        # Check if already exists
        if entity_id in self.entities:
            record = self.entities[entity_id]
            record.aliases.add(argument.text)
            if event_id not in record.event_ids:
                record.event_ids.append(event_id)
            record.evidence[chunk_id] = record.evidence.get(chunk_id, 0) + 1
            logger.debug(f"Updated entity {entity_id} (alias: {argument.text})")
            return entity_id

        # Create new entity
        canonical = self.alias_resolver.resolve(argument.text)
        record = EntityRecord(
            entity_id=entity_id,
            canonical_name=canonical,
            entity_type=entity_type,
            aliases={argument.text},
            event_ids=[event_id],
            evidence={chunk_id: 1},
        )
        self.entities[entity_id] = record
        self.canonical_to_id[canonical] = entity_id

        # Per-entity creation logging can be noisy; keep at DEBUG to reduce terminal spam.
        logger.debug(
            f"Created entity {entity_id} (type={entity_type}, canonical={canonical})"
        )
        return entity_id

    def get_entity(self, entity_id: str) -> Optional[EntityRecord]:
        """Get entity by ID."""
        return self.entities.get(entity_id)

    def list_entities(self) -> List[EntityRecord]:
        """List all admitted entities."""
        return list(self.entities.values())

    def entity_count(self) -> int:
        """Count admitted entities."""
        return len(self.entities)

    def get_entities_by_type(self, entity_type: str) -> List[EntityRecord]:
        """Get all entities of a specific type."""
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def get_entities_in_event(self, event_id: str) -> List[EntityRecord]:
        """Get all entities participating in a specific event."""
        return [e for e in self.entities.values() if event_id in e.event_ids]

    def _should_reject_text(self, text: str) -> bool:
        """Check if text should be rejected (noise filtering).
        
        FIX C: Strengthened rejection rules.
        
        Reject:
        - Pure numbers
        - Pronouns (comprehensive list)
        - Very short (< 2 chars)
        - Too long (>50 chars, likely description)
        - Common noise words/phrases
        - URL/filename patterns
        """
        if len(text) > 50:
            return True

        norm = normalize_name(text)

        # Pronouns (comprehensive blocklist - FIX C)
        pronouns = {
            "he", "she", "it", "they", "them", "his", "her", "its", "their",
            "thou", "thee", "thy", "him", "whom", "whose", "hers",
            "i", "me", "my", "we", "us", "our", "you", "your",
        }
        if norm in pronouns:
            return True

        # Pure numbers
        if norm.isdigit():
            return True

        # URL/filename patterns
        if any(p in norm for p in [".htm", "http", "www", ".com"]):
            return True

        # Common Mahabharata noise words
        noise_words = {
            "one", "two", "three", "many", "some", "other", "all",
            "the", "a", "an", "and", "or", "but", "with", "to",
            "in", "on", "at", "from", "for", "by", "of", "as",
        }
        if norm in noise_words:
            return True
            
        # Reject gerunds and action phrases (FIX C)
        if norm.startswith(("having ", "being ", "doing ", "making ")):
            return True
            
        # Reject if contains only common articles/prepositions
        tokens = norm.split()
        if len(tokens) > 1:
            common_words = {
                "the", "a", "an", "and", "or", "but", "in", "on",
                "at", "to", "from", "for", "of", "with", "by"
            }
            if all(token in common_words for token in tokens):
                return True

        return False

    def to_dict(self) -> Dict:
        """Serialize registry to dict."""
        return {
            "total_entities": len(self.entities),
            "entities": {
                entity_id: {
                    "canonical_name": record.canonical_name,
                    "entity_type": record.entity_type,
                    "aliases": list(record.aliases),
                    "event_count": len(record.event_ids),
                    "evidence_chunks": len(record.evidence),
                }
                for entity_id, record in self.entities.items()
            },
        }
