"""Phase 4 Post-Processing: Semantic refinements (Fixes D, E, F).

Runs after graph construction, before final saving.
- Fix D: Role-aware entity downgrading (conceptual → LITERAL)
- Fix E: Place recovery from event context
- Fix F: Minimum entity support threshold
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from .knowledge_graph import KnowledgeGraph
from .entity_registry import EntityRegistry, EntityRecord

logger = logging.getLogger(__name__)

# Conceptual nouns that should be downgraded
CONCEPTUAL_NOUNS = {
    "death", "duty", "virtue", "sin", "righteousness", "courage",
    "honor", "shame", "fate", "destiny", "time", "age", "moment",
    "night", "day", "year", "action", "deed", "consequence", "result",
}

# Known Mahabharata geographic places (whitelist for place recovery)
KNOWN_PLACES = {
    "kurukshetra", "indraprastha", "hastinapur", "dwarka", "panchala",
    "matsya", "khandavaprastha", "bharata", "kuru", "anga",
    "magadha", "videha", "kashi", "kalinga", "sindhu", "sauvira",
    "avanti", "malwa", "chedi", "salya", "trigarta",
    "uttara", "dakshin", "uttaravahini", "dakshinayana",
    "india", "bharat", "bharata", "subhara", "viratha",
}

# Character epithets and aliases that should NOT be recovered as places
CHARACTER_EPITHETS = {
    "partha", "dhananjaya", "bhimasena", "janardana", "vasudeva",
    "keshava", "govinda", "kesari", "vrikodara", "arjuna", "bhima",
    "krishna", "yudhishthira", "nakula", "sahadeva", "draupadi",
    "duryodhana", "karna", "bhishma", "drona", "ashwatthama",
    "shikhandin", "abhimanyu", "pandu", "kunti", "dhritarashtra",
    "gandhari", "vidura", "shalya", "shakuni", "subhadra",
}

# Pronouns, common words, and abstract phrases to exclude
EXCLUDED_WORDS = {
    "right", "his", "her", "their", "him", "them", "downloaded",
    "dharma", "karma", "the", "a", "an", "and", "or", "in", "at",
    "not", "slander", "tanks", "forests", "garlands", "floral",
    "island", "seven", "islands", "supreme", "felicity", "woodland",
}

# Regex pattern to detect abstract/conceptual phrases
ABSTRACT_PHRASE_PATTERN = re.compile(
    r"(slander|garland|felicity|supreme|woodland|tank|forest|flower|"
    r"virtue|sin|honor|shame|fate|destiny|action|deed|night|day|year|moment|age)",
    re.IGNORECASE
)

# Place patterns for extraction
PLACE_PATTERNS = [
    r"\bat\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
    r"\bin\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
    r"\bfield of\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
    r"\bnear\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
]

# Abstract event types (only)
ABSTRACT_EVENTS = {"DEATH", "VOW", "BOON", "CURSE"}


class Phase4Postprocessor:
    """Post-process KG to refine entities and recover places."""

    def __init__(self, graph: KnowledgeGraph, entity_registry: EntityRegistry):
        self.graph = graph
        self.registry = entity_registry
        self.stats = {
            "downgraded": 0,
            "places_recovered": 0,
            "entities_removed": 0,
            "edges_removed": 0,
        }

    def run(self) -> None:
        """Run all post-processing fixes."""
        logger.info("Running Phase 4 post-processing fixes...")
        
        # Fix D: Downgrade conceptual entities
        self._fix_d_downgrade_conceptual()
        
        # Fix E: Recover places from events
        self._fix_e_recover_places()
        
        # Fix F: Apply minimum support threshold
        self._fix_f_minimum_support()
        
        logger.info(f"Post-processing complete: {self.stats}")

    def _fix_d_downgrade_conceptual(self) -> None:
        """FIX D: Downgrade conceptual/abstract entities to LITERAL type."""
        logger.info("FIX D: Downgrading conceptual entities...")
        
        downgraded = []
        
        for entity_id, record in list(self.registry.entities.items()):
            if record.entity_type != "PERSON":
                continue
            
            # Check if should downgrade
            if self._should_downgrade(entity_id, record):
                # Change type to LITERAL
                record.entity_type = "LITERAL"
                downgraded.append(record.canonical_name)
                self.stats["downgraded"] += 1
        
        if downgraded:
            logger.info(f"Downgraded {len(downgraded)} entities to LITERAL: {downgraded[:10]}")

    def _should_downgrade(self, entity_id: str, record: EntityRecord) -> bool:
        """Check if entity should be downgraded based on Fix D criteria."""
        canonical = record.canonical_name.lower()
        
        # Rule 1: Must be lowercase (no proper nouns)
        if record.canonical_name != canonical:
            return False
        
        # Rule 2: Must be in conceptual nouns list
        if canonical not in CONCEPTUAL_NOUNS:
            return False
        
        # Rule 3: Analyze role distribution and event types
        object_count = 0
        abstract_only = True
        
        for event_id in record.event_ids:
            event = self.graph.events.get(event_id)
            if not event:
                continue
            
            # Check if event is abstract-only type
            if event.event_type not in ABSTRACT_EVENTS:
                abstract_only = False
            
            # Count if entity is mostly object (via edges)
            for edge in self.graph.edges:
                if edge.event_id == event_id and edge.target_id == entity_id:
                    object_count += 1
        
        # Rule 3: Most edges are incoming (object role) AND only abstract events
        total_event_edges = len([e for e in self.graph.edges if entity_id in (e.source_id, e.target_id)])
        if total_event_edges > 0:
            object_ratio = object_count / total_event_edges
        else:
            object_ratio = 0
        
        # Rule 4: No spatial, kinship, command edges
        has_structural_role = any(
            (edge.source_id == entity_id or edge.target_id == entity_id)
            and edge.edge_type in ("OCCURRED_AT", "RELATED_TO", "KINSHIP", "COMMAND")
            for edge in self.graph.edges
        )
        
        return abstract_only and object_ratio >= 0.80 and not has_structural_role

    def _fix_e_recover_places(self) -> None:
        """FIX E: Extract and recover PLACE entities from event sentences."""
        logger.info("FIX E: Recovering places from event context...")
        
        recovered = set()
        
        for event_id, event in self.graph.events.items():
            sentence = event.sentence
            
            # Try each place pattern
            for pattern in PLACE_PATTERNS:
                matches = re.finditer(pattern, sentence)
                for match in matches:
                    place_text = match.group(1).strip()
                    
                    # Skip if too short or generic
                    if len(place_text) < 3 or place_text.lower() in ("the", "a", "an"):
                        continue
                    
                    # Create or reuse PLACE entity
                    place_id = self._admit_place_entity(place_text, event_id)
                    
                    if place_id:
                        recovered.add(place_text)
                        
                        # Create OCCURRED_AT edge
                        self._create_occurred_at_edge(event_id, place_id)
        
        if recovered:
            self.stats["places_recovered"] = len(recovered)
            logger.info(f"Recovered {len(recovered)} places: {list(recovered)[:15]}")

    def _admit_place_entity(self, place_text: str, event_id: str) -> str:
        """Admit or reuse a PLACE entity."""
        # Normalize
        canonical = place_text.lower()
        
        # Rule 1: Check against exclusion lists
        if canonical in EXCLUDED_WORDS or canonical in CHARACTER_EPITHETS:
            return None
        
        # Rule 2: Check if matches abstract phrase pattern
        if ABSTRACT_PHRASE_PATTERN.search(canonical):
            return None
        
        # Rule 3: Check if this text already exists as PERSON/GROUP
        for ent_id, record in self.registry.entities.items():
            if record.canonical_name.lower() == canonical:
                # Found existing entity with same name
                if record.entity_type in ("PERSON", "GROUP"):
                    # Skip this place - it's actually a character/group
                    return None
                if record.entity_type == "PLACE":
                    # Reuse the existing place
                    if event_id not in record.event_ids:
                        record.event_ids.append(event_id)
                    return ent_id
        
        # Rule 4: Only admit places that are in KNOWN_PLACES whitelist
        # OR are multi-word geographic compounds (e.g., "Field of X" where X is known)
        has_known_place = any(p in canonical for p in KNOWN_PLACES)
        
        if not has_known_place:
            return None
        
        # Create new place entity
        place_id = f"place_{canonical.replace(' ', '_')}"
        record = EntityRecord(
            entity_id=place_id,
            canonical_name=canonical,
            entity_type="PLACE",
            aliases={place_text},
            event_ids=[event_id],
            evidence={},
        )
        self.registry.entities[place_id] = record
        return place_id

    def _create_occurred_at_edge(self, event_id: str, place_id: str) -> None:
        """Create OCCURRED_AT edge from event to place."""
        # Avoid duplicates
        if any(
            e.source_id == event_id and e.target_id == place_id and e.edge_type == "OCCURRED_AT"
            for e in self.graph.edges
        ):
            return
        
        from .knowledge_graph import GraphEdge
        edge = GraphEdge(
            source_id=event_id,
            target_id=place_id,
            edge_type="OCCURRED_AT",
            event_id=event_id,
            event_type=self.graph.events[event_id].event_type,
            weight=1,
            evidence=[self.graph.events[event_id].chunk_id],
        )
        self.graph.edges.append(edge)

    def _fix_f_minimum_support(self) -> None:
        """FIX F: Apply minimum entity support thresholds."""
        logger.info("FIX F: Applying minimum support threshold...")
        
        thresholds = {
            "PERSON": 2,
            "GROUP": 1,
            "PLACE": 1,
            "TIME": 1,
            "LITERAL": 0,  # Keep all LITERAL entities
        }
        
        to_remove = set()
        
        for entity_id, record in list(self.registry.entities.items()):
            threshold = thresholds.get(record.entity_type, 1)
            if len(record.event_ids) < threshold:
                to_remove.add(entity_id)
        
        # Remove entities and their edges
        for entity_id in to_remove:
            # Remove entity
            del self.registry.entities[entity_id]
            self.stats["entities_removed"] += 1
            
            # Remove associated edges
            edges_before = len(self.graph.edges)
            self.graph.edges = [
                e for e in self.graph.edges
                if e.source_id != entity_id and e.target_id != entity_id
            ]
            edges_removed = edges_before - len(self.graph.edges)
            self.stats["edges_removed"] += edges_removed
        
        if to_remove:
            logger.info(f"Removed {len(to_remove)} entities below support threshold")


def postprocess_graph(graph: KnowledgeGraph, entity_registry: EntityRegistry) -> None:
    """Apply all post-processing fixes to the KG."""
    processor = Phase4Postprocessor(graph, entity_registry)
    processor.run()
