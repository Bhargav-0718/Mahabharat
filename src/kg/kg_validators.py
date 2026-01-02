"""Validation checks for knowledge graph integrity (EVENT-CENTRIC).

Ensures:
- No orphan entities (all must participate in at least one event)
- No abstract types (PERSON, GROUP, PLACE, TIME only)
- Events have evidence (chunk_id, parva, section)
- Aliases collapse correctly
- No suspicious patterns
"""
from __future__ import annotations

import logging
from typing import Dict, List, Set

from .knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class GraphValidator:
    """Validates knowledge graph integrity."""

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> bool:
        """Run all validation checks.
        
        Returns:
            True if graph is valid (no critical errors)
        """
        self.errors = []
        self.warnings = []

        self._check_entity_types()
        self._check_orphan_entities()
        self._check_event_evidence()
        self._check_alias_collisions()
        self._check_suspicious_patterns()

        if self.errors:
            logger.error(f"Validation failed with {len(self.errors)} errors")
            for err in self.errors:
                logger.error(f"  - {err}")

        if self.warnings:
            logger.warning(f"Validation found {len(self.warnings)} warnings")
            for warn in self.warnings:
                logger.warning(f"  - {warn}")

        return len(self.errors) == 0

    def _check_entity_types(self) -> None:
        """Verify all entities have valid types."""
        valid_types = {"PERSON", "GROUP", "PLACE", "TIME"}
        for record in self.graph.entity_registry.list_entities():
            if record.entity_type not in valid_types:
                self.errors.append(
                    f"Invalid entity type: {record.entity_id} has type {record.entity_type}"
                )

    def _check_orphan_entities(self) -> None:
        """Verify all entities participate in at least one event."""
        for record in self.graph.entity_registry.list_entities():
            if not record.event_ids:
                self.errors.append(
                    f"Orphan entity: {record.entity_id} ({record.canonical_name}) "
                    "participates in no events"
                )

    def _check_event_evidence(self) -> None:
        """Verify all events have proper evidence metadata."""
        for event in self.graph.events.values():
            if not event.chunk_id:
                self.errors.append(f"Event {event.event_id} has no chunk_id")
            if not event.parva:
                self.errors.append(f"Event {event.event_id} has no parva")
            if not event.section:
                self.errors.append(f"Event {event.event_id} has no section")

    def _check_alias_collisions(self) -> None:
        """Check for problematic alias patterns."""
        alias_to_entities: Dict[str, Set[str]] = {}

        for record in self.graph.entity_registry.list_entities():
            for alias in record.aliases:
                if alias not in alias_to_entities:
                    alias_to_entities[alias] = set()
                alias_to_entities[alias].add(record.entity_id)

        for alias, entity_ids in alias_to_entities.items():
            if len(entity_ids) > 1:
                self.warnings.append(
                    f"Alias collision: '{alias}' appears in {len(entity_ids)} entities"
                )

    def _check_suspicious_patterns(self) -> None:
        """Check for suspicious patterns that might indicate errors."""
        for record in self.graph.entity_registry.list_entities():
            name = record.canonical_name
            if len(name) > 50:
                self.warnings.append(
                    f"Suspicious entity name: {record.entity_id} ({name[:20]}...) is very long"
                )

    def get_report(self) -> Dict:
        """Get validation report."""
        return {
            "valid": len(self.errors) == 0,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": {
                "entity_count": self.graph.entity_count(),
                "event_count": self.graph.event_count(),
                "edge_count": self.graph.edge_count(),
            },
        }


# Legacy functions for compatibility
def validate_no_self_loops(graph) -> List[str]:
    return []


def validate_entities_exist(graph) -> List[str]:
    return []


def validate_symmetry(graph) -> List[str]:
    return []


def validate_required_fields(graph) -> List[str]:
    return []


def run_validations(graph) -> Dict[str, List[str]]:
    """Legacy stub."""
    return {
        "no_self_loops": [],
        "entities_exist": [],
        "symmetry": [],
        "required_fields": [],
    }
