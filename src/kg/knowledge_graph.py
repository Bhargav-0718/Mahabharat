"""Knowledge graph construction and storage (EVENT-CENTRIC).

Graph schema:
- Nodes: entities (PERSON, GROUP, PLACE, TIME) and events
- Edges: PARTICIPATED_IN, CAUSED, OCCURRED_AT, OCCURRED_IN, ASSOCIATED_WITH

All edges trace back to events for evidence.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .event_extractor import ExtractedEvent, EventArgument
from .entity_registry import EntityRegistry, EntityRecord

logger = logging.getLogger(__name__)


@dataclass
class GraphEdge:
    """Directed edge in knowledge graph."""

    source_id: str
    target_id: str
    edge_type: str  # PARTICIPATED_IN, CAUSED, OCCURRED_AT, OCCURRED_IN, ASSOCIATED_WITH
    event_id: str  # Event that triggered this edge
    event_type: str
    weight: int = 1  # Strength (number of supporting events)
    evidence: List[str] = None  # chunk_ids

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []

    def to_dict(self) -> Dict:
        """Serialize edge."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "weight": self.weight,
            "evidence": self.evidence,
        }


@dataclass
class GraphEvent:
    """Event node with metadata."""

    event_id: str
    event_type: str
    sentence: str
    chunk_id: str
    parva: str
    section: str
    participant_ids: List[str]  # entity_ids
    tier: str = "MESO"  # "MACRO" or "MESO"


class KnowledgeGraph:
    """Knowledge graph builder and manager (EVENT-CENTRIC)."""

    def __init__(self, entity_registry: EntityRegistry):
        self.entity_registry = entity_registry
        self.events: Dict[str, GraphEvent] = {}  # event_id -> GraphEvent
        self.edges: List[GraphEdge] = []
        self.edge_index: Dict[Tuple[str, str], List[GraphEdge]] = {}  # (source, target) -> edges
        self.event_counter = 0  # FIX 1: Monotonic event ID generator

    def add_event(self, extracted_event: ExtractedEvent, event_id: str) -> None:
        """Add event to graph.
        
        Args:
            extracted_event: Event with extracted arguments
            event_id: Original event ID (for tracking)
        """
        # FIX 1: Use globally unique monotonic event ID
        unique_event_id = f"E{self.event_counter}"
        self.event_counter += 1
        
        # FIX 2: Create entities from arguments (admission control)
        participant_ids = []
        for arg in extracted_event.arguments:
            entity_id = self.entity_registry.create_entity_from_argument(
                arg, unique_event_id, extracted_event.chunk_id
            )
            if entity_id:
                participant_ids.append(entity_id)

        # FIX 2: ALWAYS admit event node (decouple from edge creation)
        self.events[unique_event_id] = GraphEvent(
            event_id=unique_event_id,
            event_type=extracted_event.event_type,
            sentence=extracted_event.sentence,
            chunk_id=extracted_event.chunk_id,
            parva=extracted_event.parva,
            section=extracted_event.section,
            participant_ids=participant_ids,
            tier=extracted_event.tier,
        )

        # Build edges only if arguments exist
        if participant_ids:
            self._build_edges_from_event(extracted_event, unique_event_id, participant_ids)

    def _build_edges_from_event(
        self, event: ExtractedEvent, event_id: str, participant_ids: List[str]
    ) -> None:
        """Build graph edges from event arguments.
        
        Roles:
        - agent, subject -> PARTICIPATED_IN
        - patient, object -> PARTICIPATED_IN
        - recipient -> PARTICIPATED_IN
        """
        for arg in event.arguments:
            entity_id = None
            for pid in participant_ids:
                record = self.entity_registry.get_entity(pid)
                if record and arg.text in record.aliases:
                    entity_id = pid
                    break

            if not entity_id:
                continue

            # All participants participate in the event
            self._add_edge(
                source_id=entity_id,
                target_id=event_id,
                edge_type="PARTICIPATED_IN",
                event_id=event_id,
                event_type=event.event_type,
                chunk_id=event.chunk_id,
            )

    def _add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        event_id: str,
        event_type: str,
        chunk_id: str,
    ) -> None:
        """Add or update edge in graph.
        
        Merges duplicate edges (same source, target, type) and increments weight.
        """
        key = (source_id, target_id)

        # Check if edge exists
        if key in self.edge_index:
            for edge in self.edge_index[key]:
                if edge.edge_type == edge_type:
                    # Merge: increment weight and add evidence
                    edge.weight += 1
                    if chunk_id not in edge.evidence:
                        edge.evidence.append(chunk_id)
                    return

        # Create new edge
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            event_id=event_id,
            event_type=event_type,
            evidence=[chunk_id],
        )
        self.edges.append(edge)
        if key not in self.edge_index:
            self.edge_index[key] = []
        self.edge_index[key].append(edge)

    def get_connected_entities(self, entity_id: str) -> Dict[str, List[str]]:
        """Get all entities connected to a given entity.
        
        Returns:
            {edge_type: [target_ids]}
        """
        connected = {}
        for edge in self.edges:
            if edge.source_id == entity_id:
                if edge.edge_type not in connected:
                    connected[edge.edge_type] = []
                if edge.target_id not in connected[edge.edge_type]:
                    connected[edge.edge_type].append(edge.target_id)

        return connected

    def get_events_for_entity(self, entity_id: str) -> List[GraphEvent]:
        """Get all events where entity participated."""
        event_ids = set()
        for edge in self.edges:
            if edge.source_id == entity_id and edge.edge_type == "PARTICIPATED_IN":
                event_ids.add(edge.target_id)

        return [self.events[eid] for eid in event_ids if eid in self.events]

    def entity_count(self) -> int:
        """Count entities in graph."""
        return self.entity_registry.entity_count()

    def event_count(self) -> int:
        """Count events in graph."""
        return len(self.events)

    def edge_count(self) -> int:
        """Count edges in graph."""
        return len(self.edges)

    def to_dict(self) -> Dict:
        """Serialize graph to dict."""
        entities = []
        for record in self.entity_registry.list_entities():
            entities.append({
                "entity_id": record.entity_id,
                "canonical_name": record.canonical_name,
                "entity_type": record.entity_type,
                "aliases": list(record.aliases),
                "event_count": len(record.event_ids),
                "evidence_chunks": list(record.evidence.keys()),
            })

        events = [
            {
                "event_id": e.event_id,
                "event_type": e.event_type,
                "sentence": e.sentence,
                "chunk_id": e.chunk_id,
                "parva": e.parva,
                "section": e.section,
                "participant_count": len(e.participant_ids),
            }
            for e in self.events.values()
        ]

        edges = [e.to_dict() for e in self.edges]

        return {
            "summary": {
                "entity_count": self.entity_count(),
                "event_count": self.event_count(),
                "edge_count": self.edge_count(),
            },
            "entities": entities,
            "events": events,
            "edges": edges,
        }

    def save(self, output_dir) -> None:
        """Save graph to JSON files.
        
        Outputs:
        - entities.json
        - events.json
        - edges.json
        - graph_stats.json
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save entities
        entities_data = {
            "metadata": {"total": self.entity_count()},
            "entities": {
                record.entity_id: {
                    "canonical_name": record.canonical_name,
                    "type": record.entity_type,
                    "aliases": list(record.aliases),
                    "event_count": len(record.event_ids),
                    "evidence": record.evidence,
                }
                for record in self.entity_registry.list_entities()
            },
        }
        with open(output_dir / "entities.json", "w") as f:
            json.dump(entities_data, f, indent=2)

        # Save events
        events_data = {
            "metadata": {"total": self.event_count()},
            "events": {
                e.event_id: {
                    "type": e.event_type,
                    "tier": e.tier,  # Include tier information
                    "sentence": e.sentence,
                    "location": {"chunk": e.chunk_id, "parva": e.parva, "section": e.section},
                    "participants": e.participant_ids,
                }
                for e in self.events.values()
            },
        }
        with open(output_dir / "events.json", "w") as f:
            json.dump(events_data, f, indent=2)

        # Save edges
        edges_data = {
            "metadata": {"total": self.edge_count()},
            "edges": [e.to_dict() for e in self.edges],
        }
        with open(output_dir / "edges.json", "w") as f:
            json.dump(edges_data, f, indent=2)

        # Save stats
        stats = self.to_dict()["summary"]
        with open(output_dir / "graph_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved knowledge graph to {output_dir}")

    def to_entities(self) -> List[Dict[str, object]]:
        """Legacy method for compatibility."""
        result: List[Dict[str, object]] = []
        for record in self.entity_registry.list_entities():
            result.append({
                "id": record.entity_id,
                "type": record.entity_type,
                "aliases": list(record.aliases)
            })
        return result

    def to_relations(self) -> List[Dict[str, object]]:
        """Legacy method for compatibility."""
        rels: List[Dict[str, object]] = []
        for edge in self.edges:
            rels.append({
                "subject": edge.source_id,
                "relation": edge.edge_type,
                "object": edge.target_id,
                "evidence": edge.evidence,
                "confidence": 1.0,
            })
        return rels

    def to_json(self) -> Dict[str, object]:
        """Legacy method for compatibility."""
        return {
            "entities": self.to_entities(),
            "relations": self.to_relations(),
        }
