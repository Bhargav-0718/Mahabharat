"""
Ontology Module

Defines node and edge types for the Mahabharata knowledge graph.
Establishes the graph schema and vocabulary.
"""

from enum import Enum
from typing import List, Dict, Any
from dataclasses import dataclass


class NodeType(Enum):
    """Knowledge graph node types."""
    CONTEXT_UNIT = "ContextUnit"
    CHARACTER = "Character"
    ALIAS = "Alias"
    PARVA = "Parva"
    SECTION = "Section"
    STORY_PHASE = "StoryPhase"
    CONCEPT = "Concept"


class EdgeType(Enum):
    """Knowledge graph edge types."""
    APPEARS_IN = "APPEARS_IN"  # Character → ContextUnit
    ALIAS_OF = "ALIAS_OF"  # Alias → Character
    PART_OF = "PART_OF"  # ContextUnit → Section → Parva
    DURING = "DURING"  # ContextUnit → StoryPhase
    NEXT = "NEXT"  # ContextUnit → ContextUnit (narrative order)
    MENTIONS = "MENTIONS"  # ContextUnit → Concept
    RELATED_TO = "RELATED_TO"  # ContextUnit → ContextUnit (thematic)


@dataclass
class Node:
    """Represents a graph node."""
    node_id: str
    node_type: NodeType
    attributes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.node_id,
            'type': self.node_type.value,
            'attributes': self.attributes
        }


@dataclass
class Edge:
    """Represents a graph edge."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    attributes: Dict[str, Any] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source_id,
            'target': self.target_id,
            'type': self.edge_type.value,
            'weight': self.weight,
            'attributes': self.attributes
        }


# Predefined Concepts for the knowledge graph
CORE_CONCEPTS = [
    "Dharma",
    "Karma",
    "Exile",
    "War",
    "Kingship",
    "Duty",
    "Lineage",
    "Curse",
    "Fortune",
    "Betrayal",
    "Loyalty",
    "Victory",
    "Defeat",
    "Death",
    "Rebirth",
]


class Ontology:
    """Knowledge graph ontology manager."""

    def __init__(self):
        """Initialize ontology with core concepts."""
        self.node_types = list(NodeType)
        self.edge_types = list(EdgeType)
        self.concepts = CORE_CONCEPTS

    def validate_node(self, node: Node) -> bool:
        """Validate node against ontology."""
        return node.node_type in self.node_types

    def validate_edge(self, edge: Edge) -> bool:
        """Validate edge against ontology."""
        return edge.edge_type in self.edge_types

    def get_node_schema(self) -> Dict[str, List[str]]:
        """Return schema for each node type."""
        return {
            NodeType.CONTEXT_UNIT.value: ['unit_id', 'parva', 'section', 'story_phase', 'text'],
            NodeType.CHARACTER.value: ['name', 'role', 'parva_first_appearance'],
            NodeType.ALIAS.value: ['alias_name', 'canonical_name'],
            NodeType.PARVA.value: ['parva_number', 'name'],
            NodeType.SECTION.value: ['parva', 'section_number'],
            NodeType.STORY_PHASE.value: ['phase_name', 'parvas'],
            NodeType.CONCEPT.value: ['concept_name'],
        }
