"""
Graph Builder Module

Constructs the knowledge graph from Context Units and metadata.
Implements node and edge creation based on the ontology.
"""

import networkx as nx
from typing import List, Dict, Any, Tuple
import logging
import json
from pathlib import Path

from ontology import Node, Edge, NodeType, EdgeType, Ontology

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build and manage the Mahabharata knowledge graph."""

    def __init__(self):
        """Initialize graph builder."""
        self.graph = nx.MultiDiGraph()
        self.ontology = Ontology()
        self.node_count = 0
        self.edge_count = 0

    def add_node(self, node: Node) -> bool:
        """
        Add node to graph.
        
        Args:
            node: Node object
            
        Returns:
            True if added, False if validation fails
        """
        if not self.ontology.validate_node(node):
            logger.warning(f"Invalid node type: {node.node_type}")
            return False
        
        self.graph.add_node(
            node.node_id,
            type=node.node_type.value,
            **node.attributes
        )
        self.node_count += 1
        return True

    def add_edge(self, edge: Edge) -> bool:
        """
        Add edge to graph.
        
        Args:
            edge: Edge object
            
        Returns:
            True if added, False if validation fails
        """
        if not self.ontology.validate_edge(edge):
            logger.warning(f"Invalid edge type: {edge.edge_type}")
            return False
        
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            type=edge.edge_type.value,
            weight=edge.weight,
            **edge.attributes
        )
        self.edge_count += 1
        return True

    def add_context_unit_node(self, context_unit: Dict[str, Any]) -> str:
        """
        Create and add a Context Unit node.
        
        Args:
            context_unit: Context Unit dictionary
            
        Returns:
            Node ID
        """
        node_id = context_unit['unit_id']
        node = Node(
            node_id=node_id,
            node_type=NodeType.CONTEXT_UNIT,
            attributes={
                'parva': context_unit['parva'],
                'section': context_unit['section'],
                'story_phase': context_unit['story_phase'],
                'text': context_unit['text'],
                'paragraphs': context_unit['paragraphs']
            }
        )
        self.add_node(node)
        return node_id

    def add_character_node(self, character_name: str, first_appearance: str = None) -> str:
        """
        Create and add a Character node.
        
        Args:
            character_name: Character name
            first_appearance: Parva of first appearance
            
        Returns:
            Node ID
        """
        node_id = f"CHAR_{character_name.upper().replace(' ', '_')}"
        node = Node(
            node_id=node_id,
            node_type=NodeType.CHARACTER,
            attributes={
                'name': character_name,
                'first_appearance': first_appearance
            }
        )
        self.add_node(node)
        return node_id

    def add_parva_node(self, parva_number: int, parva_name: str) -> str:
        """
        Create and add a Parva node.
        
        Args:
            parva_number: Parva number (1-18)
            parva_name: Parva name
            
        Returns:
            Node ID
        """
        node_id = f"PARVA_{parva_number:02d}"
        node = Node(
            node_id=node_id,
            node_type=NodeType.PARVA,
            attributes={
                'parva_number': parva_number,
                'name': parva_name
            }
        )
        self.add_node(node)
        return node_id

    def add_story_phase_node(self, phase_name: str, parvas: List[int] = None) -> str:
        """
        Create and add a Story Phase node.
        
        Args:
            phase_name: Story phase name
            parvas: List of Parva numbers
            
        Returns:
            Node ID
        """
        node_id = f"PHASE_{phase_name.upper().replace(' ', '_')}"
        node = Node(
            node_id=node_id,
            node_type=NodeType.STORY_PHASE,
            attributes={
                'name': phase_name,
                'parvas': parvas or []
            }
        )
        self.add_node(node)
        return node_id

    def link_context_unit_to_section(self, cu_id: str, parva: str, section: str):
        """Link Context Unit to its Section."""
        section_id = f"SEC_{parva}_{section}"
        edge = Edge(cu_id, section_id, EdgeType.PART_OF)
        self.add_edge(edge)

    def link_context_unit_to_phase(self, cu_id: str, phase_name: str):
        """Link Context Unit to its Story Phase."""
        phase_id = f"PHASE_{phase_name.upper().replace(' ', '_')}"
        edge = Edge(cu_id, phase_id, EdgeType.DURING)
        self.add_edge(edge)

    def link_consecutive_units(self, cu_id1: str, cu_id2: str):
        """Link consecutive Context Units in narrative order."""
        edge = Edge(cu_id1, cu_id2, EdgeType.NEXT)
        self.add_edge(edge)

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph)
        }

    def save_graph(self, output_path: str):
        """
        Save graph to file.
        
        Args:
            output_path: Path to save graph (GraphML format)
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(self.graph, output_file)
        logger.info(f"Saved graph to {output_path}")


if __name__ == '__main__':
    # Example usage
    builder = GraphBuilder()
