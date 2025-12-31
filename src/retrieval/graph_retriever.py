"""
Graph Retriever Module

Performs graph traversal and filtering to retrieve relevant Context Units.
Implements entity resolution, story-phase filtering, and community detection.
"""

import logging
from typing import List, Dict, Any, Set, Tuple
import networkx as nx

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Retrieve Context Units using graph traversal and filtering."""

    def __init__(self, graph: nx.MultiDiGraph):
        """
        Initialize graph retriever.
        
        Args:
            graph: NetworkX knowledge graph
        """
        self.graph = graph

    def retrieve_by_entity(
        self,
        entity_name: str,
        story_phase: str = None
    ) -> List[str]:
        """
        Retrieve Context Units mentioning an entity.
        
        Args:
            entity_name: Character or concept name
            story_phase: Optional story phase filter
            
        Returns:
            List of Context Unit IDs
        """
        results = []
        
        # Find entity node
        entity_node = f"CHAR_{entity_name.upper().replace(' ', '_')}"
        if entity_node not in self.graph:
            logger.warning(f"Entity not found: {entity_name}")
            return results
        
        # Find APPEARS_IN edges
        for _, target, data in self.graph.out_edges(entity_node, data=True):
            if data.get('type') == 'APPEARS_IN':
                # Filter by story phase if specified
                if story_phase:
                    target_phase = self.graph.nodes[target].get('story_phase')
                    if target_phase == story_phase:
                        results.append(target)
                else:
                    results.append(target)
        
        return results

    def retrieve_by_story_phase(self, story_phase: str) -> List[str]:
        """
        Retrieve all Context Units in a story phase.
        
        Args:
            story_phase: Story phase name
            
        Returns:
            List of Context Unit IDs
        """
        results = []
        phase_node = f"PHASE_{story_phase.upper().replace(' ', '_')}"
        
        if phase_node not in self.graph:
            logger.warning(f"Story phase not found: {story_phase}")
            return results
        
        # Find DURING edges pointing to this phase
        for source, target, data in self.graph.in_edges(phase_node, data=True):
            if data.get('type') == 'DURING':
                if self.graph.nodes[source].get('type') == 'ContextUnit':
                    results.append(source)
        
        return results

    def retrieve_by_parva_section(self, parva: str, section: str = None) -> List[str]:
        """
        Retrieve Context Units from a Parva and optional Section.
        
        Args:
            parva: Parva name
            section: Optional section identifier
            
        Returns:
            List of Context Unit IDs
        """
        results = []
        
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('type') == 'ContextUnit':
                if attrs.get('parva') == parva:
                    if section is None or attrs.get('section') == section:
                        results.append(node)
        
        return results

    def retrieve_context_neighborhood(
        self,
        context_unit_id: str,
        depth: int = 1
    ) -> List[str]:
        """
        Retrieve Context Units related by graph proximity.
        
        Args:
            context_unit_id: Context Unit ID
            depth: Traversal depth (default 1)
            
        Returns:
            List of related Context Unit IDs
        """
        if context_unit_id not in self.graph:
            logger.warning(f"Context Unit not found: {context_unit_id}")
            return []
        
        results = []
        visited = set()
        queue = [(context_unit_id, 0)]
        
        while queue:
            node, current_depth = queue.pop(0)
            if node in visited or current_depth > depth:
                continue
            
            visited.add(node)
            
            # Get neighbors via NEXT, RELATED_TO, MENTIONS edges
            for neighbor, data in self.graph[node].items():
                for edge_key in self.graph[node][neighbor]:
                    edge_data = self.graph[node][neighbor][edge_key]
                    if edge_data.get('type') in ['NEXT', 'RELATED_TO']:
                        if neighbor not in visited:
                            results.append(neighbor)
                            queue.append((neighbor, current_depth + 1))
        
        return results

    def retrieve_combined(
        self,
        entities: List[str] = None,
        story_phase: str = None,
        parva: str = None
    ) -> List[str]:
        """
        Retrieve Context Units using combined filters.
        
        Args:
            entities: List of entity names to filter by
            story_phase: Story phase to filter by
            parva: Parva to filter by
            
        Returns:
            List of Context Unit IDs
        """
        results_per_filter = []
        
        if entities:
            for entity in entities:
                results_per_filter.append(set(
                    self.retrieve_by_entity(entity, story_phase)
                ))
        
        if story_phase and not entities:
            results_per_filter.append(set(self.retrieve_by_story_phase(story_phase)))
        
        if parva:
            results_per_filter.append(set(
                self.retrieve_by_parva_section(parva)
            ))
        
        # Intersection of all filters
        if results_per_filter:
            combined = results_per_filter[0]
            for result_set in results_per_filter[1:]:
                combined = combined.intersection(result_set)
            return list(combined)
        
        return []


if __name__ == '__main__':
    # Example usage
    pass
