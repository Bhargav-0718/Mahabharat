"""
Query Parser Module

Parses user queries for entity extraction, story-phase filtering, and intent.
Supports temporal and character-based queries.
"""

import logging
from typing import List, Dict, Any, Optional
import re

logger = logging.getLogger(__name__)


class QueryParser:
    """Parse and extract structured information from user queries."""

    def __init__(self):
        """Initialize query parser."""
        self.temporal_keywords = {
            'exile': ['exile', 'agyatvasa', 'forest', '13 years'],
            'war': ['war', 'kurukshetra', 'battle', 'fight'],
            'dice': ['dice', 'game', 'gambling'],
        }

    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse user query into structured components.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with extracted entities and intent
        """
        return {
            'original_query': query,
            'entities': self._extract_entities(query),
            'story_phases': self._extract_story_phases(query),
            'intent': self._detect_intent(query),
            'temporal_scope': self._extract_temporal_scope(query)
        }

    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities (character names, places)."""
        # Placeholder: integrate with spaCy NER
        entities = []
        major_characters = [
            'Arjuna', 'Yudhishthira', 'Bhima', 'Draupadi', 'Krishna',
            'Karna', 'Duryodhana', 'Ashvatthama', 'Drona', 'Bhishma'
        ]
        for char in major_characters:
            if char.lower() in query.lower():
                entities.append(char)
        return entities

    def _extract_story_phases(self, query: str) -> List[str]:
        """Extract story phase references from query."""
        phases = []
        phase_keywords = {
            'Origins': ['creation', 'ancient', 'beginning'],
            'RiseOfThePandavas': ['rise', 'youth', 'swayamvara'],
            'Exile': ['exile', 'forest', 'agyatvasa', 'wandering'],
            'PreludeToWar': ['prelude', 'negotiation', 'diplomacy'],
            'KurukshetraWar': ['war', 'kurukshetra', 'battle'],
            'ImmediateAftermath': ['aftermath', 'mourning'],
            'PostWarInstruction': ['instruction', 'teaching'],
            'WithdrawalFromTheWorld': ['withdrawal', 'heaven', 'death']
        }
        
        query_lower = query.lower()
        for phase, keywords in phase_keywords.items():
            if any(kw in query_lower for kw in keywords):
                phases.append(phase)
        
        return phases

    def _detect_intent(self, query: str) -> str:
        """Detect query intent: factual, temporal, narrative, etc."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['when', 'during', 'time']):
            return 'temporal'
        elif any(word in query_lower for word in ['where', 'location', 'place']):
            return 'location'
        elif any(word in query_lower for word in ['why', 'reason', 'cause']):
            return 'causal'
        elif any(word in query_lower for word in ['what happened', 'what did']):
            return 'narrative'
        else:
            return 'factual'

    def _extract_temporal_scope(self, query: str) -> Optional[str]:
        """Extract temporal scope if mentioned."""
        query_lower = query.lower()
        
        if 'agyatvasa' in query_lower:
            return 'Exile.Year13.Agyatvasa'
        elif 'forest' in query_lower:
            return 'Exile.ForestYears'
        elif 'war' in query_lower:
            return 'KurukshetraWar'
        
        return None


class EntityResolver:
    """Resolve entity mentions to canonical forms."""

    def __init__(self, alias_map: Dict[str, str] = None):
        """
        Initialize entity resolver.
        
        Args:
            alias_map: Dictionary mapping aliases to canonical names
        """
        self.alias_map = alias_map or self._load_default_aliases()

    def _load_default_aliases(self) -> Dict[str, str]:
        """Load default character aliases."""
        return {
            'partha': 'Arjuna',
            'dhananjaya': 'Arjuna',
            'brihannala': 'Arjuna',
            'dharmaraja': 'Yudhishthira',
            'kanka': 'Yudhishthira',
            'bhimasena': 'Bhima',
            'vrikodara': 'Bhima',
            'ballava': 'Bhima',
            'krishnaa': 'Draupadi',
            'panchali': 'Draupadi',
            'sairindhri': 'Draupadi',
            'vasudeva': 'Krishna',
            'hari': 'Krishna',
            'govinda': 'Krishna',
        }

    def resolve(self, entity_name: str) -> str:
        """
        Resolve entity to canonical form.
        
        Args:
            entity_name: Entity mention
            
        Returns:
            Canonical entity name
        """
        entity_lower = entity_name.lower()
        return self.alias_map.get(entity_lower, entity_name)


if __name__ == '__main__':
    # Example usage
    parser = QueryParser()
    result = parser.parse_query("Where did Arjuna go during the Agyatvasa?")
    print(result)
