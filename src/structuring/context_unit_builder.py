"""
Context Unit Builder Module

Groups 1-3 consecutive paragraphs within a Section into Context Units.
Each Context Unit represents a single narrative fact or event.
"""

from typing import List, Dict, Optional
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ContextUnitBuilder:
    """Build Context Units from paragraphs and semantic boundaries."""

    def __init__(self, min_paragraphs: int = 1, max_paragraphs: int = 3):
        """
        Initialize Context Unit builder.
        
        Args:
            min_paragraphs: Minimum paragraphs per Context Unit (default 1)
            max_paragraphs: Maximum paragraphs per Context Unit (default 3)
        """
        self.min_paragraphs = min_paragraphs
        self.max_paragraphs = max_paragraphs

    def build_context_units(
        self,
        parva: str,
        section: str,
        paragraphs: List[Dict[str, any]],
        story_phase: str
    ) -> List[Dict[str, any]]:
        """
        Build Context Units from paragraphs.
        
        Args:
            parva: Parva name
            section: Section identifier
            paragraphs: List of paragraph dictionaries
            story_phase: Story phase identifier
            
        Returns:
            List of Context Unit dictionaries
        """
        context_units = []
        unit_id = 0
        
        for i in range(0, len(paragraphs), self.max_paragraphs):
            unit_paragraphs = paragraphs[i:i + self.max_paragraphs]
            
            unit = {
                'unit_id': f"{parva}_{section}_CU{unit_id}",
                'parva': parva,
                'section': section,
                'story_phase': story_phase,
                'paragraphs': [p['text'] for p in unit_paragraphs],
                'text': ' '.join([p['text'] for p in unit_paragraphs]),
                'paragraph_indices': [p['index'] for p in unit_paragraphs],
                'entities': [],  # To be filled by NER
                'embedding': None  # To be filled by embedding service
            }
            
            context_units.append(unit)
            unit_id += 1
        
        logger.info(f"Built {len(context_units)} Context Units for {parva} {section}")
        return context_units

    def save_context_units(
        self,
        context_units: List[Dict[str, any]],
        output_path: str
    ):
        """
        Save Context Units to JSONL file.
        
        Args:
            context_units: List of Context Unit dictionaries
            output_path: Output JSONL file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'a') as f:
            for unit in context_units:
                f.write(json.dumps(unit) + '\n')
        
        logger.info(f"Saved {len(context_units)} Context Units to {output_path}")


if __name__ == '__main__':
    # Example usage
    builder = ContextUnitBuilder()
