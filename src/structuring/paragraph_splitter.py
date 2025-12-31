"""
Paragraph Splitter Module

Splits extracted text into paragraphs using heuristic rules.
Preserves paragraph boundaries for Context Unit construction.
"""

from typing import List, Dict
import logging
import re

logger = logging.getLogger(__name__)


class ParagraphSplitter:
    """Split parsed text into structured paragraphs."""

    def __init__(self):
        """Initialize paragraph splitter with heuristics."""
        # Empty line indicates paragraph boundary
        self.paragraph_boundary = re.compile(r'\n\s*\n+')

    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs using heuristic rules.
        
        Args:
            text: Raw or semi-processed text
            
        Returns:
            List of paragraph strings
        """
        # Split on double newlines
        paragraphs = self.paragraph_boundary.split(text)
        
        # Clean whitespace
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        logger.info(f"Split text into {len(paragraphs)} paragraphs")
        return paragraphs

    def split_section_into_paragraphs(self, section_text: str) -> List[Dict[str, any]]:
        """
        Split a Section into labeled paragraphs with metadata.
        
        Args:
            section_text: Text of a single Section
            
        Returns:
            List of paragraph dictionaries with position metadata
        """
        paragraphs = self.split_into_paragraphs(section_text)
        
        para_objects = [
            {
                'index': i,
                'text': para,
                'length': len(para)
            }
            for i, para in enumerate(paragraphs)
        ]
        
        return para_objects


if __name__ == '__main__':
    # Example usage
    splitter = ParagraphSplitter()
