"""
PDF Parser Module - Phase 1 Ingestion

Extracts raw text from KM Ganguly's Mahabharata PDF.
Removes headers, footers, and page numbers while preserving structural markers
(Parva titles, Section headers, paragraph boundaries).

This is the first stage of the ingestion pipeline.
Output: raw page text with metadata (no structural parsing yet).
"""

import pdfplumber
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class PDFParser:
    """
    Parse KM Ganguly Mahabharata PDF and extract raw text.
    
    Responsibilities:
    - Load PDF from pdfplumber
    - Extract text from each page
    - Remove common headers/footers/page numbers
    - Preserve Parva titles and Section headers
    - Return page-by-page structured output
    """

    def __init__(self, pdf_path: str):
        """
        Initialize PDF parser.
        
        Args:
            pdf_path: Absolute path to the KM Ganguly Mahabharata PDF
            
        Raises:
            FileNotFoundError: If PDF does not exist
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        
        logger.info(f"Initialized PDFParser for: {self.pdf_path}")

    def _clean_page_text(self, text: str, page_number: int) -> str:
        """
        Clean extracted page text.
        
        Removes:
        - Trailing/leading whitespace per line
        - Common header/footer patterns
        - Isolated page numbers
        - Multiple consecutive blank lines (collapse to double newlines)
        
        Preserves:
        - Parva titles (all caps patterns)
        - Section headers (SECTION I, SECTION II, etc.)
        - Paragraph boundaries (single blank lines)
        
        Args:
            text: Raw extracted text from page
            page_number: Page number (for context)
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove trailing/leading whitespace
            line = line.strip()
            
            # Skip isolated page numbers (often at footer)
            if re.match(r'^\d+$', line):
                continue
            
            # Skip common header patterns (usually book/parva info)
            if re.match(r'^MAHABHARATA|^THE MAHABHARATA|^BOOK OF', line, re.IGNORECASE):
                # Keep Parva titles (they often contain "PARVA")
                if 'PARVA' in line.upper():
                    cleaned_lines.append(line)
                continue
            
            # Keep all other lines (including empty ones for paragraph structure)
            cleaned_lines.append(line)
        
        # Reconstruct text
        text = '\n'.join(cleaned_lines)
        
        # Collapse multiple consecutive blank lines to double newlines
        # This preserves paragraph boundaries (double newline) but removes excess whitespace
        text = re.sub(r'\n\n\n+', '\n\n', text)
        
        return text.strip()

    def parse(self) -> List[Dict[str, Any]]:
        """
        Parse PDF and extract text with metadata.
        
        Iterates through all pages, extracts text, cleans headers/footers,
        and returns structured page data.
        
        Returns:
            List of page dictionaries with keys:
            - page_number: int (1-indexed)
            - text: str (cleaned text)
            - word_count: int
            - has_section_marker: bool (contains "SECTION")
            - has_parva_marker: bool (contains "PARVA")
            - metadata: dict (PDF metadata)
        """
        pages = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"PDF has {total_pages} pages")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract raw text
                    raw_text = page.extract_text()
                    if not raw_text:
                        logger.warning(f"Page {page_num}: No text extracted")
                        continue
                    
                    # Clean the text
                    cleaned_text = self._clean_page_text(raw_text, page_num)
                    if not cleaned_text:
                        logger.warning(f"Page {page_num}: Text cleaned to empty")
                        continue
                    
                    # Create page record
                    page_record = {
                        'page_number': page_num,
                        'text': cleaned_text,
                        'word_count': len(cleaned_text.split()),
                        'has_section_marker': 'SECTION' in cleaned_text.upper(),
                        'has_parva_marker': 'PARVA' in cleaned_text.upper(),
                        'metadata': {
                            'producer': pdf.metadata.get('Producer', 'Unknown'),
                            'title': pdf.metadata.get('Title', 'The Mahabharata'),
                        }
                    }
                    
                    pages.append(page_record)
                    
                    if page_num % 100 == 0:
                        logger.info(f"Processed {page_num}/{total_pages} pages")
                
                logger.info(f"Successfully extracted {len(pages)} pages from {self.pdf_path.name}")
                
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            raise
        
        return pages

    def save_parsed_pages(
        self,
        pages: List[Dict[str, Any]],
        output_dir: str,
        output_filename: str = 'parsed_pages.jsonl'
    ) -> Path:
        """
        Save parsed pages to JSONL file (one JSON object per line).
        
        Args:
            pages: List of page dictionaries
            output_dir: Output directory path
            output_filename: Output filename (default: parsed_pages.jsonl)
            
        Returns:
            Path to output file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / output_filename
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for page in pages:
                    f.write(json.dumps(page, ensure_ascii=False) + '\n')
            
            logger.info(f"Saved {len(pages)} pages to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving parsed pages: {e}")
            raise


def main():
    """
    Example usage: Parse the Mahabharata PDF.
    """
    import sys
    from pathlib import Path

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    pdf_path = PROJECT_ROOT / 'data' / 'raw_pdf' / 'MahabharataOfVyasa-EnglishTranslationByKMGanguli.pdf'

    if not pdf_path.exists():
        logger.error(f"PDF not found at {pdf_path}")
        sys.exit(1)

    parser = PDFParser(str(pdf_path))
    pages = parser.parse()

    output_dir = PROJECT_ROOT / 'data' / 'parsed_text'
    parser.save_parsed_pages(pages, str(output_dir))

    logger.info("Phase 1 PDF parsing complete!")



if __name__ == '__main__':
    main()
