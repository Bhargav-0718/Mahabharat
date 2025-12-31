"""
Phase 1 Integration Module

Orchestrates the complete Phase 1 ingestion pipeline:
1. PDF Parsing (pdf_parser.py)
2. Section Extraction (section_extractor.py)

This module provides a single entry point for the entire Phase 1 workflow.
"""

import logging
import json
import sys
import hashlib
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Import Phase 1 modules
from pdf_parser import PDFParser
from section_extractor import SectionExtractor, StructuredTextBuilder

logger = logging.getLogger(__name__)


class Phase1Pipeline:
    """
    Orchestrate Phase 1 ingestion pipeline.
    
    Pipeline stages:
    1. Load PDF from data/raw_pdf/
    2. Parse pages and clean headers/footers
    3. Extract Parva -> Section -> Paragraph structure
    4. Output structured JSON to data/parsed_text/
    """

    def __init__(
        self,
        pdf_path: str,
        output_dir: str = 'data/parsed_text'
    ):
        """
        Initialize Phase 1 pipeline.
        
        Args:
            pdf_path: Path to KM Ganguly Mahabharata PDF
            output_dir: Output directory for intermediate artifacts
        """
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # FIX: TASK 4 - Checkpoint file for resumability
        self.checkpoint_file = self.output_dir / 'phase1_checkpoint.json'
        
        logger.info(f"Initialized Phase 1 Pipeline")
        logger.info(f"  PDF Path: {pdf_path}")
        logger.info(f"  Output Dir: {output_dir}")

    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of a file for change detection.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex digest of SHA256 hash, or empty string if file not found
        """
        if not file_path.exists():
            return ""
        try:
            sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to compute hash for {file_path}: {e}")
            return ""

    def _validate_checkpoint_hashes(self, checkpoint: Dict[str, Any]) -> bool:
        """
        Validate that checkpoint hashes match current files.
        If inputs (PDF) changed, invalidate all downstream stages.
        
        Args:
            checkpoint: Loaded checkpoint dictionary
            
        Returns:
            True if checkpoint is still valid, False if invalidated
        """
        if 'pdf_hash' not in checkpoint:
            return True  # No hash to validate (first run)
        
        # Check if PDF has changed
        current_pdf_hash = self._compute_file_hash(Path(self.pdf_path))
        if current_pdf_hash and current_pdf_hash != checkpoint.get('pdf_hash', ''):
            logger.warning("PDF file has changed - invalidating all stages")
            return False
        
        return True

    def _load_checkpoint(self) -> Dict[str, Any]:
        """
        Load checkpoint file if it exists and validate hash integrity.
        
        Returns:
            Checkpoint dictionary or empty dict if no checkpoint or validation fails
        """
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                    
                # Validate checkpoint hashes to detect input changes
                if not self._validate_checkpoint_hashes(checkpoint):
                    logger.warning("Checkpoint hashes invalid - starting fresh")
                    return {}
                
                logger.info("Loaded existing checkpoint for resumability")
                return checkpoint
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                return {}
        return {}

    def _save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Save checkpoint file with file hashes for integrity tracking.
        
        Args:
            checkpoint: Checkpoint dictionary with stage completion status
        """
        checkpoint['timestamp'] = str(datetime.now())
        
        # Store hashes for change detection on resume
        checkpoint['pdf_hash'] = self._compute_file_hash(Path(self.pdf_path))
        if (self.output_dir / 'parsed_pages.jsonl').exists():
            checkpoint['parsed_pages_hash'] = self._compute_file_hash(
                self.output_dir / 'parsed_pages.jsonl'
            )
        if (self.output_dir / 'mahabharata_structure.json').exists():
            checkpoint['structure_hash'] = self._compute_file_hash(
                self.output_dir / 'mahabharata_structure.json'
            )
        
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2)
            logger.info(f"Saved checkpoint to {self.checkpoint_file}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _artifact_exists(self, artifact_path: Path) -> bool:
        """
        Check if output artifact exists and is valid.
        
        Args:
            artifact_path: Path to artifact file
            
        Returns:
            True if artifact exists and is readable, False otherwise
        """
        if artifact_path.exists() and artifact_path.stat().st_size > 0:
            return True
        return False

    def _load_parsed_pages_from_jsonl(self, jsonl_file: Path) -> list:
        """
        Load previously parsed pages from JSONL file.
        
        Args:
            jsonl_file: Path to parsed_pages.jsonl
            
        Returns:
            List of page dictionaries
        """
        pages = []
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        pages.append(json.loads(line))
            logger.info(f"Loaded {len(pages)} pages from cached JSONL")
            return pages
        except Exception as e:
            logger.error(f"Failed to load cached pages: {e}")
            raise

    def _load_structure_from_json(self, json_file: Path) -> Dict[str, Any]:
        """
        Load previously extracted structure from JSON file.
        
        Args:
            json_file: Path to mahabharata_structure.json
            
        Returns:
            Structure dictionary
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                structure = json.load(f)
            logger.info(f"Loaded structure with {len(structure['mahabharata']['parvas'])} Parvas from cached JSON")
            return structure
        except Exception as e:
            logger.error(f"Failed to load cached structure: {e}")
            raise

    def run(self) -> Dict[str, Any]:
        """
        Execute complete Phase 1 pipeline with checkpoint/resume support.
        
        Returns:
            Dictionary with pipeline results and paths to output files
        """
        logger.info("=" * 70)
        logger.info("PHASE 1 INGESTION PIPELINE - START")
        logger.info("=" * 70)
        
        results = {
            'stage': 'Phase 1 - PDF Parsing & Structural Extraction',
            'status': 'in_progress',
            'outputs': {},
            'resumed': False
        }
        
        # FIX: TASK 4 - Load checkpoint for resumability
        checkpoint = self._load_checkpoint()
        
        try:
            # Stage 1: Parse PDF
            logger.info("\n[Stage 1/2] PDF PARSING")
            logger.info("-" * 70)
            
            parsed_pages_file = self.output_dir / 'parsed_pages.jsonl'
            # If parsed pages artifact exists, always reuse it (even without checkpoint)
            if self._artifact_exists(parsed_pages_file):
                logger.info("Stage 1 artifact detected. Skipping PDF parsing and loading cached pages")
                results['resumed'] = True
                pages = self._load_parsed_pages_from_jsonl(parsed_pages_file)
                # Ensure checkpoint marks stage 1 complete for future runs
                checkpoint['pdf_parsing_complete'] = True
                checkpoint['parsed_pages_file'] = str(parsed_pages_file)
                self._save_checkpoint(checkpoint)
            else:
                pages = self._stage_pdf_parsing()
                checkpoint['pdf_parsing_complete'] = True
                checkpoint['parsed_pages_file'] = str(parsed_pages_file)
                self._save_checkpoint(checkpoint)
            
            results['outputs']['parsed_pages'] = {
                'file': str(parsed_pages_file),
                'page_count': len(pages)
            }
            
            # Stage 2: Extract Structure
            logger.info("\n[Stage 2/2] SECTION EXTRACTION")
            logger.info("-" * 70)
            
            structure_file = self.output_dir / 'mahabharata_structure.json'
            # If structure already exists and checkpoint says Stage 2 is complete, skip it
            if checkpoint.get('section_extraction_complete', False) and self._artifact_exists(structure_file):
                logger.info("Stage 2 already complete. Skipping section extraction (resuming from checkpoint)")
                # Load structure from JSON
                structure = self._load_structure_from_json(structure_file)
            else:
                structure = self._stage_section_extraction(pages)
                # Update checkpoint after successful Stage 2
                checkpoint['section_extraction_complete'] = True
                checkpoint['structure_file'] = str(structure_file)
                self._save_checkpoint(checkpoint)
            
            results['outputs']['structured_document'] = {
                'file': str(structure_file),
                'parva_count': len(structure['mahabharata']['parvas'])
            }
            
            results['status'] = 'complete'
            
            logger.info("\n" + "=" * 70)
            logger.info("PHASE 1 INGESTION PIPELINE - COMPLETE")
            logger.info("=" * 70)
            logger.info("\nOutput Summary:")
            logger.info(f"  - Parsed Pages: {results['outputs']['parsed_pages']['page_count']} pages")
            logger.info(f"    -> {results['outputs']['parsed_pages']['file']}")
            logger.info(f"  - Structured Document: {results['outputs']['structured_document']['parva_count']} Parvas")
            logger.info(f"    -> {results['outputs']['structured_document']['file']}")
            
            return results
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            logger.error(f"Phase 1 pipeline failed: {e}", exc_info=True)
            raise

    def _stage_pdf_parsing(self) -> list:
        """
        Stage 1: Parse PDF and extract pages.
        
        Returns:
            List of page dictionaries
        """
        logger.info("Loading PDF...")
        
        try:
            parser = PDFParser(self.pdf_path)
            logger.info("PDF loader initialized")
            
            logger.info("Parsing pages and cleaning headers/footers...")
            pages = parser.parse()
            logger.info(f"Extracted {len(pages)} pages")
            
            logger.info("Saving parsed pages to JSONL...")
            parser.save_parsed_pages(pages, str(self.output_dir))
            logger.info(f"Saved to {self.output_dir / 'parsed_pages.jsonl'}")
            
            return pages
            
        except Exception as e:
            logger.error(f"PDF parsing failed: {e}")
            raise

    def _stage_section_extraction(self, pages: list) -> Dict[str, Any]:
        """
        Stage 2: Extract Parva/Section structure from parsed pages.
        
        Args:
            pages: List of page dictionaries from Stage 1
            
        Returns:
            Structured document with Parva -> Section -> Paragraph hierarchy
        """
        logger.info("Extracting Parva boundaries...")
        
        try:
            extractor = SectionExtractor()
            logger.info("Section extractor initialized")
            
            logger.info("Building hierarchical structure...")
            structure = StructuredTextBuilder.build_from_extractor(extractor, pages)
            logger.info(f"Extracted {len(structure['mahabharata']['parvas'])} Parvas")
            
            # Log section counts per parva
            for parva in structure['mahabharata']['parvas']:
                logger.info(
                    f"  - {parva['parva_name']}: {parva['section_count']} sections"
                )
            
            logger.info("Saving structured document...")
            StructuredTextBuilder.save_structure(
                structure,
                str(self.output_dir / 'mahabharata_structure.json')
            )
            logger.info(f"Saved to {self.output_dir / 'mahabharata_structure.json'}")
            
            return structure
            
        except Exception as e:
            logger.error(f"Section extraction failed: {e}")
            raise


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for Phase 1 pipeline.
    
    Args:
        verbose: Enable verbose (DEBUG) logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('phase1_ingestion.log')
        ]
    )


def main():
    """
    Execute Phase 1 pipeline from command line.
    
    Usage:
        python phase1_pipeline.py [--pdf <path>] [--verbose]
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Mahabharata-SemRAG Phase 1 Ingestion Pipeline'
    )
    parser.add_argument(
        '--pdf',
        type=str,
        default='data/raw_pdf/MahabharataOfVyasa-EnglishTranslationByKMGanguli.pdf',
        help='Path to KM Ganguly Mahabharata PDF'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/parsed_text',
        help='Output directory for Phase 1 artifacts'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    # Validate PDF exists
    if not Path(args.pdf).exists():
        logger.error(f"PDF not found: {args.pdf}")
        sys.exit(1)
    
    # Run pipeline
    try:
        pipeline = Phase1Pipeline(args.pdf, args.output_dir)
        results = pipeline.run()
        
        # Print results as JSON
        print("\n" + "=" * 70)
        print("PHASE 1 RESULTS")
        print("=" * 70)
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
