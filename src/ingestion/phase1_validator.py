"""
Phase 1 Testing & Validation Utilities

Tools to validate Phase 1 output and debug the ingestion pipeline.
Includes:
- Structure validation
- Statistics generation
- Sample inspection
- Edge case detection
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)


class Phase1Validator:
    """Validate Phase 1 output and structure integrity."""

    @staticmethod
    def validate_parsed_pages(jsonl_file: str) -> Dict[str, Any]:
        """
        Validate parsed_pages.jsonl file.
        
        Checks:
        - File exists and is readable
        - Each line is valid JSON
        - Required fields present
        - Page numbers sequential
        - Text not empty
        
        Args:
            jsonl_file: Path to parsed_pages.jsonl
            
        Returns:
            Validation report dictionary
        """
        report = {
            'file': jsonl_file,
            'valid': True,
            'page_count': 0,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        file_path = Path(jsonl_file)
        if not file_path.exists():
            report['valid'] = False
            report['errors'].append(f"File not found: {jsonl_file}")
            return report
        
        try:
            pages = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        page = json.loads(line)
                        pages.append(page)
                        
                        # Check required fields
                        required_fields = ['page_number', 'text', 'word_count']
                        for field in required_fields:
                            if field not in page:
                                report['errors'].append(
                                    f"Line {line_num}: Missing field '{field}'"
                                )
                                report['valid'] = False

                        # Optional marker flags should exist for downstream checks
                        for opt_field in ['has_section_marker', 'has_parva_marker']:
                            if opt_field not in page:
                                report['warnings'].append(
                                    f"Line {line_num}: Missing optional field '{opt_field}'"
                                )
                        
                        # Check text not empty
                        if not page.get('text', '').strip():
                            report['warnings'].append(
                                f"Line {line_num}: Page {page.get('page_number')} has empty text"
                            )
                    
                    except json.JSONDecodeError as e:
                        report['errors'].append(f"Line {line_num}: Invalid JSON - {e}")
                        report['valid'] = False
            
            report['page_count'] = len(pages)
            
            # Check page numbers sequential
            if pages:
                expected_pages = list(range(1, max(p['page_number'] for p in pages) + 1))
                actual_pages = sorted([p['page_number'] for p in pages])
                missing = set(expected_pages) - set(actual_pages)
                if missing:
                    report['warnings'].append(
                        f"Missing pages: {sorted(missing)}"
                    )
            
            # Calculate statistics
            total_words = sum(p.get('word_count', 0) for p in pages)
            total_chars = sum(len(p.get('text', '')) for p in pages)
            
            report['statistics'] = {
                'total_pages': len(pages),
                'total_words': total_words,
                'total_characters': total_chars,
                'avg_words_per_page': total_words // len(pages) if pages else 0,
                'pages_with_section_marker': sum(1 for p in pages if p.get('has_section_marker')),
                'pages_with_parva_marker': sum(1 for p in pages if p.get('has_parva_marker')),
            }
        
        except Exception as e:
            report['valid'] = False
            report['errors'].append(f"Error reading file: {e}")
        
        return report

    @staticmethod
    def validate_structure(json_file: str) -> Dict[str, Any]:
        """
        Validate mahabharata_structure.json file.
        
        Checks:
        - File exists and is valid JSON
        - All 18 Parvas present
        - Parva names canonical
        - Sections have paragraphs
        - No duplicate sections within Parva
        
        Args:
            json_file: Path to mahabharata_structure.json
            
        Returns:
            Validation report dictionary
        """
        report = {
            'file': json_file,
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        file_path = Path(json_file)
        if not file_path.exists():
            report['valid'] = False
            report['errors'].append(f"File not found: {json_file}")
            return report
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                structure = json.load(f)
            
            if 'mahabharata' not in structure:
                report['valid'] = False
                report['errors'].append("Missing 'mahabharata' key in structure")
                return report
            
            parvas = structure['mahabharata'].get('parvas', [])
            canonical_names = [
                "Adi Parva", "Sabha Parva", "Vana Parva", "Virata Parva",
                "Udyoga Parva", "Bhishma Parva", "Drona Parva", "Karna Parva",
                "Shalya Parva", "Sauptika Parva", "Stri Parva", "Shanti Parva",
                "Anushasan Parva", "Ashvamedhika Parva", "Ashramavasika Parva",
                "Mausala Parva", "Mahaprasthanika Parva", "Svargarohanika Parva"
            ]
            
            # FIX: TASK 3 - Make missing Parvas a HARD ERROR (not warning)
            # Must have exactly 18 Parvas for valid Phase 1 output
            if len(parvas) != 18:
                report['valid'] = False
                report['errors'].append(
                    f"Expected exactly 18 Parvas, found {len(parvas)}"
                )
            
            # ENHANCED: Detect duplicate Parva names (critical error)
            parva_name_list = [p.get('parva_name', 'Unknown') for p in parvas]
            parva_name_counts = Counter(parva_name_list)
            duplicates = [name for name, count in parva_name_counts.items() if count > 1]
            if duplicates:
                report['valid'] = False
                for dup_name in duplicates:
                    report['errors'].append(
                        f"Duplicate Parva found: '{dup_name}' appears {parva_name_counts[dup_name]} times"
                    )

            # Canonical name check
            unknowns = [name for name in parva_name_list if name not in canonical_names]
            if unknowns:
                report['warnings'].append(
                    f"Non-canonical Parva names: {unknowns}"
                )
            
            # Collect statistics
            total_sections = 0
            total_paragraphs = 0
            parva_names = []
            empty_sections = 0
            
            for parva in parvas:
                parva_names.append(parva.get('parva_name', 'Unknown'))
                sections = parva.get('sections', [])
                total_sections += len(sections)

                # Check section_count matches actual
                advertised = parva.get('section_count', len(sections))
                if advertised != len(sections):
                    report['warnings'].append(
                        f"{parva.get('parva_name')}: section_count={advertised} but found {len(sections)} sections"
                    )
                
                # Check sections have paragraphs
                seen_section_nums = set()
                for section in sections:
                    paragraphs = section.get('paragraphs', [])
                    total_paragraphs += len(paragraphs)

                    # Duplicate section number detection (hard error)
                    sn = str(section.get('section_number', '')).strip()
                    if sn in seen_section_nums:
                        report['valid'] = False
                        report['errors'].append(
                            f"Duplicate section number '{sn}' in {parva.get('parva_name')}"
                        )
                    seen_section_nums.add(sn)

                    if not paragraphs:
                        empty_sections += 1
                        report['warnings'].append(
                            f"Section {section.get('section_number')} in "
                            f"{parva.get('parva_name')}: No paragraphs"
                        )
            
            report['statistics'] = {
                'parva_count': len(parvas),
                'total_sections': total_sections,
                'total_paragraphs': total_paragraphs,
                'parva_names': parva_names,
                'sections_per_parva': {
                    parva['parva_name']: parva.get('section_count', len(parva.get('sections', [])))
                    for parva in parvas
                },
                'empty_sections': empty_sections
            }
        
        except json.JSONDecodeError as e:
            report['valid'] = False
            report['errors'].append(f"Invalid JSON: {e}")
        except Exception as e:
            report['valid'] = False
            report['errors'].append(f"Error reading file: {e}")
        
        return report


class Phase1Inspector:
    """Inspect and display Phase 1 output details."""

    @staticmethod
    def show_page_sample(jsonl_file: str, page_num: int = 1) -> None:
        """
        Display a sample page from parsed_pages.jsonl.
        
        Args:
            jsonl_file: Path to parsed_pages.jsonl
            page_num: Page number to display (1-indexed)
        """
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i == page_num:
                    page = json.loads(line)
                    print(f"\n{'='*70}")
                    print(f"Page {page['page_number']}")
                    print(f"{'='*70}")
                    print(f"Word Count: {page['word_count']}")
                    print(f"Has Section Marker: {page['has_section_marker']}")
                    print(f"Has Parva Marker: {page['has_parva_marker']}")
                    print(f"\nText (first 1000 chars):\n")
                    print(page['text'][:1000])
                    if len(page['text']) > 1000:
                        print(f"\n... ({len(page['text']) - 1000} more characters)")
                    return
        
        print(f"Page {page_num} not found in {jsonl_file}")

    @staticmethod
    def show_structure_summary(json_file: str) -> None:
        """
        Display summary of mahabharata_structure.json.
        
        Args:
            json_file: Path to mahabharata_structure.json
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            structure = json.load(f)
        
        print(f"\n{'='*70}")
        print("MAHABHARATA STRUCTURE SUMMARY")
        print(f"{'='*70}\n")
        
        parvas = structure['mahabharata'].get('parvas', [])
        print(f"Total Parvas: {len(parvas)}\n")
        
        for i, parva in enumerate(parvas, 1):
            sections = parva.get('sections', [])
            total_paras = sum(s.get('paragraph_count', 0) for s in sections)
            
            print(f"{i:2d}. {parva['parva_name']:30s} | "
                  f"Sections: {len(sections):3d} | "
                  f"Paragraphs: {total_paras:4d}")

    @staticmethod
    def show_parva_sample(
        json_file: str,
        parva_name: str,
        section_num: str = 'I',
        para_start: int = 1,
        para_count: int = 1
    ) -> None:
        """
        Display sample paragraphs from a specific Section.
        
        Args:
            json_file: Path to mahabharata_structure.json
            parva_name: Name of Parva (e.g., "Adi Parva")
            section_num: Section number (e.g., "I", "1")
            para_start: Starting paragraph number (1-indexed)
            para_count: Number of paragraphs to show
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            structure = json.load(f)
        
        parvas = structure['mahabharata'].get('parvas', [])
        
        # Find parva
        parva = None
        for p in parvas:
            if p['parva_name'].lower() == parva_name.lower():
                parva = p
                break
        
        if not parva:
            print(f"Parva not found: {parva_name}")
            return
        
        # Find section
        section = None
        for s in parva.get('sections', []):
            if str(s.get('section_number', '')) == str(section_num):
                section = s
                break
        
        if not section:
            print(f"Section {section_num} not found in {parva_name}")
            return
        
        # Display paragraphs
        paragraphs = section.get('paragraphs', [])
        print(f"\n{'='*70}")
        print(f"{parva['parva_name']} - Section {section_num}")
        print(f"{'='*70}\n")
        
        for i in range(para_start - 1, min(para_start - 1 + para_count, len(paragraphs))):
            para = paragraphs[i]
            print(f"Paragraph {i + 1}:\n")
            print(para)
            print(f"\n{'-'*70}\n")


def generate_validation_report(
    parsed_pages_file: str,
    structure_file: str,
    output_file: str = 'phase1_validation_report.json'
) -> None:
    """
    Generate comprehensive validation report for Phase 1 output.
    
    Args:
        parsed_pages_file: Path to parsed_pages.jsonl
        structure_file: Path to mahabharata_structure.json
        output_file: Output report filename
    """
    logger.info("Generating Phase 1 validation report...")
    
    report = {
        'timestamp': datetime.utcnow().isoformat(),
        'parsed_pages': Phase1Validator.validate_parsed_pages(parsed_pages_file),
        'structure': Phase1Validator.validate_structure(structure_file),
    }
    
    # Overall status
    report['overall_valid'] = (
        report['parsed_pages']['valid'] and
        report['structure']['valid']
    )
    
    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved validation report to {output_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("PHASE 1 VALIDATION REPORT")
    print(f"{'='*70}\n")
    
    print("PARSED PAGES:")
    print(f"  Valid: {report['parsed_pages']['valid']}")
    print(f"  Pages: {report['parsed_pages']['statistics'].get('total_pages', 0)}")
    if report['parsed_pages']['errors']:
        print(f"  Errors: {len(report['parsed_pages']['errors'])}")
        for error in report['parsed_pages']['errors'][:3]:
            print(f"    - {error}")
    
    print("\nSTRUCTURE:")
    print(f"  Valid: {report['structure']['valid']}")
    print(f"  Parvas: {report['structure']['statistics'].get('parva_count', 0)}")
    print(f"  Sections: {report['structure']['statistics'].get('total_sections', 0)}")
    print(f"  Paragraphs: {report['structure']['statistics'].get('total_paragraphs', 0)}")
    
    print(f"\nOVERALL: {'PASSED' if report['overall_valid'] else 'FAILED'}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Generate validation report
    generate_validation_report(
        'data/parsed_text/parsed_pages.jsonl',
        'data/parsed_text/mahabharata_structure.json'
    )
    
    # Show samples
    try:
        Phase1Inspector.show_page_sample(
            'data/parsed_text/parsed_pages.jsonl',
            page_num=10
        )
        Phase1Inspector.show_structure_summary(
            'data/parsed_text/mahabharata_structure.json'
        )
        Phase1Inspector.show_parva_sample(
            'data/parsed_text/mahabharata_structure.json',
            parva_name='Adi Parva',
            section_num='I',
            para_count=2
        )
    except Exception as e:
        logger.warning(f"Could not display samples: {e}")
