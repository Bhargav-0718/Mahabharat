"""
Section Extractor Module - Phase 1 Ingestion

Identifies Parva and Section boundaries in parsed page text.
Builds the Parva -> Section -> Paragraph hierarchy from parsed PDF pages.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class SectionExtractor:
    """
    Extract Parva and Section structure from parsed PDF pages.
    """

    def __init__(self):
        # BOOK headers (allow optional “Part one of three” line between BOOK and PARVA)
        self.parva_title_pattern = re.compile(
            r'^\s*BOOK\s+(\d{1,2})\s*(?:\r?\n\s*[^\n]*?){0,3}\r?\n\s*([A-Z][A-Z \-]+PARVA)\b',
            re.IGNORECASE | re.MULTILINE
        )

        # BOOK headers (single-line / colon)
        self.parva_title_pattern_inline = re.compile(
            r'^\s*BOOK\s+(\d{1,2})\s*:\s*([A-Z][^\n]*?PARVA)\b',
            re.IGNORECASE | re.MULTILINE
        )

        # SECTION headers (Roman or decimal, also capture lines like
        # "The Mahabharata, Book X: <Parva>: Section N")
        self.section_header_pattern = re.compile(
            r'^[ \t]*(?:SECTION|The\s+Mahabharata[^\n]*?Section)[ \t]+([IVXLCDM]+|\d+)\b.*$',
            re.IGNORECASE | re.MULTILINE
        )

        self.parva_names = [
            "Adi Parva", "Sabha Parva", "Vana Parva", "Virata Parva",
            "Udyoga Parva", "Bhishma Parva", "Drona Parva", "Karna Parva",
            "Shalya Parva", "Sauptika Parva", "Stri Parva", "Shanti Parva",
            "Anushasan Parva", "Ashvamedhika Parva", "Ashramavasika Parva",
            "Mausala Parva", "Mahaprasthanika Parva", "Svargarohanika Parva"
        ]

    def extract_structure_from_pages(
        self,
        pages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        # Find first real BOOK header (skip TOC/preface). Scan pages >= 50 to avoid early TOC hits.
        start_page_num = None
        for p in pages:
            pn = p.get("page_number", 0)
            if pn < 50:
                continue
            txt = p.get("text", "")
            if re.search(self.parva_title_pattern, txt) or re.search(self.parva_title_pattern_inline, txt):
                start_page_num = pn
                break

        if start_page_num is None:
            start_page_num = 1
            logger.warning("No BOOK header found during page scan; using full document")
        else:
            logger.info(
                f"Frontmatter cutoff: first BOOK header on page {start_page_num}; discarding pages 1-{start_page_num-1}"
            )

        content_pages = [p for p in pages if p.get("page_number", 0) >= start_page_num]
        combined_text = "\n\n".join(page["text"] for page in content_pages)

        # Remove obvious TOC / index noise
        combined_text = re.sub(
            r"Table of Contents\s+Index[^\n]*",
            "",
            combined_text,
            flags=re.IGNORECASE
        )
        combined_text = re.sub(
            r"Downloaded from:.*?\n",
            "",
            combined_text,
            flags=re.IGNORECASE
        )

        # Trim to first book header in combined text (after page-based cutoff)
        first_book = (
            re.search(self.parva_title_pattern, combined_text)
            or re.search(self.parva_title_pattern_inline, combined_text)
        )
        if first_book:
            combined_text = combined_text[first_book.start():]

        logger.info(f"Combined {len(pages)} pages into single text")

        parvas = self._extract_parvas(combined_text)

        logger.info(f"Extracted {len(parvas)} Parvas from combined text")

        return {"mahabharata": {"parvas": parvas}}

    def _extract_parvas(self, combined_text: str) -> List[Dict[str, Any]]:
        parvas = []

        matches = list(re.finditer(self.parva_title_pattern, combined_text))
        if not matches:
            matches = list(re.finditer(self.parva_title_pattern_inline, combined_text))

        # Keep only the first occurrence per book number (handles multi-part parvas)
        seen = set()
        unique_matches = []
        for m in sorted(matches, key=lambda x: x.start()):
            bn = int(m.group(1))
            if bn in seen:
                continue
            seen.add(bn)
            unique_matches.append(m)
        matches = unique_matches

        if not matches:
            logger.error("No BOOK headers found")
            return parvas

        # Deduplicate: keep only the first match for each book number
        # (Books 12 and 13 have multiple parts with separate headers)
        seen_books = {}
        unique_matches = []
        for match in matches:
            book_number = int(match.group(1))
            if book_number not in seen_books:
                seen_books[book_number] = True
                unique_matches.append(match)

        matches = unique_matches

        for idx, match in enumerate(matches):
            book_number = int(match.group(1))
            raw_name = match.group(2).strip()
            parva_name = self._normalize_parva_name(raw_name)

            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(combined_text)
            book_text = combined_text[start:end]

            sections = self._extract_sections(book_text)
            sections = self._filter_narrative_sections(sections)

            # FULL_TEXT fallback for section-less Parvas
            if not sections:
                logger.warning(
                    f"BOOK {book_number}: {parva_name} has no SECTION markers; "
                    "treating entire BOOK as continuous narrative"
                )
                paragraphs = self._extract_paragraphs(book_text)
                sections = [{
                    "section_number": "FULL_TEXT",
                    "paragraph_count": len(paragraphs),
                    "paragraphs": paragraphs
                }]

            parvas.append({
                "parva_number": book_number,
                "parva_name": parva_name,
                "section_count": len(sections),
                "sections": sections
            })

            logger.info(
                f"Extracted BOOK {book_number}: {parva_name} "
                f"with {len(sections)} sections"
            )

        if len(parvas) != 18:
            logger.error(
                f"Parva count mismatch: expected 18, found {len(parvas)}"
            )

        return parvas

    def _extract_sections(self, parva_text: str) -> List[Dict[str, Any]]:
        sections = []
        matches = list(re.finditer(self.section_header_pattern, parva_text))

        # Keep only the first occurrence of each section number to avoid
        # splitting the same section across paginated repeats
        seen_sections = set()
        unique_matches = []
        for m in matches:
            sn = m.group(1).strip()
            if sn in seen_sections:
                continue
            seen_sections.add(sn)
            unique_matches.append(m)
        matches = unique_matches

        for idx, match in enumerate(matches):
            section_number = match.group(1).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(parva_text)

            section_text = parva_text[start:end]
            paragraphs = self._extract_paragraphs(section_text)

            if not paragraphs:
                continue

            sections.append({
                "section_number": section_number,
                "paragraph_count": len(paragraphs),
                "paragraphs": paragraphs
            })

        return sections

    def _filter_narrative_sections(
        self, sections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        filtered = []
        found_narrative = False
        threshold = 60

        for section in sections:
            paragraphs = section.get("paragraphs", [])

            has_real = any(len(p) >= threshold for p in paragraphs)
            has_mixed_case = any(
                re.search(r"[a-z].*[A-Z]|[A-Z].*[a-z]", p) for p in paragraphs
            )

            if has_real or has_mixed_case or found_narrative:
                filtered.append(section)
                if has_real:
                    found_narrative = True

        return filtered

    def _extract_paragraphs(self, text: str) -> List[str]:
        if not text:
            return []

        normalized = re.sub(r"\n(?!\s*\n)", " ", text.strip())
        paragraphs = re.split(r"\n\s*\n+", normalized)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if len(paragraphs) == 1 and len(paragraphs[0]) > 400:
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraphs[0])
            if len(sentences) > 1:
                paragraphs = [s.strip() for s in sentences if s.strip()]

        return paragraphs

    def _normalize_parva_name(self, raw_name: str) -> str:
        name = raw_name.replace("-", " ").strip()
        name = re.sub(r"\s+Parva.*$", " Parva", name, flags=re.IGNORECASE)
        name = name.title()

        replacements = {
            "Santi Parva": "Shanti Parva",
            "Anusasana Parva": "Anushasan Parva",
            "Aswamedha Parva": "Ashvamedhika Parva",
            "Asramavasika Parva": "Ashramavasika Parva",
        }
        name = replacements.get(name, name)

        if not name.endswith("Parva"):
            name += " Parva"

        for known in self.parva_names:
            if known.lower() == name.lower():
                return known

        logger.warning(f"Parva name not in canonical list: {name}")
        return name


class StructuredTextBuilder:
    @staticmethod
    def build_from_extractor(
        extractor: SectionExtractor,
        pages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        structure = extractor.extract_structure_from_pages(pages)

        structure["metadata"] = {
            "source": "KM Ganguly English Prose Translation",
            "total_pages": len(pages),
            "extraction_stage": "phase_1_structural_parsing"
        }

        return structure

    @staticmethod
    def save_structure(structure: Dict[str, Any], output_file: str) -> Path:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(structure, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved structured document to {output_path}")
        return output_path
def main():
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parsed_pages_path = Path("data/parsed_text/parsed_pages.jsonl")
    if not parsed_pages_path.exists():
        logger.error("parsed_pages.jsonl not found. Run Phase 1 PDF parsing first.")
        sys.exit(1)

    pages = []
    with open(parsed_pages_path, "r", encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))

    logger.info(f"Loaded {len(pages)} parsed pages")

    extractor = SectionExtractor()
    structure = StructuredTextBuilder.build_from_extractor(extractor, pages)

    output_path = "data/parsed_text/mahabharata_structure.json"
    StructuredTextBuilder.save_structure(structure, output_path)

    parva_count = len(structure["mahabharata"]["parvas"])
    logger.info(f"Section extraction complete — extracted {parva_count} Parvas")


if __name__ == "__main__":
    main()
