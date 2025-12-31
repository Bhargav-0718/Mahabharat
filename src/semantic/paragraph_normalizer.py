import re
from typing import List


class ParagraphNormalizer:
    """Normalize paragraph text without summarization or reordering."""

    _header_patterns = [
        re.compile(r"^table of contents.*$", re.IGNORECASE),
        re.compile(r"^index\b.*$", re.IGNORECASE),
        re.compile(r"^downloaded from:.*$", re.IGNORECASE),
        re.compile(r"^file:///.*$", re.IGNORECASE),
    ]

    @staticmethod
    def normalize(text: str) -> str:
        if not text:
            return ""

        # Replace intra-line breaks with spaces, keep paragraph boundaries external to this function.
        cleaned = text.replace("\r", "")
        cleaned = re.sub(r"\s+\n\s+", " ", cleaned)
        cleaned = re.sub(r"\n", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Remove leftover nav/header/footer noise at the start of the paragraph.
        for pattern in ParagraphNormalizer._header_patterns:
            if pattern.match(cleaned):
                cleaned = ""
                break

        return cleaned

    @staticmethod
    def normalize_paragraphs(paragraphs: List[str]) -> List[str]:
        normalized: List[str] = []
        for para in paragraphs:
            norm = ParagraphNormalizer.normalize(para)
            if norm:
                normalized.append(norm)
        return normalized
