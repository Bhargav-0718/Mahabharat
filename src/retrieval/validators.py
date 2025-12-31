from typing import Dict, List, Set


def validate_answer(answer: str) -> None:
    if not answer or not answer.strip():
        raise ValueError("Answer text is empty")


def validate_citations(citations: List[str], known_ids: Set[str]) -> None:
    if len(citations) != len(set(citations)):
        raise ValueError("Duplicate citations detected")
    for cid in citations:
        if cid not in known_ids:
            raise ValueError(f"Citation not found in index: {cid}")
