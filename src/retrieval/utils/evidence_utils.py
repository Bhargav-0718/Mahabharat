"""Evidence utilities for Phase 3.

Provides:
- Death/defeat aggregation across chunks (non-LLM).
- Conservative evidence validation to guard against hallucinations.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

# Conservative keyword lists to avoid false positives.
DEATH_VERBS: List[str] = [
    "killed",
    "slew",
    "slain",
    "fell",
    "cut down",
    "shot",
    "struck",
    "laid low",
    "death",
]

ACTION_KEYWORDS: List[str] = [
    "killed",
    "slew",
    "slain",
    "defeated",
    "command",
    "leader",
    "senapati",
    "appointed",
    "made",
    "became",
    "fell",
    "died",
]

STOPWORDS = {
    "the",
    "and",
    "of",
    "in",
    "to",
    "a",
    "an",
    "on",
    "at",
    "by",
    "for",
    "with",
    "as",
    "from",
    "that",
    "this",
    "these",
    "those",
    "who",
    "whom",
    "whose",
    "which",
    "what",
    "when",
    "where",
    "how",
    "why",
    "is",
    "was",
    "were",
    "are",
    "be",
    "been",
    "it",
    "its",
    "their",
    "his",
    "her",
    "him",
    "she",
    "he",
    "they",
    "them",
    "you",
    "your",
    "yours",
    "we",
    "our",
    "ours",
    "but",
    "or",
}


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-zA-Z']+", text.lower()) if len(t) > 3 and t not in STOPWORDS}


def aggregate_death_evidence(
    chunks: Sequence[Dict[str, object]],
    target_entity: str,
    agent_aliases: Sequence[str],
    window: int = 2,
    top_n: int = 20,
    max_citations: int = 6,
) -> Dict[str, object]:
    """Scan top-N chunks for death + agent evidence with co-occurrence preference.

    Returns keys: supported, death_chunks, agent_chunks, cooccur_chunks, citations.
    """
    if not target_entity:
        return {"supported": False, "death_chunks": [], "agent_chunks": [], "cooccur_chunks": [], "citations": []}

    target_l = target_entity.lower()
    aliases_l = [a.lower() for a in agent_aliases]

    death_chunks: List[Dict[str, object]] = []
    agent_chunks: List[Dict[str, object]] = []
    cooccur_chunks: List[Dict[str, object]] = []

    for chunk in list(chunks)[:top_n]:
        text = str(chunk.get("text", ""))
        lowered = text.lower()
        sentences = split_sentences(lowered)

        entity_idxs = [i for i, s in enumerate(sentences) if target_l in s]
        verb_idxs = [i for i, s in enumerate(sentences) if any(v in s for v in DEATH_VERBS)]
        alias_idxs = [i for i, s in enumerate(sentences) if any(alias in s for alias in aliases_l)]

        death_hit = False
        for ei in entity_idxs:
            if any(abs(ei - vi) <= window for vi in verb_idxs):
                death_hit = True
                break
        if death_hit:
            death_chunks.append(chunk)

        alias_hit = bool(alias_idxs)
        if alias_hit:
            agent_chunks.append(chunk)

        cooccur = False
        if death_hit and alias_hit:
            for ei in entity_idxs:
                if any(abs(ei - ai) <= window for ai in alias_idxs):
                    cooccur = True
                    break
        if cooccur:
            cooccur_chunks.append(chunk)

    supported = bool(cooccur_chunks) or bool(death_chunks and agent_chunks)

    # Build a conservative citation list: prefer co-occurrence, then death, then agent.
    citations: List[str] = []
    ordered_sets = [cooccur_chunks, death_chunks, agent_chunks]
    for group in ordered_sets:
        for c in group:
            cid = c.get("chunk_id")
            if cid and cid not in citations:
                citations.append(cid)
            if len(citations) >= max_citations:
                break
        if len(citations) >= max_citations:
            break

    return {
        "supported": supported,
        "death_chunks": death_chunks,
        "agent_chunks": agent_chunks,
        "cooccur_chunks": cooccur_chunks,
        "citations": citations,
    }


def _extract_key_terms(sentence: str) -> Tuple[List[str], List[str], List[str]]:
    names = re.findall(r"\b[A-Z][a-zA-Z]+\b", sentence)
    actions = [kw for kw in ACTION_KEYWORDS if kw in sentence.lower()]
    tokens = [t for t in re.findall(r"[a-zA-Z']+", sentence.lower()) if len(t) > 3 and t not in STOPWORDS]
    return names, actions, tokens


def is_sentence_supported(sentence: str, chunk_texts: Sequence[str]) -> bool:
    if not sentence.strip():
        return True

    names, actions, tokens = _extract_key_terms(sentence)
    names_l = [n.lower() for n in names]

    for text in chunk_texts:
        if not text:
            continue
        lower_text = text.lower()

        if names_l and not all(n in lower_text for n in names_l):
            continue
        if actions and not any(a in lower_text for a in actions):
            continue

        token_overlap = len(set(tokens) & chunk_tokens(lower_text))
        needed = max(2, min(4, len(tokens))) if tokens else 0
        if token_overlap < needed:
            continue

        return True

    return False


def validate_answer_against_chunks(
    answer: str,
    cited_ids: Sequence[str],
    chunk_lookup: Dict[str, str],
) -> bool:
    """Conservative evidence check: every sentence must be supported by cited text."""
    if not answer.strip():
        return False

    chunk_texts = [chunk_lookup[cid] for cid in cited_ids if cid in chunk_lookup and chunk_lookup[cid].strip()]
    if not chunk_texts:
        return False

    sentences = split_sentences(answer.strip())
    if not sentences:
        return False

    return all(is_sentence_supported(sent, chunk_texts) for sent in sentences)
