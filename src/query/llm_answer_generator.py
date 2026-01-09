"""Phase 5C (revised): LLM Answer Generator.

Synthesizes a concise, grounded answer from retrieved chunks and KG events
using an LLM with a structured prompt and citation validation.
"""
from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not available; install with: pip install openai")


class LLMAnswerGenerator:
    """Generate a grounded answer from evidence (events + chunks)."""

    def __init__(self, llm_client: Any = None, model: str = "gpt-4o-mini") -> None:
        self.model = model
        if llm_client is not None:
            self.client = llm_client
        elif OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            logger.warning("No LLM client available; falling back to deterministic synthesis")
            self.client = None

    def generate(self, question: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Return a grounded answer using LLM synthesis over retrieved evidence."""
        events: List[Dict[str, Any]] = evidence.get("events", [])
        chunks: List[Dict[str, Any]] = evidence.get("chunks", [])

        if self.client:
            return self._llm_synthesize(question, chunks, events)
        else:
            return self._fallback_synthesize(question, chunks, events)

    def _build_prompt(self, question: str, chunks: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> str:
        """Build grounded prompt with question, chunks, and KG events."""
        prompt_parts = [
            "You are a Mahabharata scholar. Answer the user's question using ONLY the provided evidence.",
            "Be concise (1-3 sentences). Cite chunk IDs in [brackets] after relevant statements.",
            "",
            f"QUESTION: {question}",
            "",
            "RETRIEVED TEXT CHUNKS:",
        ]

        for i, chunk in enumerate(chunks[:6], 1):
            cid = chunk.get("chunk_id", f"C{i}")
            text = chunk.get("text", "").strip()
            snippet = text[:400] if len(text) > 400 else text
            prompt_parts.append(f"[{cid}] {snippet}")

        if events:
            prompt_parts.append("")
            prompt_parts.append("KNOWLEDGE GRAPH EVENTS:")
            for i, ev in enumerate(events[:5], 1):
                eid = ev.get("event_id", f"E{i}")
                sent = ev.get("sentence", "").strip()
                snippet = sent[:300] if len(sent) > 300 else sent
                prompt_parts.append(f"[{eid}] {snippet}")

        prompt_parts.extend([
            "",
            "ANSWER (cite IDs in brackets):",
        ])

        return "\n".join(prompt_parts)

    def _llm_synthesize(self, question: str, chunks: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call LLM to synthesize answer from evidence."""
        prompt = self._build_prompt(question, chunks, events)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions about the Mahabharata based strictly on provided evidence."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300,
            )

            answer_text = response.choices[0].message.content.strip()

            # Extract cited IDs from answer (look for [ID] patterns)
            import re
            cited_chunks = re.findall(r"\[([A-Z]\d{2}-S\d+-C\d+)\]", answer_text)
            cited_events = re.findall(r"\[(E\d+)\]", answer_text)

            # Validate citations against provided evidence
            valid_chunk_ids = {c.get("chunk_id") for c in chunks if c.get("chunk_id")}
            valid_event_ids = {e.get("event_id") for e in events if e.get("event_id")}

            cited_chunks = [cid for cid in cited_chunks if cid in valid_chunk_ids]
            cited_events = [eid for eid in cited_events if eid in valid_event_ids]

            # If no chunk citations extracted, use top chunks; always include event citations
            if not cited_chunks:
                cited_chunks = [c.get("chunk_id") for c in chunks[:5] if c.get("chunk_id")]
            if not cited_events:
                cited_events = [e.get("event_id") for e in events[:3] if e.get("event_id")]

            return {
                "answer": answer_text,
                "citations": {
                    "chunks": cited_chunks[:6],
                    "events": cited_events[:6],
                },
                "confidence": "high" if (cited_chunks or cited_events) else "medium",
            }

        except Exception as exc:
            logger.warning("LLM synthesis failed: %s; using fallback", exc)
            return self._fallback_synthesize(question, chunks, events)

    def _fallback_synthesize(self, question: str, chunks: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Deterministic fallback when LLM unavailable."""
        if not chunks and not events:
            return {
                "answer": "No grounded answer was found from the provided evidence.",
                "citations": {"chunks": [], "events": []},
                "confidence": "low",
            }

        # Extract first sentence from top chunk
        answer_parts = []
        if chunks:
            text = chunks[0].get("text", "").strip()
            first_sent = text.split(".")[0].strip()
            if first_sent:
                answer_parts.append(first_sent + ".")

        if events and len(answer_parts) == 0:
            sent = events[0].get("sentence", "").strip()
            first_sent = sent.split(".")[0].strip()
            if first_sent:
                answer_parts.append(first_sent + ".")

        answer_text = " ".join(answer_parts) if answer_parts else "Relevant passages retrieved but unable to synthesize answer."

        return {
            "answer": answer_text,
            "citations": {
                "chunks": [c.get("chunk_id") for c in chunks[:5] if c.get("chunk_id")],
                "events": [e.get("event_id") for e in events[:5] if e.get("event_id")],
            },
            "confidence": "medium",
        }
