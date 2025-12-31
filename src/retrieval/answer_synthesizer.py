import logging
import os
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class AnswerSynthesizer:
    """Answer synthesizer with strict citation discipline.

    Uses GPT-4o-mini when OPENAI_API_KEY is available; otherwise falls back to a
    deterministic extractive summary with inline citations.
    """

    def __init__(
        self,
        max_chunks: int = 5,
        max_sentences: int = 3,
        model_name: str = "gpt-4o-mini",
    ) -> None:
        self.max_chunks = max_chunks
        self.max_sentences = max_sentences
        self.model_name = model_name

    def _first_sentences(self, text: str, max_sentences: int) -> str:
        if not text:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        trimmed = [s for s in sentences if s]
        return " ".join(trimmed[:max_sentences]).strip()

    def _compose_extractive(self, chunks: List[Dict[str, str]]) -> Tuple[str, List[str]]:
        pieces: List[str] = []
        citations: List[str] = []
        for chunk in chunks[: self.max_chunks]:
            cid = chunk.get("chunk_id")
            text = chunk.get("text", "")
            if not cid or not text:
                continue
            snippet = self._first_sentences(text, max_sentences=1)
            if not snippet:
                continue
            pieces.append(f"{snippet} ({cid})")
            citations.append(cid)

        if not pieces:
            return "No supporting evidence found.", []
        return " " + " ".join(pieces), citations

    def _build_prompt(self, question: str, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
        context_lines = []
        for chunk in chunks[: self.max_chunks]:
            cid = chunk.get("chunk_id")
            text = chunk.get("text", "")
            if cid and text:
                context_lines.append(f"[{cid}] {text}")

        context_block = "\n".join(context_lines)
        user_content = (
            "You are answering a question strictly using the provided Mahabharata chunks.\n"
            "Rules:\n"
            "- Cite chunk IDs in every statement you make.\n"
            "- If you cannot answer, say so.\n"
            "- Do not invent citations or facts.\n\n"
            f"Question: {question}\n\n"
            f"Chunks:\n{context_block}\n\n"
            "Respond with a concise answer (2-4 sentences) and include citations in-line, e.g., (P01-S001-C001)."
        )
        return [
            {"role": "system", "content": "You are a careful assistant that only uses provided chunks."},
            {"role": "user", "content": user_content},
        ]

    def _extract_citations(self, text: str) -> List[str]:
        return re.findall(r"P\d{2}-S\d{3}-C\d{3}", text)

    def _call_llm(self, question: str, chunks: List[Dict[str, str]]) -> Tuple[str, List[str]]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency check
            raise RuntimeError("openai package is required to call GPT-4o-mini") from exc

        client = OpenAI(api_key=api_key)
        messages = self._build_prompt(question, chunks)
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=400,
        )
        answer = resp.choices[0].message.content.strip()
        citations = self._extract_citations(answer)
        return answer, citations

    def synthesize(self, question: str, retrieved: List[Dict[str, str]]) -> Dict[str, object]:
        if not retrieved:
            return {"answer": "No relevant chunks found.", "citations": []}

        # Try LLM first; fall back to extractive if unavailable
        try:
            answer_text, citations = self._call_llm(question, retrieved)
        except Exception as exc:
            logger.warning("LLM unavailable, using extractive fallback: %s", exc)
            answer_text, citations = self._compose_extractive(retrieved)

        if not citations:
            # ensure we always return some citations from retrieved top chunks
            citations = [c["chunk_id"] for c in retrieved[: self.max_chunks] if c.get("chunk_id")]

        return {
            "answer": answer_text,
            "citations": citations,
        }
