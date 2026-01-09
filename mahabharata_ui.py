"""Mahabharata QA Streamlit UI.

Runs Phase 3 semantic retrieval first; if no semantic answer, falls back to
KG + LLM synthesis. Provides chunk/event transparency and PDF download.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# Ensure src/ is on path for module imports when run via `streamlit run`
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))
QUERY_PKG = SRC / "query"
sys.path.insert(0, str(QUERY_PKG))

from retrieval.phase3_pipeline import build_expansions, detect_death_query, parva_boost_map
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from retrieval.answer_synthesizer import AnswerSynthesizer
from retrieval.utils.evidence_utils import validate_answer_against_chunks
from retrieval.validators import validate_answer, validate_citations

from query.query_planner import build_query_plan, load_entity_registry
from query.graph_executor import KGLoader, QueryExecutor
from query.evidence_collector import EvidenceCollector
from query.llm_answer_generator import LLMAnswerGenerator


logger = logging.getLogger(__name__)

RAW_PDF_PATH = Path("data/raw_pdf/MahabharataOfVyasa-EnglishTranslationByKMGanguli.pdf")


@st.cache_resource(show_spinner=False)
def load_phase3_components() -> Dict[str, Any]:
    try:
        retriever = Retriever(
            chunks_path=Path("data/semantic_chunks/chunks.jsonl"),
            manifest_path=Path("data/semantic_chunks/embedding_manifest.json"),
            index_path=Path("data/retrieval/faiss.index"),
            id_map_path=Path("data/retrieval/id_mapping.json"),
        )
        reranker = Reranker()
        synthesizer = AnswerSynthesizer()
        return {"retriever": retriever, "reranker": reranker, "synthesizer": synthesizer, "error": None}
    except Exception as exc:  # pragma: no cover - UI-facing guard
        logger.warning("Phase 3 unavailable: %s", exc)
        return {"error": str(exc)}


@st.cache_resource(show_spinner=False)
def load_kg_components() -> Dict[str, Any]:
    try:
        registry = load_entity_registry("data/kg/entity_registry.json")
        entities, events, edges = KGLoader.load_graphs(
            entities_path="data/kg/entities.json",
            events_path="data/kg/events.json",
            edges_path="data/kg/edges.json",
        )
        executor = QueryExecutor(entities, events, edges)
        collector = EvidenceCollector(executor)
        generator = LLMAnswerGenerator()
        return {
            "registry": registry,
            "collector": collector,
            "generator": generator,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - UI-facing guard
        logger.warning("KG pipeline unavailable: %s", exc)
        return {"error": str(exc)}


def run_semantic_pipeline(question: str) -> Dict[str, Any]:
    comps = load_phase3_components()
    if comps.get("error"):
        return {"status": "unavailable", "error": comps["error"]}

    retriever: Retriever = comps["retriever"]
    reranker: Reranker = comps["reranker"]
    synthesizer: AnswerSynthesizer = comps["synthesizer"]

    entity = detect_death_query(question)
    expansions = build_expansions(entity) if entity else None
    parva_boost = parva_boost_map(entity)

    stage1 = retriever.retrieve_expanded(
        query=question,
        expanded_queries=expansions,
        top_k_stage1=30,
        parva_boost=parva_boost,
    )

    if not stage1:
        return {"status": "empty", "chunks": [], "answer": "", "citations": []}

    reranked = reranker.rerank(query=question, candidates=stage1, top_k=max(10, len(stage1)), entity=entity)
    top_for_synth = reranked[:5]
    bundle = synthesizer.synthesize(question, top_for_synth)

    chunk_lookup = {c["chunk_id"]: c.get("text", "") for c in reranked if c.get("chunk_id")}
    if not validate_answer_against_chunks(bundle.get("answer", ""), bundle.get("citations", []), chunk_lookup):
        bundle["citations"] = []

    validate_answer(bundle.get("answer", ""))
    validate_citations(bundle.get("citations", []), set(chunk_lookup.keys()))

    return {
        "status": "ok",
        "answer": bundle.get("answer", ""),
        "citations": bundle.get("citations", []),
        "chunks": reranked,
    }


def run_kg_pipeline(question: str) -> Dict[str, Any]:
    comps = load_kg_components()
    if comps.get("error"):
        return {"status": "unavailable", "error": comps["error"]}

    registry = comps["registry"]
    collector: EvidenceCollector = comps["collector"]
    generator: LLMAnswerGenerator = comps["generator"]

    plan = build_query_plan(question, registry)
    evidence = collector.collect(plan, question)
    llm_answer = generator.generate(question, evidence)

    citations = llm_answer.get("citations", {}) if isinstance(llm_answer.get("citations", {}), dict) else {}

    return {
        "status": "ok",
        "answer": llm_answer.get("answer", ""),
        "events": evidence.get("events", []),
        "chunks": evidence.get("chunks", []),
        "citations": citations,
    }


def themed_header():
    st.set_page_config(page_title="Mahabharata QA", page_icon="🪔", layout="wide")
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');
            :root {--bg1:#0f172a;--bg2:#111827;--accent:#f97316;--accent2:#fcd34d;--card:#111827cc;--muted:#9ca3af;}
            .stApp {background: radial-gradient(circle at 20% 20%, rgba(249,115,22,0.08), transparent 25%),
                            radial-gradient(circle at 80% 0%, rgba(252,211,77,0.08), transparent 25%),
                            linear-gradient(135deg, #0f172a 0%, #111827 50%, #0b1224 100%);
                     color: #e5e7eb; font-family: 'Inter', sans-serif;}
            h1, h2, h3 {font-family: 'Playfair Display', serif; letter-spacing:0.5px;}
            .metric-card {background: var(--card); padding: 1rem; border-radius: 14px; border: 1px solid #1f2937;}
            .pill {background: rgba(249,115,22,0.12); border: 1px solid rgba(249,115,22,0.3); color: #fbbf24;
                   padding: 0.35rem 0.75rem; border-radius: 999px; font-size: 0.9rem; display: inline-block;}
            .answer-card {background: var(--card); padding: 1.25rem; border-radius: 14px; border: 1px solid #1f2937;
                          box-shadow: 0 20px 60px rgba(0,0,0,0.25);}
            .section-title {color: #fcd34d; font-weight: 600; margin-bottom: 0.35rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='pill'>Mahabharata SemRAG</div>", unsafe_allow_html=True)
    st.markdown("<h1>Mahabharata QA Portal</h1>", unsafe_allow_html=True)
    st.caption("Semantic retrieval first, Knowledge Graph if needed. Grounded answers with citations.")


def chunk_badge(chunk: Dict[str, Any]) -> str:
    parva = chunk.get("parva_name") or chunk.get("parva") or "Unknown Parva"
    sec = chunk.get("section_number") or chunk.get("section_index") or "?"
    score = chunk.get("relevance_score") or chunk.get("score")
    score_str = f"{score:.3f}" if isinstance(score, float) else str(score or "-")
    return f"Parva: {parva} | Section: {sec} | Score: {score_str}"


def render_chunks(chunks: List[Dict[str, Any]], cited: Optional[List[str]] = None) -> None:
    cited_set = set(cited or [])
    for chunk in chunks:
        cid = chunk.get("chunk_id")
        if not cid:
            continue
        highlighted = cid in cited_set
        with st.expander(f"📜 {cid} " + ("(cited)" if highlighted else ""), expanded=highlighted):
            st.caption(chunk_badge(chunk))
            text = chunk.get("text", "").strip()
            st.write(text if text else "(no text)")


def render_events(events: List[Dict[str, Any]], cited: Optional[List[str]] = None) -> None:
    cited_set = set(cited or [])
    for ev in events:
        eid = ev.get("event_id") or ev.get("id")
        if not eid:
            continue
        highlighted = eid in cited_set
        with st.expander(f"⚔️ {eid} " + ("(cited)" if highlighted else ""), expanded=highlighted):
            sent = ev.get("sentence", "").strip()
            st.write(sent if sent else "(no sentence)")


def read_pdf_bytes() -> Optional[bytes]:
    if not RAW_PDF_PATH.exists():
        return None
    return RAW_PDF_PATH.read_bytes()


def main() -> None:
    themed_header()

    st.sidebar.markdown("---")
    pdf_bytes = read_pdf_bytes()
    if pdf_bytes:
        st.sidebar.download_button(
            label="📥 Download Source PDF",
            data=pdf_bytes,
            file_name=RAW_PDF_PATH.name,
            mime="application/pdf",
            use_container_width=True,
        )
        st.sidebar.markdown(f"Source: `{RAW_PDF_PATH.name}`")
    else:
        st.sidebar.warning("Source PDF not found.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Built for grounded Mahabharata QA. Phase 3 → KG fallback.")

    query = st.text_area("Your question", placeholder="Who was Karna's father?", height=90)
    col_a, col_b = st.columns([1, 1])
    with col_a:
        run_btn = st.button("🔥 Ask", type="primary", use_container_width=True)
    with col_b:
        clear_btn = st.button("🧹 Clear", use_container_width=True)

    if clear_btn:
        st.experimental_rerun()

    if not run_btn:
        st.stop()

    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Running Phase 3 semantic retrieval..."):
        sem_result = run_semantic_pipeline(query)

    use_semantic = sem_result.get("status") == "ok" and bool(sem_result.get("citations"))

    if use_semantic:
        final = {
            "source": "Semantic Retrieval (Phase 3)",
            "answer": sem_result.get("answer", ""),
            "citations": sem_result.get("citations", []),
            "chunks": sem_result.get("chunks", []),
            "events": [],
        }
    else:
        with st.spinner("Semantic retrieval empty or invalid. Falling back to KG + LLM..."):
            kg_result = run_kg_pipeline(query)
        final = {
            "source": "KG + LLM",
            "answer": kg_result.get("answer", ""),
            "citations": kg_result.get("citations", {}),
            "chunks": kg_result.get("chunks", []),
            "events": kg_result.get("events", []),
            "kg_status": kg_result.get("status"),
            "kg_error": kg_result.get("error"),
        }

    st.markdown(f"<div class='answer-card'><div class='section-title'>Answer ({final['source']})</div><h3>{final['answer']}</h3></div>", unsafe_allow_html=True)

    if final.get("citations"):
        st.markdown("### Citations")
        if isinstance(final["citations"], dict):
            chunks_cited = final["citations"].get("chunks", [])
            events_cited = final["citations"].get("events", [])
            if chunks_cited:
                st.write("Chunks:", ", ".join(chunks_cited))
            if events_cited:
                st.write("Events:", ", ".join(events_cited))
        else:
            st.write(", ".join(final["citations"]))

    if final.get("chunks"):
        st.markdown("### Retrieved Chunks")
        cited_chunks = final["citations"].get("chunks", []) if isinstance(final.get("citations"), dict) else final.get("citations", [])
        render_chunks(final["chunks"], cited_chunks)

    if final.get("events"):
        st.markdown("### KG Events")
        cited_events = final["citations"].get("events", []) if isinstance(final.get("citations"), dict) else []
        render_events(final["events"][:20], cited_events)

    if not final.get("citations"):
        st.info("No citations returned. Consider refining your query.")


if __name__ == "__main__":
    main()
