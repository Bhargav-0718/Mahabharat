# Phase 3 Overview — Retrieval, Rerank, and Evidence-Aligned Answering

This document captures what Phase 3 does, how it is validated, and how to run it.

## Goals
- Recall the right semantic chunks using FAISS + query expansion.
- Promote event/death evidence with lightweight reranking and aggregation.
- Enforce evidence alignment so answers never outrun retrieved text.

## Inputs and Outputs
- Inputs:
  - data/semantic_chunks/chunks.jsonl
  - data/semantic_chunks/embedding_manifest.json
  - data/retrieval/faiss.index (built from manifest)
  - data/retrieval/id_mapping.json
- Primary outputs (Phase 3 runtime):
  - Console answer + citations via phase3_pipeline.py
  - Query logs (if enabled) via QueryLogger
  - phase3_validation_report.json (from phase3_tester.py)

## Pipeline Stages
Implementation: [src/retrieval/phase3_pipeline.py](src/retrieval/phase3_pipeline.py)

1) Query intent detection & expansion
- detect_death_query: regex for "who killed/slew/defeated X" and "how did X die".
- build_expansions: death/defeat variants plus war-parva expansions.
- parva_boost_map: soft boosts for Bhishma/Drona/Karna/Shalya/Sauptika Parvas.

2) Stage 1 recall (Retriever)
- [src/retrieval/retriever.py](src/retrieval/retriever.py) wraps FAISS IndexFlatIP over normalized embeddings (768-dim, jinaai/jina-embeddings-v2-base-en).
- retrieve_expanded merges primary and expanded queries, applies optional parva boosts, dedupes by best score.
- Returns scored chunks with parva/section metadata.

3) Stage 2 heuristic rerank (Reranker)
- [src/retrieval/reranker.py](src/retrieval/reranker.py) boosts entity mentions and death verbs for event-style questions; sorts and keeps a wider slate (max(top_k, 20)).

4) Cross-chunk aggregation for death/defeat
- [src/retrieval/utils/evidence_utils.py](src/retrieval/utils/evidence_utils.py)
  - aggregate_death_evidence: scans top-N (20) for entity + death-verb windows (±2 sentences), agent aliases, and co-occurrence; returns prioritized, capped citations (max 6).
  - agent_aliases_for_entity (in phase3_pipeline) maps common deaths (Karna→Arjuna, Drona→Dhrishtadyumna, Bhishma→Shikhandi/Arjuna, Jayadratha→Arjuna, Bhurishravas→Satyaki).
- If supported, emits deterministic answer text with citations; otherwise abstains.

5) Answer synthesis + evidence guard (non-death)
- AnswerSynthesizer uses GPT-4o-mini when OPENAI_API_KEY is set; otherwise extractive fallback.
- validate_answer_against_chunks: every sentence must be supported by cited text; otherwise the system abstains.
- validate_answer / validate_citations guard empty/unknown IDs.

## How to Run
- Standard query: `D:/AI/Mahabharat/.venv/Scripts/python.exe src/retrieval/phase3_pipeline.py --query "Who killed Karna?" --top-k 5`
- Force index rebuild: add `--force-rebuild-index`
- Death intent auto-triggers expansion + aggregation; non-death uses rerank + synthesis + evidence guard.

## Validation (phase3_tester.py)
- Location: [src/retrieval/phase3_tester.py](src/retrieval/phase3_tester.py)
- Runs deterministic checks (no LLM required) and writes phase3_validation_report.json:
  - File existence: chunks, embedding_manifest, faiss.index, id_mapping.
  - FAISS load consistency (index size == id map length).
  - Retrieval sanity for a death query ("Who killed Karna?") using aggregation; confirms supported evidence and returns capped citations.
  - Evidence guard unit checks (sentence support true/false cases).
- Usage: `D:/AI/Mahabharat/.venv/Scripts/python.exe src/retrieval/phase3_tester.py --output phase3_validation_report.json`

## Current State (2025-12-31)
- Embeddings: 8,632 vectors, 768-dim (jinaai/jina-embeddings-v2-base-en), L2-normalized in FAISS IndexFlatIP.
- Retrieval: two-stage (expanded recall 30+, rerank to 20, death aggregation optional).
- Evidence discipline: deterministic abstain when citations do not support claims.

## Follow-ups
- Add more agent alias maps as needed (e.g., Bhagadatta, Shalya).
- If Senapati queries need better coverage, extend ACTION_KEYWORDS/aliases in evidence_utils to reduce false negatives.
