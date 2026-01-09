"""Phase 5 Orchestrator: end-to-end KG QA pipeline.

Loads Phase 4 KG artifacts once, accepts a user question, and runs:
Planner → EvidenceCollector → LLMAnswerGenerator. Outputs a user-facing answer
with citations (events + chunks).
"""
from __future__ import annotations

import argparse
from typing import Any, Dict

from query_planner import build_query_plan, load_entity_registry
from graph_executor import KGLoader, QueryExecutor
from evidence_collector import EvidenceCollector
from llm_answer_generator import LLMAnswerGenerator


def run_pipeline(question: str, paths: Dict[str, str]) -> None:
    # Load registry and KG once
    entity_registry = load_entity_registry(paths["registry"])
    if not entity_registry:
        raise RuntimeError(f"Entity registry not found at {paths['registry']}")

    entities, events, edges = KGLoader.load_graphs(
        entities_path=paths["entities"],
        events_path=paths["events"],
        edges_path=paths["edges"],
    )

    planner_plan = build_query_plan(question, entity_registry)
    executor = QueryExecutor(entities, events, edges)
    collector = EvidenceCollector(executor)
    evidence = collector.collect(planner_plan, question)

    generator = LLMAnswerGenerator()
    llm_answer = generator.generate(question, evidence)

    print(f"Question: {question}")
    print("ANSWER:\n" + llm_answer.get("answer", ""))

    citations = llm_answer.get("citations", {})
    chunk_ids = citations.get("chunks", []) if isinstance(citations, dict) else []
    event_ids = citations.get("events", []) if isinstance(citations, dict) else []

    print("\nCITATIONS:")
    if chunk_ids:
        print("Chunks:", ", ".join(chunk_ids))
    if event_ids:
        print("KG Events:", ", ".join(event_ids))
    if not chunk_ids and not event_ids:
        print("None")

    if chunk_ids:
        print("\nREFERENCED CHUNKS:")
        chunk_lookup = {c.get("chunk_id"): c for c in evidence.get("chunks", []) if c.get("chunk_id")}
        for cid in chunk_ids:
            chunk = chunk_lookup.get(cid)
            if not chunk:
                continue
            # Extract metadata
            page = chunk.get("page", "N/A")
            parva = chunk.get("parva", "N/A")
            score = chunk.get("relevance_score", chunk.get("score", "N/A"))
            if isinstance(score, float):
                score = f"{score:.3f}"
            metadata = f"  Metadata: page={page}, parva={parva}, relevance_score={score}"
            text = chunk.get("text", "").strip()
            snippet = text if len(text) <= 600 else text[:597].rsplit(" ", 1)[0] + "..."
            print(f"[{cid}]")
            print(f"{metadata}")
            print(f"  Text: {snippet}\n")

    if event_ids:
        print("KG EVENTS:")
        event_lookup = {e.get("event_id"): e for e in evidence.get("events", []) if e.get("event_id")}
        for eid in event_ids:
            ev = event_lookup.get(eid)
            if not ev:
                continue
            sentence = ev.get("sentence", "").strip()
            snippet = sentence if len(sentence) <= 600 else sentence[:597].rsplit(" ", 1)[0] + "..."
            print(f"[{eid}] {snippet}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 KG QA pipeline")
    parser.add_argument("question", help="Natural language question to answer")
    parser.add_argument("--registry_path", default="data/kg/entity_registry.json")
    parser.add_argument("--entities_path", default="data/kg/entities.json")
    parser.add_argument("--events_path", default="data/kg/events.json")
    parser.add_argument("--edges_path", default="data/kg/edges.json")
    args = parser.parse_args()

    paths = {
        "registry": args.registry_path,
        "entities": args.entities_path,
        "events": args.events_path,
        "edges": args.edges_path,
    }

    run_pipeline(args.question, paths)


if __name__ == "__main__":
    main()
