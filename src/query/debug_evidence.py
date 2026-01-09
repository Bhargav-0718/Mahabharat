"""Debug: inspect what evidence is collected for a query."""
from __future__ import annotations

import json
from query_planner import build_query_plan, load_entity_registry
from graph_executor import KGLoader, QueryExecutor
from evidence_collector import EvidenceCollector

def debug_evidence(question: str) -> None:
    entity_registry = load_entity_registry("data/kg/entity_registry.json")
    entities, events, edges = KGLoader.load_graphs(
        entities_path="data/kg/entities.json",
        events_path="data/kg/events.json",
        edges_path="data/kg/edges.json",
    )

    planner_plan = build_query_plan(question, entity_registry)
    executor = QueryExecutor(entities, events, edges)
    collector = EvidenceCollector(executor)
    evidence = collector.collect(planner_plan, question)

    print(f"Question: {question}\n")
    print(f"Events retrieved: {len(evidence.get('events', []))}")
    for ev in evidence.get("events", [])[:3]:
        print(f"  - {ev.get('event_id')}: {ev.get('sentence', '')[:100]}")

    print(f"\nChunks retrieved: {len(evidence.get('chunks', []))}")
    for chunk in evidence.get("chunks", [])[:3]:
        print(f"  - {chunk.get('chunk_id')}: page={chunk.get('page')}, parva={chunk.get('parva')}, score={chunk.get('relevance_score')}")
        print(f"    Text: {chunk.get('text', '')[:100]}")

if __name__ == "__main__":
    debug_evidence("Who was Shakuni's son?")
