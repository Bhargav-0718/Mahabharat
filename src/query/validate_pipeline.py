"""Phase 5 Pipeline Validation.

Runs a set of sanity checks over the KG QA pipeline and exits non-zero on failure.
Intended for quick CI/local validation.
"""
from __future__ import annotations

import sys
import time
from typing import Dict, List

from query_planner import build_query_plan, load_entity_registry
from graph_executor import KGLoader, QueryExecutor
from evidence_collector import EvidenceCollector
from llm_answer_generator import LLMAnswerGenerator

# Minimal but representative validation queries
VALIDATION_QUERIES = [
    "Who killed Karna?",
    "Who defeated Bhishma?",
    "Who were Kunti's sons?",
    "Who was Shakuni's son?",
]

# Acceptance thresholds
MIN_EVENTS = 1      # require at least one KG event
MIN_CHUNKS = 1      # require at least one semantic chunk
MIN_CITATIONS = 1   # require at least one citation (chunk or event)
MAX_RUNTIME = 8.0   # seconds per query (soft warning)


def validate_query(question: str, components: Dict[str, object]) -> Dict[str, object]:
    start = time.time()
    try:
        planner_plan = build_query_plan(question, components["entity_registry"])
        evidence = components["collector"].collect(planner_plan, question)
        llm_answer = components["generator"].generate(question, evidence)
        elapsed = time.time() - start

        num_events = len(evidence.get("events", []))
        num_chunks = len(evidence.get("chunks", []))
        chunk_cites = len(llm_answer.get("citations", {}).get("chunks", []))
        event_cites = len(llm_answer.get("citations", {}).get("events", []))

        success = (
            num_events >= MIN_EVENTS
            and num_chunks >= MIN_CHUNKS
            and (chunk_cites + event_cites) >= MIN_CITATIONS
            and bool(llm_answer.get("answer", "").strip())
        )

        return {
            "question": question,
            "success": success,
            "elapsed": elapsed,
            "num_events": num_events,
            "num_chunks": num_chunks,
            "chunk_cites": chunk_cites,
            "event_cites": event_cites,
            "answer": llm_answer.get("answer", ""),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - start
        return {
            "question": question,
            "success": False,
            "elapsed": elapsed,
            "num_events": 0,
            "num_chunks": 0,
            "chunk_cites": 0,
            "event_cites": 0,
            "answer": "",
            "error": str(exc),
        }


def load_components(paths: Dict[str, str]) -> Dict[str, object]:
    entity_registry = load_entity_registry(paths["registry"])
    if not entity_registry:
        raise RuntimeError(f"Entity registry not found at {paths['registry']}")

    entities, events, edges = KGLoader.load_graphs(
        entities_path=paths["entities"],
        events_path=paths["events"],
        edges_path=paths["edges"],
    )

    executor = QueryExecutor(entities, events, edges)
    collector = EvidenceCollector(executor)
    generator = LLMAnswerGenerator()

    return {
        "entity_registry": entity_registry,
        "collector": collector,
        "generator": generator,
    }


def main() -> None:
    paths = {
        "registry": "data/kg/entity_registry.json",
        "entities": "data/kg/entities.json",
        "events": "data/kg/events.json",
        "edges": "data/kg/edges.json",
    }

    components = load_components(paths)

    results = []
    failures = []

    for q in VALIDATION_QUERIES:
        result = validate_query(q, components)
        results.append(result)
        ok = result["success"]
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {q}")
        if ok:
            print(
                f"  Answer: {result['answer'][:160]}...\n"
                f"  Evidence: {result['num_chunks']} chunks, {result['num_events']} events;"
                f" Citations: {result['chunk_cites']} chunks, {result['event_cites']} events;"
                f" Time: {result['elapsed']:.2f}s"
            )
            if result["elapsed"] > MAX_RUNTIME:
                print(f"  Warning: runtime {result['elapsed']:.2f}s exceeds soft limit {MAX_RUNTIME}s")
        else:
            print(f"  Error: {result['error']}")
            failures.append(result)

    print("\nSUMMARY")
    print("======")
    total = len(results)
    passed = sum(1 for r in results if r["success"])
    failed = total - passed
    print(f"Passed: {passed}/{total}")
    if failed:
        print("Failed queries:")
        for r in failures:
            print(f"  - {r['question']}: {r['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
