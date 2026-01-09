"""Phase 5 Pipeline Test Script.

Runs a suite of test queries through the KG QA pipeline and reports results.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

from query_planner import build_query_plan, load_entity_registry
from graph_executor import KGLoader, QueryExecutor
from evidence_collector import EvidenceCollector
from llm_answer_generator import LLMAnswerGenerator


# Test queries covering different question types
TEST_QUERIES = [
    # Death/defeat queries
    "Who killed Karna?",
    "Who defeated Bhishma?",
    "How did Drona die?",
    "Who killed Abhimanyu?",
    
    # Relationship queries
    "Who were Kunti's sons?",
    "Who was Arjuna's father?",
    "Who were the Pandavas?",
    "Who was Shakuni's son?",
    
    # Role/identity queries
    "Who was the commander of the Kaurava army?",
    "Who was Krishna?",
    "Who taught archery to the Pandavas?",
    
    # Event queries
    "What happened at Kurukshetra?",
    "Why did the Pandavas go to exile?",
    "What was the Rajasuya sacrifice?",
    
    # Complex queries
    "Who fought in the Kurukshetra war?",
    "What weapons did Arjuna possess?",
    "Who were the main warriors on the Kaurava side?",
]


class PipelineTester:
    """Test harness for Phase 5 KG QA pipeline."""
    
    def __init__(self, paths: Dict[str, str]) -> None:
        self.paths = paths
        self.results: List[Dict] = []
        
        # Load KG once
        print("Loading KG artifacts...")
        self.entity_registry = load_entity_registry(paths["registry"])
        if not self.entity_registry:
            raise RuntimeError(f"Entity registry not found at {paths['registry']}")
        
        self.entities, self.events, self.edges = KGLoader.load_graphs(
            entities_path=paths["entities"],
            events_path=paths["events"],
            edges_path=paths["edges"],
        )
        
        self.executor = QueryExecutor(self.entities, self.events, self.edges)
        self.collector = EvidenceCollector(self.executor)
        self.generator = LLMAnswerGenerator()
        
        print(f"Loaded {len(self.entities)} entities, {len(self.events)} events, {len(self.edges)} edges\n")
    
    def run_query(self, question: str) -> Dict:
        """Execute single query and collect metrics."""
        start_time = time.time()
        
        try:
            planner_plan = build_query_plan(question, self.entity_registry)
            evidence = self.collector.collect(planner_plan, question)
            llm_answer = self.generator.generate(question, evidence)
            
            elapsed = time.time() - start_time
            
            return {
                "question": question,
                "success": True,
                "answer": llm_answer.get("answer", ""),
                "confidence": llm_answer.get("confidence", ""),
                "num_events": len(evidence.get("events", [])),
                "num_chunks": len(evidence.get("chunks", [])),
                "chunk_citations": len(llm_answer.get("citations", {}).get("chunks", [])),
                "event_citations": len(llm_answer.get("citations", {}).get("events", [])),
                "elapsed_time": elapsed,
                "error": None,
            }
        except Exception as exc:
            elapsed = time.time() - start_time
            return {
                "question": question,
                "success": False,
                "answer": "",
                "confidence": "error",
                "num_events": 0,
                "num_chunks": 0,
                "chunk_citations": 0,
                "event_citations": 0,
                "elapsed_time": elapsed,
                "error": str(exc),
            }
    
    def run_all_queries(self, queries: List[str]) -> None:
        """Run all test queries and collect results."""
        print(f"Running {len(queries)} test queries...\n")
        print("=" * 100)
        
        for i, question in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] {question}")
            print("-" * 100)
            
            result = self.run_query(question)
            self.results.append(result)
            
            if result["success"]:
                print(f"✓ Answer: {result['answer'][:150]}...")
                print(f"  Confidence: {result['confidence']}")
                print(f"  Evidence: {result['num_chunks']} chunks, {result['num_events']} events")
                print(f"  Citations: {result['chunk_citations']} chunks, {result['event_citations']} events")
                print(f"  Time: {result['elapsed_time']:.2f}s")
            else:
                print(f"✗ Error: {result['error']}")
                print(f"  Time: {result['elapsed_time']:.2f}s")
        
        print("\n" + "=" * 100)
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print test summary statistics."""
        total = len(self.results)
        success = sum(1 for r in self.results if r["success"])
        failed = total - success
        
        avg_time = sum(r["elapsed_time"] for r in self.results) / total if total > 0 else 0
        avg_chunks = sum(r["num_chunks"] for r in self.results if r["success"]) / success if success > 0 else 0
        avg_events = sum(r["num_events"] for r in self.results if r["success"]) / success if success > 0 else 0
        
        print("\n📊 TEST SUMMARY")
        print("=" * 100)
        print(f"Total queries:     {total}")
        print(f"Successful:        {success} ({100*success/total:.1f}%)")
        print(f"Failed:            {failed} ({100*failed/total:.1f}%)")
        print(f"Avg response time: {avg_time:.2f}s")
        print(f"Avg chunks/query:  {avg_chunks:.1f}")
        print(f"Avg events/query:  {avg_events:.1f}")
        
        if failed > 0:
            print("\n❌ Failed queries:")
            for r in self.results:
                if not r["success"]:
                    print(f"  • {r['question']}: {r['error']}")


def main() -> None:
    """Run Phase 5 pipeline tests."""
    paths = {
        "registry": "data/kg/entity_registry.json",
        "entities": "data/kg/entities.json",
        "events": "data/kg/events.json",
        "edges": "data/kg/edges.json",
    }
    
    tester = PipelineTester(paths)
    tester.run_all_queries(TEST_QUERIES)


if __name__ == "__main__":
    main()
