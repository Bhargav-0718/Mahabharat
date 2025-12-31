"""Phase 3 tester: deterministic validations for retrieval layer.

Outputs a JSON report (phase3_validation_report.json by default) covering:
- Required file presence
- FAISS index/id-map consistency
- Death-query retrieval + aggregation sanity (no LLM required)
- Evidence guard unit checks
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import sys
from pathlib import Path as _Path

# Allow execution as a script
if __package__ is None or __package__ == "":
    _src_dir = _Path(__file__).resolve().parents[1]
    if str(_src_dir) not in sys.path:
        sys.path.append(str(_src_dir))
    from retrieval.faiss_index import FaissIndex
    from retrieval.phase3_pipeline import (
        agent_aliases_for_entity,
        build_expansions,
        parva_boost_map,
    )
    from retrieval.reranker import Reranker
    from retrieval.retriever import Retriever
    from retrieval.utils.evidence_utils import (
        aggregate_death_evidence,
        validate_answer_against_chunks,
    )
else:
    from .faiss_index import FaissIndex
    from .phase3_pipeline import (
        agent_aliases_for_entity,
        build_expansions,
        parva_boost_map,
    )
    from .reranker import Reranker
    from .retriever import Retriever
    from .utils.evidence_utils import (
        aggregate_death_evidence,
        validate_answer_against_chunks,
    )

DEFAULT_OUTPUT = Path("phase3_validation_report.json")


def check_files() -> Dict[str, object]:
    required = [
        Path("data/semantic_chunks/chunks.jsonl"),
        Path("data/semantic_chunks/embedding_manifest.json"),
        Path("data/retrieval/faiss.index"),
        Path("data/retrieval/id_mapping.json"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    status = "pass" if not missing else "fail"
    return {"name": "files_exist", "status": status, "missing": missing}


def check_faiss() -> Dict[str, object]:
    try:
        indexer = FaissIndex()
        index, ids = indexer.ensure(force=False)
        status = "pass"
        detail = {"ntotal": index.ntotal, "id_count": len(ids)}
    except Exception as exc:  # pragma: no cover - runtime guard
        status = "fail"
        detail = {"error": str(exc)}
    return {"name": "faiss_load", "status": status, "detail": detail}


def check_death_retrieval() -> Dict[str, object]:
    query = "Who killed Karna?"
    entity = "Karna"
    try:
        retriever = Retriever()
        reranker = Reranker()
        expansions = build_expansions(entity)
        boosts = parva_boost_map(entity)
        stage1 = retriever.retrieve_expanded(
            query=query,
            expanded_queries=expansions,
            top_k_stage1=30,
            parva_boost=boosts,
        )
        reranked = reranker.rerank(query=query, candidates=stage1, top_k=20, entity=entity)
        agent_meta = agent_aliases_for_entity(entity)
        agg = aggregate_death_evidence(
            chunks=reranked,
            target_entity=entity,
            agent_aliases=agent_meta[1] if agent_meta else [],
            top_n=20,
        )
        supported = bool(agg.get("supported"))
        citations = agg.get("citations", [])
        status = "pass" if supported and citations else "fail"
        detail = {
            "retrieved_stage1": len(stage1),
            "reranked": len(reranked),
            "supported": supported,
            "citations": citations,
        }
    except Exception as exc:  # pragma: no cover - runtime guard
        status = "fail"
        detail = {"error": str(exc)}
    return {"name": "death_query_retrieval", "status": status, "detail": detail}


def check_evidence_guard() -> Dict[str, object]:
    try:
        chunk_lookup: Dict[str, str] = {
            "P00-S000-C000": "Arjuna killed Karna in battle.",
            "P00-S000-C001": "Karna fell on the battlefield after being struck.",
        }
        answer_ok = "Karna was killed by Arjuna in battle."
        supported_true = validate_answer_against_chunks(answer_ok, ["P00-S000-C000"], chunk_lookup)

        answer_bad = "Bhishma was appointed the Senapati on day one."
        supported_false = validate_answer_against_chunks(answer_bad, ["P00-S000-C000"], chunk_lookup)

        status = "pass" if supported_true and not supported_false else "fail"
        detail = {
            "supported_true": supported_true,
            "supported_false": supported_false,
        }
    except Exception as exc:  # pragma: no cover - runtime guard
        status = "fail"
        detail = {"error": str(exc)}
    return {"name": "evidence_guard", "status": status, "detail": detail}


def build_report(checks: List[Dict[str, object]]) -> Dict[str, object]:
    overall = "pass" if all(ch.get("status") == "pass" for ch in checks) else "fail"
    return {
        "phase": "phase3",
        "status": overall,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 tester (retrieval layer validation)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to write validation JSON")
    args = parser.parse_args()

    checks = [
        check_files(),
        check_faiss(),
        check_death_retrieval(),
        check_evidence_guard(),
    ]

    report = build_report(checks)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote Phase 3 validation report to {args.output}")
    if report["status"] != "pass":
        print("One or more checks failed; see report for details.")


if __name__ == "__main__":
    main()
