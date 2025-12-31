import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Support execution as a script
if __package__ is None or __package__ == "":
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.append(str(src_dir))
    from retrieval.answer_synthesizer import AnswerSynthesizer
    from retrieval.query_logger import QueryLogger
    from retrieval.retriever import Retriever
    from retrieval.reranker import Reranker
    from retrieval.validators import validate_answer, validate_citations
    from retrieval.utils.evidence_utils import (
        aggregate_death_evidence,
        validate_answer_against_chunks,
    )
else:
    from .answer_synthesizer import AnswerSynthesizer
    from .query_logger import QueryLogger
    from .retriever import Retriever
    from .reranker import Reranker
    from .validators import validate_answer, validate_citations
    from .utils.evidence_utils import (
        aggregate_death_evidence,
        validate_answer_against_chunks,
    )

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress noisy transformer/model warnings for cleaner UX
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    from transformers import logging as hf_logging  # type: ignore

    hf_logging.set_verbosity_error()
except Exception:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3: Retrieval and Answer Synthesis")
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument("--top-k", type=int, default=5, help="Top K chunks to retrieve")
    parser.add_argument("--force-rebuild-index", action="store_true", help="Rebuild FAISS index from embeddings")
    parser.add_argument("--model", default="jinaai/jina-embeddings-v2-base-en", help="Embedding model (must match Phase 2)")
    return parser.parse_args()


def detect_death_query(query: str) -> Optional[str]:
    patterns = [
        r"who\s+(killed|slew|slain|defeated)\s+(?P<name>[A-Za-z][A-Za-z\s'\-]+)\??",
        r"how\s+did\s+(?P<name>[A-Za-z][A-Za-z\s'\-]+)\s+die\??",
    ]
    q = query.lower()
    for pat in patterns:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            name = m.group("name").strip(" ?!.,")
            if name:
                return " ".join(word.capitalize() for word in name.split())
    return None


def build_expansions(entity: str) -> List[str]:
    variants = {
        f"death of {entity}",
        f"{entity} death",
        f"{entity} slain in battle",
        f"{entity} killed in battle",
        f"{entity} slain",
        f"{entity} fall in battle",
        f"{entity} defeat",
    }
    war_parvas = [
        "Bhishma Parva",
        "Drona Parva",
        "Karna Parva",
        "Shalya Parva",
        "Sauptika Parva",
    ]
    for parva in war_parvas:
        variants.add(f"{entity} {parva}")
        variants.add(f"death of {entity} {parva}")
    return sorted(variants)


def parva_boost_map(entity: Optional[str]) -> Dict[str, float]:
    # Primary war parvas receive a soft boost; Drona Parva slightly higher for Drona queries.
    if not entity:
        return {}
    base = {
        "Bhishma Parva": 1.08,
        "Drona Parva": 1.10,
        "Karna Parva": 1.08,
        "Shalya Parva": 1.05,
        "Sauptika Parva": 1.05,
    }
    if entity and entity.lower() == "drona":
        base["Drona Parva"] = 1.12
    if entity and entity.lower() == "bhishma":
        base["Bhishma Parva"] = 1.12
    if entity and entity.lower() == "karna":
        base["Karna Parva"] = 1.12
    return base


# Canonical killer aliases for high-frequency death queries.
AGENT_ALIAS_MAP: Dict[str, Tuple[str, List[str]]] = {
    "karna": (
        "Arjuna",
        ["arjuna", "partha", "dhananjaya", "pandava", "parth"],
    ),
    "drona": (
        "Dhrishtadyumna",
        ["dhrishtadyumna", "dhristadyumna", "panchala prince", "prince of the panchalas"],
    ),
    "bhishma": (
        "Shikhandi and Arjuna",
        ["shikhandi", "shikhandin", "arjuna", "partha", "dhananjaya"],
    ),
    "jayadratha": (
        "Arjuna",
        ["arjuna", "partha", "dhananjaya"],
    ),
    "bhurishravas": (
        "Satyaki",
        ["satyaki", "yuyudhana"],
    ),
}


def agent_aliases_for_entity(entity: Optional[str]) -> Optional[Tuple[str, List[str]]]:
    if not entity:
        return None
    return AGENT_ALIAS_MAP.get(entity.lower())


def load_pipeline(args: argparse.Namespace) -> dict:
    retriever = Retriever(
        chunks_path=Path("data/semantic_chunks/chunks.jsonl"),
        manifest_path=Path("data/semantic_chunks/embedding_manifest.json"),
        index_path=Path("data/retrieval/faiss.index"),
        id_map_path=Path("data/retrieval/id_mapping.json"),
        model_name=args.model,
        force_rebuild_index=args.force_rebuild_index,
    )
    reranker = Reranker()
    synthesizer = AnswerSynthesizer()
    logger_obj = QueryLogger()
    return {
        "retriever": retriever,
        "reranker": reranker,
        "synthesizer": synthesizer,
        "logger": logger_obj,
    }


def run_pipeline(args: argparse.Namespace) -> None:
    components = load_pipeline(args)
    retriever: Retriever = components["retriever"]
    reranker: Reranker = components["reranker"]
    synthesizer: AnswerSynthesizer = components["synthesizer"]
    qlogger: QueryLogger = components["logger"]

    # Detect death/defeat intent and prepare expansions
    entity = detect_death_query(args.query)
    expansions = build_expansions(entity) if entity else None
    parva_boost = parva_boost_map(entity)

    # Stage 1: widened recall with query expansions
    if entity and expansions:
        print(f"Detected death/defeat query for '{entity}'. Using expanded queries ({len(expansions)} variants).")
    retrieved_stage1 = retriever.retrieve_expanded(
        query=args.query,
        expanded_queries=expansions,
        top_k_stage1=30,
        parva_boost=parva_boost,
    )

    print(f"Stage 1 retrieved {len(retrieved_stage1)} candidates.")

    if not retrieved_stage1:
        answer_bundle = {"answer": "The retrieved passages do not explicitly state the answer.", "citations": []}
        print("ANSWER:\n" + answer_bundle.get("answer", ""))
        print("\nCITATIONS:")
        return

    # Stage 2: heuristic rerank with wider slate for aggregation
    stage2_top_k = max(args.top_k, 20)
    reranked = reranker.rerank(
        query=args.query,
        candidates=retrieved_stage1,
        top_k=stage2_top_k,
        entity=entity,
    )

    print(f"Stage 2 selected top {len(reranked)} chunks (max {stage2_top_k}).")

    # Cross-chunk aggregation for death/defeat factoids (non-LLM)
    answer_bundle: Dict[str, object]
    agent_meta = agent_aliases_for_entity(entity)
    if entity and agent_meta:
        agg = aggregate_death_evidence(
            chunks=reranked,
            target_entity=entity,
            agent_aliases=agent_meta[1],
            top_n=20,
        )
        if agg.get("supported"):
            answer_bundle = {
                "answer": f"{entity} was killed by {agent_meta[0]} during the Kurukshetra war, based on the retrieved passages.",
                "citations": agg.get("citations", []),
            }
        else:
            answer_bundle = {
                "answer": "The retrieved passages do not explicitly establish who killed the named person.",
                "citations": [],
            }
    elif entity and not agent_meta:
        answer_bundle = {
            "answer": "The retrieved passages do not explicitly establish who killed the named person.",
            "citations": [],
        }
    else:
        # Default LLM synthesis on the narrower top-k
        top_for_synth = reranked[: args.top_k]
        answer_bundle = synthesizer.synthesize(args.query, top_for_synth)

        # Evidence alignment guard
        chunk_lookup = {c["chunk_id"]: c.get("text", "") for c in reranked if c.get("chunk_id")}
        if not validate_answer_against_chunks(
            answer_bundle.get("answer", ""), answer_bundle.get("citations", []), chunk_lookup
        ):
            answer_bundle = {
                "answer": "The retrieved passages do not explicitly establish the answer.",
                "citations": [],
            }

    # Validation
    known_ids: List[str] = [item["chunk_id"] for item in reranked]
    validate_answer(answer_bundle.get("answer", ""))
    validate_citations(answer_bundle.get("citations", []), set(known_ids))

    qlogger.log(
        query=args.query,
        retrieved_ids=[r["chunk_id"] for r in retrieved_stage1],
        used_ids=answer_bundle.get("citations", []),
    )

    # Output
    print("ANSWER:\n" + answer_bundle.get("answer", ""))
    print("\nCITATIONS:")
    for cid in answer_bundle.get("citations", []):
        print(cid)

    # Print referenced chunks for transparency
    cited_set = set(answer_bundle.get("citations", []))
    if cited_set:
        print("\nREFERENCED CHUNKS:")
        for cid in answer_bundle.get("citations", []):
            chunk = next((c for c in reranked if c.get("chunk_id") == cid), None)
            if chunk:
                text = chunk.get("text", "").strip()
                snippet = text if len(text) <= 600 else text[:600].rsplit(" ", 1)[0] + "..."
                print(f"[{cid}] {snippet}\n")


if __name__ == "__main__":
    run_pipeline(parse_args())
