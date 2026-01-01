"""Phase 4 KG pipeline: entity/alias extraction, relation building, KG validation."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Enable execution as a script
if __package__ is None or __package__ == "":
    _src_dir = Path(__file__).resolve().parents[1]
    if str(_src_dir) not in sys.path:
        sys.path.append(str(_src_dir))
    from kg.alias_resolver import AliasResolver
    from kg.entity_extractor import EntityExtractor
    from kg.knowledge_graph import KnowledgeGraph
    from kg.kg_validators import run_validations
    from kg.relation_extractor import RelationExtractor
    from kg.schemas import KGStats, RelationRecord
else:
    from .alias_resolver import AliasResolver
    from .entity_extractor import EntityExtractor
    from .knowledge_graph import KnowledgeGraph
    from .kg_validators import run_validations
    from .relation_extractor import RelationExtractor
    from .schemas import KGStats, RelationRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ALIASES_PATH = Path("src/kg/data/alias_map.json")
PATTERNS_PATH = Path("src/kg/data/relation_patterns.json")
CHUNKS_PATH = Path("data/semantic_chunks/chunks.jsonl")
OUTPUT_DIR = Path("data/kg")
CHECKPOINT_PATH = OUTPUT_DIR / "kg_checkpoint.json"


def load_alias_map() -> Dict[str, Dict[str, object]]:
    if not ALIASES_PATH.exists():
        raise FileNotFoundError(f"alias_map.json not found at {ALIASES_PATH}")
    return json.loads(ALIASES_PATH.read_text(encoding="utf-8"))


def load_chunks(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def should_skip(force: bool) -> bool:
    if force:
        return False
    if not CHECKPOINT_PATH.exists():
        return False
    try:
        chk = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        completed = chk.get("completed")
        recorded_mtime = chk.get("chunks_mtime")
        current_mtime = CHUNKS_PATH.stat().st_mtime if CHUNKS_PATH.exists() else None
        if completed and recorded_mtime == current_mtime:
            logger.info("Checkpoint is current; skipping rebuild. Use --force to override.")
            return True
    except Exception:
        return False
    return False


def save_checkpoint() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed": True,
        "chunks_mtime": CHUNKS_PATH.stat().st_mtime if CHUNKS_PATH.exists() else None,
        "timestamp": os.path.getmtime(CHUNKS_PATH) if CHUNKS_PATH.exists() else None,
    }
    CHECKPOINT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_entities(alias_map: Dict[str, Dict[str, object]], kg: KnowledgeGraph) -> None:
    for entity_id, meta in alias_map.items():
        etype = str(meta.get("type", "")).upper()
        aliases = meta.get("aliases", []) or []
        kg.add_entity(entity_id, etype, aliases)


def process_chunks(
    chunks: List[Dict[str, object]],
    extractor: EntityExtractor,
    resolver: AliasResolver,
    rel_extractor: RelationExtractor,
) -> Tuple[List[RelationRecord], int]:
    relations: List[RelationRecord] = []
    resolved_count = 0
    for row in chunks:
        chunk_id = row.get("chunk_id")
        text = row.get("text", "")
        mentions = extractor.extract(chunk_id, text)
        resolved = resolver.resolve_mentions(mentions)
        if resolved:
            resolved_count += len(resolved)
        rels = rel_extractor.extract_relations(text, chunk_id, resolved)
        relations.extend(rels)
    return relations, resolved_count


def save_outputs(kg: KnowledgeGraph, stats: KGStats) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entities_path = OUTPUT_DIR / "entities.json"
    relations_path = OUTPUT_DIR / "relations.json"
    kg_json_path = OUTPUT_DIR / "knowledge_graph.json"
    stats_path = OUTPUT_DIR / "kg_stats.json"

    entities_path.write_text(json.dumps(kg.to_entities(), indent=2), encoding="utf-8")
    relations_path.write_text(json.dumps(kg.to_relations(), indent=2), encoding="utf-8")
    kg_json_path.write_text(json.dumps(kg.to_json(), indent=2), encoding="utf-8")
    stats_path.write_text(
        json.dumps(
            {
                "entities": stats.entities,
                "relations": stats.relations,
                "edges": stats.edges,
                "nodes": stats.nodes,
                "warnings": stats.warnings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_checkpoint()


def run_pipeline(force: bool = False, validate_only: bool = False) -> None:
    if should_skip(force) and not validate_only:
        logger.info("Phase 4 outputs already present and current.")
        return

    alias_map = load_alias_map()
    extractor = EntityExtractor(alias_map)
    resolver = AliasResolver(alias_map)
    rel_extractor = RelationExtractor(PATTERNS_PATH)

    kg = KnowledgeGraph()
    build_entities(alias_map, kg)

    chunks = load_chunks(CHUNKS_PATH)
    relations: List[RelationRecord] = []
    resolved_mentions = 0

    if not validate_only:
        relations, resolved_mentions = process_chunks(chunks, extractor, resolver, rel_extractor)
        for rel in relations:
            kg.add_relation(rel)

    validations = run_validations(kg.graph)
    warnings_flat = [w for lst in validations.values() for w in lst]

    stats = KGStats(
        entities=len(kg.to_entities()),
        relations=len(relations),
        edges=kg.graph.number_of_edges(),
        nodes=kg.graph.number_of_nodes(),
        warnings=warnings_flat,
    )

    if not validate_only:
        save_outputs(kg, stats)
        logger.info("Saved Phase 4 KG outputs to %s", OUTPUT_DIR)
        logger.info("Relations extracted: %d | Resolved mentions: %d", len(relations), resolved_mentions)
    else:
        logger.info("Validate-only run: entities=%d nodes=%d edges=%d warnings=%d", len(kg.to_entities()), stats.nodes, stats.edges, len(warnings_flat))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 KG pipeline")
    parser.add_argument("--force", action="store_true", help="Rebuild KG even if checkpoint is fresh")
    parser.add_argument("--validate-only", action="store_true", help="Run validations only")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(force=args.force, validate_only=args.validate_only)
