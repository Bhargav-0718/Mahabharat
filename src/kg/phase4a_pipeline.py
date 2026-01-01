"""Phase 4A pipeline: entity extraction and alias resolution only."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

# Enable execution as script
if __package__ is None or __package__ == "":
    _src_dir = Path(__file__).resolve().parents[1]
    if str(_src_dir) not in sys.path:
        sys.path.append(str(_src_dir))
    from kg.entity_recognizer import EntityRecognizer
    from kg.entity_resolver import AliasResolver, _normalize
    from kg.schemas import EntityMention
else:
    from .entity_recognizer import EntityRecognizer
    from .entity_resolver import AliasResolver, _normalize
    from .schemas import EntityMention

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHUNKS_PATH = Path("data/semantic_chunks/chunks.jsonl")
OUTPUT_DIR = Path("data/kg")
CHECKPOINT_PATH = OUTPUT_DIR / "phase4a_checkpoint.json"
MENTIONS_PATH = OUTPUT_DIR / "entity_mentions.json"
SEED_ALIASES_PATH = Path("src/kg/data/seed_aliases.json")
MENTIONS_CHECKPOINT = OUTPUT_DIR / "phase4a_mentions_checkpoint.json"


def load_chunks(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Chunks not found: {path}")
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_seed_aliases() -> Dict[str, Dict[str, object]]:
    """Load curated seed aliases to anchor canonical entity IDs."""
    if not SEED_ALIASES_PATH.exists():
        logger.warning("Seed aliases not found at %s; proceeding without seeding", SEED_ALIASES_PATH)
        return {}
    return json.loads(SEED_ALIASES_PATH.read_text(encoding="utf-8"))


def should_skip(force: bool) -> bool:
    if force:
        return False
    if not CHECKPOINT_PATH.exists():
        return False
    try:
        chk = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if chk.get("completed"):
            chunks_mtime = CHUNKS_PATH.stat().st_mtime if CHUNKS_PATH.exists() else None
            if chk.get("chunks_mtime") == chunks_mtime:
                logger.info("Phase 4A complete and current; skipping. Use --force to override.")
                return True
    except Exception:
        pass
    return False


def load_cached_mentions() -> List[EntityMention]:
    if not MENTIONS_PATH.exists():
        return []
    data = json.loads(MENTIONS_PATH.read_text(encoding="utf-8"))
    mentions: List[EntityMention] = []
    for row in data:
        mentions.append(
            EntityMention(
                text=row.get("mention_text", ""),
                type=row.get("type", ""),
                chunk_id=row.get("chunk_id", ""),
                start=row.get("start"),
                end=row.get("end"),
            )
        )
    return mentions


def mentions_checkpoint_current() -> bool:
    if not MENTIONS_CHECKPOINT.exists():
        return False
    try:
        chk = json.loads(MENTIONS_CHECKPOINT.read_text(encoding="utf-8"))
        if chk.get("mentions_saved"):
            chunks_mtime = CHUNKS_PATH.stat().st_mtime if CHUNKS_PATH.exists() else None
            return chk.get("chunks_mtime") == chunks_mtime
    except Exception:
        return False
    return False


def save_checkpoint() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed": True,
        "chunks_mtime": CHUNKS_PATH.stat().st_mtime if CHUNKS_PATH.exists() else None,
    }
    CHECKPOINT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_mentions_checkpoint() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "mentions_saved": True,
        "chunks_mtime": CHUNKS_PATH.stat().st_mtime if CHUNKS_PATH.exists() else None,
    }
    MENTIONS_CHECKPOINT.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def extract_entities(chunks: List[Dict[str, object]], reuse_mentions: bool = False) -> tuple:
    """Extract and resolve entities. Optionally reuse cached mentions to skip NER."""
    resolver = AliasResolver(similarity_threshold=0.88, min_mention_frequency=2)
    all_mentions: List[EntityMention] = []

    if reuse_mentions:
        if not mentions_checkpoint_current() or not MENTIONS_PATH.exists():
            reuse_mentions = False
        else:
            cached = load_cached_mentions()
            if cached:
                logger.info("Reusing %d cached mentions from %s", len(cached), MENTIONS_PATH)
                all_mentions = cached
            else:
                logger.info("Cached mentions not found; running extraction.")
                reuse_mentions = False

    if not reuse_mentions:
        recognizer = EntityRecognizer()
        mention_to_chunks: Dict[str, set] = defaultdict(set)

        logger.info("Extracting entities from %d chunks...", len(chunks))
        for chunk in tqdm(chunks, desc="Extracting entities", unit="chunk", ncols=80):
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text", "")
            mentions = recognizer.extract(chunk_id, text)
            all_mentions.extend(mentions)
            for m in mentions:
                mention_to_chunks[_normalize(m.text)].add(chunk_id)
        # Save mentions immediately so we can resume later
        mentions_list = []
        for m in all_mentions:
            mentions_list.append(
                {
                    "mention_text": m.text,
                    "normalized": _normalize(m.text),
                    "canonical": m.text,  # temporary before resolution
                    "type": m.type,
                    "chunk_id": m.chunk_id,
                    "start": m.start,
                    "end": m.end,
                }
            )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "entity_mentions.json").write_text(json.dumps(mentions_list, indent=2), encoding="utf-8")
        save_mentions_checkpoint()
    
    logger.info("Resolving aliases (per type)...")
    resolver.add_mentions(all_mentions)
    canonical_map = resolver.resolve(all_mentions, chunks)

    # Build entity mapping
    entity_to_mentions: Dict[str, List[str]] = defaultdict(list)
    entity_to_type: Dict[str, str] = {}
    entity_mention_count: Dict[str, int] = defaultdict(int)
    entity_chunk_ids: Dict[str, set] = defaultdict(set)
    entity_first_last: Dict[str, tuple] = {}

    for m in all_mentions:
        norm = _normalize(m.text)
        canonical = canonical_map.get(norm, m.text)
        entity_id = resolver.build_entity_id(canonical, m.type)
        entity_to_type[entity_id] = m.type
        entity_to_mentions[entity_id].append(m.text)
        entity_mention_count[entity_id] += 1
        entity_chunk_ids[entity_id].add(m.chunk_id)
        if entity_id not in entity_first_last:
            entity_first_last[entity_id] = (m.chunk_id, m.chunk_id)
        else:
            _, last = entity_first_last[entity_id]
            entity_first_last[entity_id] = (entity_first_last[entity_id][0], m.chunk_id)

    logger.info("Resolved %d unique mentions into %d canonical entities", len(set(_normalize(m.text) for m in all_mentions)), len(entity_to_type))
    return (
        entity_to_type,
        entity_to_mentions,
        entity_mention_count,
        entity_chunk_ids,
        entity_first_last,
        canonical_map,
        all_mentions,
    )


def save_outputs(
    entity_to_type: Dict[str, str],
    entity_to_mentions: Dict[str, List[str]],
    entity_mention_count: Dict[str, int],
    entity_chunk_ids: Dict[str, set],
    entity_first_last: Dict[str, tuple],
    canonical_map: Dict[str, str],
    all_mentions: List[EntityMention],
) -> None:
    """Write output JSON files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # entities.json
    entities = []
    for entity_id in sorted(entity_to_type.keys()):
        entities.append(
            {
                "entity_id": entity_id,
                "type": entity_to_type[entity_id],
                "mention_count": entity_mention_count[entity_id],
                "first_chunk": entity_first_last[entity_id][0],
                "last_chunk": entity_first_last[entity_id][1],
                "chunk_ids": sorted(entity_chunk_ids[entity_id]),
            }
        )
    (OUTPUT_DIR / "entities.json").write_text(json.dumps(entities, indent=2), encoding="utf-8")

    # entity_aliases.json
    aliases_grouped: Dict[str, Dict[str, object]] = {}
    for entity_id in entity_to_type:
        mention_variants = list(set(entity_to_mentions[entity_id]))
        mention_variants.sort(key=lambda x: entity_mention_count[entity_id], reverse=True)
        aliases_grouped[entity_id] = {
            "type": entity_to_type[entity_id],
            "canonical": mention_variants[0],
            "aliases": mention_variants,
        }
    (OUTPUT_DIR / "entity_aliases.json").write_text(json.dumps(aliases_grouped, indent=2), encoding="utf-8")

    # entity_mentions.json (rewrite to keep canonical updates)
    mentions_list = []
    for m in all_mentions:
        norm = _normalize(m.text)
        canonical = canonical_map.get(norm, m.text)
        mentions_list.append(
            {
                "mention_text": m.text,
                "normalized": norm,
                "canonical": canonical,
                "type": m.type,
                "chunk_id": m.chunk_id,
                "start": m.start,
                "end": m.end,
            }
        )
    (OUTPUT_DIR / "entity_mentions.json").write_text(json.dumps(mentions_list, indent=2), encoding="utf-8")
    save_mentions_checkpoint()

    # phase4a_stats.json
    stats = {
        "total_mentions": len(all_mentions),
        "unique_mentions": len(set(_normalize(m.text) for m in all_mentions)),
        "entities": len(entity_to_type),
        "entity_types": dict(sorted(set((v, list(entity_to_type.values()).count(v)) for v in set(entity_to_type.values())))),
    }
    (OUTPUT_DIR / "phase4a_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    save_checkpoint()


def run_pipeline(force: bool = False, stage: str = "all") -> None:
    if stage == "all" and should_skip(force):
        return

    chunks = load_chunks(CHUNKS_PATH)
    _seed_aliases = load_seed_aliases()

    if stage == "extract":
        extract_entities(chunks, reuse_mentions=False if force else True)
        logger.info("Extraction complete. Mentions saved at %s", MENTIONS_PATH)
        return

    if stage == "resolve":
        if not mentions_checkpoint_current() or not MENTIONS_PATH.exists():
            raise RuntimeError("Mentions cache missing or stale. Run stage 'extract' first or use --force.")
        # Skip extraction; load cached
        all_mentions = load_cached_mentions()
        # Minimal fields for downstream
        resolver = AliasResolver(similarity_threshold=0.88, min_mention_frequency=2)
        resolver.add_mentions(all_mentions)
        canonical_map = resolver.resolve(all_mentions, chunks)
        # Build simple maps from cached mentions
        entity_to_mentions: Dict[str, List[str]] = defaultdict(list)
        entity_to_type: Dict[str, str] = {}
        entity_mention_count: Dict[str, int] = defaultdict(int)
        entity_chunk_ids: Dict[str, set] = defaultdict(set)
        entity_first_last: Dict[str, tuple] = {}
        for m in all_mentions:
            norm = _normalize(m.text)
            canonical = canonical_map.get(norm, m.text)
            entity_id = resolver.build_entity_id(canonical, m.type)
            entity_to_type[entity_id] = m.type
            entity_to_mentions[entity_id].append(m.text)
            entity_mention_count[entity_id] += 1
            entity_chunk_ids[entity_id].add(m.chunk_id)
            if entity_id not in entity_first_last:
                entity_first_last[entity_id] = (m.chunk_id, m.chunk_id)
            else:
                entity_first_last[entity_id] = (entity_first_last[entity_id][0], m.chunk_id)

        save_outputs(
            entity_to_type,
            entity_to_mentions,
            entity_mention_count,
            entity_chunk_ids,
            entity_first_last,
            canonical_map,
            all_mentions,
        )
        logger.info("Resolution complete. Outputs in %s", OUTPUT_DIR)
        return

    # stage == "all"
    (
        entity_to_type,
        entity_to_mentions,
        entity_mention_count,
        entity_chunk_ids,
        entity_first_last,
        canonical_map,
        all_mentions,
    ) = extract_entities(chunks, reuse_mentions=not force)

    save_outputs(
        entity_to_type,
        entity_to_mentions,
        entity_mention_count,
        entity_chunk_ids,
        entity_first_last,
        canonical_map,
        all_mentions,
    )
    logger.info("Phase 4A complete. Outputs in %s", OUTPUT_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4A: Entity extraction and alias resolution")
    parser.add_argument("--force", action="store_true", help="Rebuild even if checkpoint is current")
    parser.add_argument(
        "--stage",
        choices=["all", "extract", "resolve"],
        default="all",
        help="Run only extraction, only resolution (requires cached mentions), or full pipeline",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(force=args.force, stage=args.stage)
