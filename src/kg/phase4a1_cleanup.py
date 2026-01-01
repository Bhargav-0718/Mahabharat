"""Phase 4A.1: Entity cleanup and consolidation.

This module performs deterministic cleanup on Phase 4A outputs:
- Normalizes aliases
- Removes noisy entities
- Resolves type conflicts
- Consolidates duplicate entities
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/kg")
ENTITIES_INPUT = DATA_DIR / "entities.json"
ALIASES_INPUT = DATA_DIR / "entity_aliases.json"
MENTIONS_INPUT = DATA_DIR / "entity_mentions.json"

ENTITIES_CLEAN = DATA_DIR / "entities_clean.json"
MENTIONS_CLEAN = DATA_DIR / "mentions_clean.json"
CLEANUP_REPORT = DATA_DIR / "entity_cleanup_report.json"

# Common noun blacklist
COMMON_NOUN_BLACKLIST = {"rishi", "brahmana", "king", "princes", "warrior", "sage", "gods", "celestials", "deva"}

# Type priority (lower = higher priority)
TYPE_PRIORITY = {"PERSON": 0, "GROUP": 1, "PLACE": 2, "WEAPON": 3, "TIME": 4, "UNKNOWN": 5}


def normalize_alias(text: str) -> str:
    """Normalize alias for matching.
    
    Performs:
    - lowercase
    - remove URLs
    - strip leading/trailing punctuation
    - remove possessives ('s)
    - remove trailing hyphens
    - collapse whitespace
    """
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Strip leading/trailing punctuation
    text = text.strip(".,;:'\"-!?()[]{}") 
    # Remove possessives
    text = re.sub(r"'s\b", "", text)
    # Remove trailing hyphens
    text = text.rstrip("-")
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def is_url_mention(text: str) -> bool:
    """Check if mention contains URL."""
    return bool(re.search(r"https?://|www\.", text))


def is_section_header(text: str) -> bool:
    """Check if mention is a section/book header."""
    return bool(re.search(r"\b(SECTION|BOOK|PARVA|ADI|P\d{2}-S)\b", text, re.IGNORECASE))


def is_all_caps_short(text: str) -> bool:
    """Check if all caps and length <= 6."""
    return text.isupper() and len(text) <= 6 and len(text) > 0


def is_only_punctuation_digits(text: str) -> bool:
    """Check if only punctuation/digits (no letters)."""
    return not bool(re.search(r"[a-zA-Z]", text))


def is_narrator_dialogue(text: str, context: str = "") -> bool:
    """Check if text is narrator/dialogue prefix."""
    patterns = [
        r"\b\w+\s+(?:said|continued|replied|answered),?--",
        r"\b(?:Vaisampayana|Sauti|Sanjaya|Dhritarashtra|Yudhishthira)",
    ]
    full = f"{text} {context}".lower()
    return any(re.search(p, full, re.IGNORECASE) for p in patterns)


def strip_narrator_prefix(text: str, context: str = "") -> str:
    """Try to strip narrator prefix from text."""
    # Pattern: "X said,-- text" or "X continued,-- text"
    # Match at start of text or context
    match = re.match(r"^(.+?)\s+(?:said|continued|replied|answered),?--\s*(.*)$", text, re.IGNORECASE)
    if match and match.group(2):
        return match.group(2)
    return text


def load_phase4a_outputs() -> Tuple[Dict, Dict, List[Dict]]:
    """Load Phase 4A outputs.
    
    Returns:
        (entities_dict, aliases_dict, mentions_list)
    """
    # Load entities.json
    if not ENTITIES_INPUT.exists():
        raise FileNotFoundError(f"Not found: {ENTITIES_INPUT}")
    with open(ENTITIES_INPUT, encoding="utf-8") as f:
        entities_list = json.load(f)
    entities = {e["entity_id"]: e for e in entities_list}

    # Load entity_aliases.json
    if not ALIASES_INPUT.exists():
        raise FileNotFoundError(f"Not found: {ALIASES_INPUT}")
    with open(ALIASES_INPUT, encoding="utf-8") as f:
        aliases = json.load(f)

    # Load entity_mentions.json
    if not MENTIONS_INPUT.exists():
        raise FileNotFoundError(f"Not found: {MENTIONS_INPUT}")
    with open(MENTIONS_INPUT, encoding="utf-8") as f:
        mentions = json.load(f)

    logger.info(f"Loaded Phase 4A outputs: {len(entities)} entities, {len(mentions)} mentions")
    return entities, aliases, mentions


def filter_noise_mentions(
    mentions: List[Dict], entities: Dict, aliases: Dict
) -> Tuple[List[Dict], Dict[str, int]]:
    """Apply hard noise filters to mentions.
    
    Returns:
        (filtered_mentions, drop_stats)
    """
    drop_reasons = defaultdict(int)
    filtered = []

    for mention in mentions:
        text = mention.get("mention_text", "")

        # Check all noise filters
        if is_url_mention(text):
            drop_reasons["url_mention"] += 1
            continue
        if is_section_header(text):
            drop_reasons["section_header"] += 1
            continue
        if is_all_caps_short(text):
            drop_reasons["all_caps_short"] += 1
            continue
        if is_only_punctuation_digits(text):
            drop_reasons["punctuation_only"] += 1
            continue

        filtered.append(mention)

    logger.info(f"Filtered noise mentions: {len(mentions) - len(filtered)} dropped")
    for reason, count in sorted(drop_reasons.items()):
        logger.info(f"  {reason}: {count}")

    return filtered, drop_reasons


def cleanup_narrator_dialogue(mentions: List[Dict]) -> Tuple[List[Dict], int]:
    """Clean narrator/dialogue prefixes from mention text.
    
    Returns:
        (cleaned_mentions, dropped_count)
    """
    dropped = 0
    cleaned = []

    for mention in mentions:
        text = mention.get("mention_text", "")
        context = mention.get("context", "")

        if is_narrator_dialogue(text, context):
            stripped = strip_narrator_prefix(text, context)
            norm = normalize_alias(stripped)

            # If nothing meaningful remains, drop
            if not norm:
                dropped += 1
                continue

            # Update mention
            mention = mention.copy()
            mention["mention_text"] = stripped

        cleaned.append(mention)

    logger.info(f"Cleaned narrator/dialogue: {dropped} dropped")
    return cleaned, dropped


def filter_common_nouns(
    entities: Dict, aliases: Dict, mentions: List[Dict]
) -> Tuple[Dict, int]:
    """Filter out common noun entities without proper context.
    
    Returns:
        (filtered_entities, dropped_count)
    """
    dropped = 0
    filtered = {}

    # Build entity -> mentions map
    entity_mentions = defaultdict(list)
    for mention in mentions:
        eid = mention.get("entity_id", "")
        if eid:
            entity_mentions[eid].append(mention)

    for eid, entity in entities.items():
        # Try to get canonical name
        canonical = ""
        if eid in aliases:
            canonical = aliases[eid].get("canonical", "")
        if not canonical:
            canonical = entity.get("entity_id", "")

        norm = normalize_alias(canonical)

        # Check if in blacklist
        if norm in COMMON_NOUN_BLACKLIST:
            # Check if any mention has proper noun context
            ment_list = entity_mentions.get(eid, [])
            has_proper_context = False

            for m in ment_list:
                # Check if mention text is capitalized (likely proper noun)
                text = m.get("mention_text", "")
                if text and text[0].isupper() and normalize_alias(text) != norm:
                    has_proper_context = True
                    break

            if not has_proper_context:
                dropped += 1
                continue

        filtered[eid] = entity

    logger.info(f"Filtered common nouns: {dropped} dropped")
    return filtered, dropped


def arbitrate_entity_types(
    entities: Dict, aliases: Dict, mentions: List[Dict]
) -> Tuple[Dict, int]:
    """Resolve entity type conflicts.
    
    Uses priority: PERSON > GROUP > PLACE > WEAPON > TIME > UNKNOWN
    
    Returns:
        (arbitrated_entities, conflicts_resolved)
    """
    # Build entity -> mention_types map
    entity_types = defaultdict(lambda: defaultdict(int))
    for mention in mentions:
        eid = mention.get("entity_id", "")
        etype = mention.get("type", "UNKNOWN")
        if eid:
            entity_types[eid][etype] += 1

    arbitrated = {}
    conflicts_resolved = 0

    for eid, entity in entities.items():
        entity = entity.copy()
        mention_type_counts = entity_types.get(eid, {})

        if mention_type_counts:
            unique_types = set(mention_type_counts.keys())

            if len(unique_types) > 1:
                conflicts_resolved += 1
                # Pick highest priority (lowest number)
                best_type = min(unique_types, key=lambda t: TYPE_PRIORITY.get(t, 999))
                entity["type"] = best_type
                logger.debug(f"Entity {eid}: {dict(mention_type_counts)} -> {best_type}")
            elif len(unique_types) == 1:
                entity["type"] = list(unique_types)[0]

        arbitrated[eid] = entity

    logger.info(f"Entity type conflicts resolved: {conflicts_resolved}")
    return arbitrated, conflicts_resolved


def consolidate_entities(
    entities: Dict, aliases: Dict, mentions: List[Dict]
) -> Tuple[Dict, List[Dict], int]:
    """Merge entities by normalized canonical name.
    
    Returns:
        (consolidated_entities, updated_mentions, merged_count)
    """
    # Map: normalized_name -> [entity_ids]
    norm_to_eids = defaultdict(list)
    for eid in entities.keys():
        canonical = ""
        if eid in aliases:
            canonical = aliases[eid].get("canonical", "")
        if not canonical:
            canonical = eid

        norm = normalize_alias(canonical)
        if norm:
            norm_to_eids[norm].append(eid)

    # Build merge map: old_eid -> canonical_eid
    merge_map = {}
    keep_eids = set(entities.keys())
    merged_count = 0

    for norm_name, eids in norm_to_eids.items():
        if len(eids) > 1:
            # Keep earliest (by entity_id lexicographic order)
            eids_sorted = sorted(eids)
            canonical_eid = eids_sorted[0]
            for other_eid in eids_sorted[1:]:
                merge_map[other_eid] = canonical_eid
                keep_eids.discard(other_eid)
                merged_count += 1
                logger.debug(f"Merge {other_eid} into {canonical_eid}")

    logger.info(f"Entity consolidation: {merged_count} merged")

    # Apply merge to entities: collect aliases
    consolidated = {}
    for eid in keep_eids:
        entity = entities[eid].copy()

        # Collect all aliases from merged entities
        all_aliases = set()
        if eid in aliases:
            all_aliases.update(aliases[eid].get("aliases", []))
        for other_eid, target in merge_map.items():
            if target == eid:
                if other_eid in aliases:
                    all_aliases.update(aliases[other_eid].get("aliases", []))

        # Update entity with merged aliases if applicable
        entity["aliases"] = sorted(list(all_aliases))
        consolidated[eid] = entity

    # Update mention entity_ids
    updated_mentions = []
    for mention in mentions:
        mention = mention.copy()
        eid = mention.get("entity_id", "")
        if eid in merge_map:
            mention["entity_id"] = merge_map[eid]
        updated_mentions.append(mention)

    return consolidated, updated_mentions, merged_count


def cleanup_pipeline() -> None:
    """Execute full cleanup pipeline."""
    logger.info("=" * 80)
    logger.info("Phase 4A.1: Entity Cleanup & Consolidation")
    logger.info("=" * 80)

    # Load Phase 4A outputs
    entities, aliases, mentions = load_phase4a_outputs()

    entities_before = len(entities)
    mentions_before = len(mentions)

    stats = {
        "entities_before": entities_before,
        "mentions_before": mentions_before,
        "drop_reasons": {},
    }

    # Step 1: Filter noise mentions
    mentions, drop_stats_noise = filter_noise_mentions(mentions, entities, aliases)
    stats["drop_reasons"].update(drop_stats_noise)

    # Step 2: Clean narrator/dialogue
    mentions, dropped_narrator = cleanup_narrator_dialogue(mentions)
    stats["drop_reasons"]["narrator_dialogue"] = dropped_narrator

    # Step 3: Filter common nouns
    entities, dropped_common = filter_common_nouns(entities, aliases, mentions)
    stats["drop_reasons"]["common_nouns"] = dropped_common

    # Step 4: Arbitrate types
    entities, conflicts_resolved = arbitrate_entity_types(entities, aliases, mentions)
    stats["type_conflicts_resolved"] = conflicts_resolved

    # Step 5: Consolidate entities
    entities, mentions, merged_count = consolidate_entities(entities, aliases, mentions)

    # Final counts
    entities_after = len(entities)
    mentions_after = len(mentions)

    logger.info("=" * 80)
    logger.info(f"Entities: {entities_before} → {entities_after} (-{entities_before - entities_after})")
    logger.info(f"Mentions: {mentions_before} → {mentions_after} (-{mentions_before - mentions_after})")
    logger.info(f"Entities consolidated: {merged_count}")
    logger.info("=" * 80)

    # Prepare outputs
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Convert entities dict to list for JSON
    entities_list = list(entities.values())

    # Write entities_clean.json
    with open(ENTITIES_CLEAN, "w", encoding="utf-8") as f:
        json.dump(entities_list, f, indent=2)
    logger.info(f"Wrote {len(entities_list)} entities to {ENTITIES_CLEAN}")

    # Write mentions_clean.json
    with open(MENTIONS_CLEAN, "w", encoding="utf-8") as f:
        json.dump(mentions, f, indent=2)
    logger.info(f"Wrote {len(mentions)} mentions to {MENTIONS_CLEAN}")

    # Generate cleanup report
    stats.update(
        {
            "entities_after": entities_after,
            "mentions_after": mentions_after,
            "dropped_entities_count": entities_before - entities_after,
            "dropped_mentions_count": mentions_before - mentions_after,
        }
    )

    with open(CLEANUP_REPORT, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Wrote cleanup report to {CLEANUP_REPORT}")

    logger.info("Phase 4A.1 cleanup complete.")


if __name__ == "__main__":
    cleanup_pipeline()
