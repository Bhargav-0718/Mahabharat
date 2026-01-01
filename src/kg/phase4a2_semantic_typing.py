"""Phase 4A.2: Semantic typing and ontology refinement.

This module refines entity types using deterministic semantic rules:
- Hard ontology overrides (TEXT, PARVA, RITUAL, MATERIAL)
- Vocative/title detection
- Role locking for major characters
- Polysemy resolution using context and frequency
- Audit trail of all type changes
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
ENTITIES_INPUT = DATA_DIR / "entities_clean.json"
MENTIONS_INPUT = DATA_DIR / "mentions_clean.json"

ENTITIES_OUTPUT = DATA_DIR / "entities_semantic.json"
MENTIONS_OUTPUT = DATA_DIR / "mentions_semantic.json"
ONTOLOGY_STATS = DATA_DIR / "ontology_stats.json"
TYPE_CHANGES_LOG = DATA_DIR / "entity_type_changes.json"

# Extended ontology
ONTOLOGY_TYPES = {
    "PERSON",   # Individual characters
    "GROUP",    # Collectives, families, armies
    "PLACE",    # Locations, kingdoms
    "WEAPON",   # Arms, tools
    "TEXT",     # Sacred texts (Mahabharata, Ramayana, etc.)
    "PARVA",    # Books/sections of Mahabharata
    "RITUAL",   # Rituals, ceremonies
    "MATERIAL", # Physical objects, food, materials
    "CONCEPT",  # Abstract ideas
    "TITLE",    # Vocative/titles
}

# Hard ontology overrides
HARD_TEXT_ENTITIES = {"mahabharata", "ramayana", "harivansa", "purana", "puranas", "veda", "vedas"}

HARD_PARVA_ENTITIES = {
    "sambhava", "jatugrihadaha", "hidimbabadha", "bakabadha", "chitraratha",
    "swayamvara", "vaivahika", "viduragamana", "rajyalabha", "arjunabanavasa",
    "subhadraharana", "khandavadaha", "mayavisayana", "sabhaparva", "aranyaka",
    "vairata", "udyoga", "bhishmaparva", "dronaparva", "karnaparva", "salyaparva",
    "striparva", "santaparva", "asvamedhaparva", "asramvasika", "mausala",
    "mahaprasthanika", "svargarohanika",
}

HARD_RITUAL_ENTITIES = {
    "asvamedha", "asvamedhika", "homa", "atiratra", "rajasuya", "soma",
    "sacrifice", "yajna", "abhiseka",
}

HARD_MATERIAL_ENTITIES = {
    "gold", "honey", "garlands", "ghee", "food", "flower", "flowers",
    "cloth", "garland", "silk", "oil", "perfume", "incense", "fragrance",
}

# Major characters (force PERSON type)
MAJOR_CHARACTERS = {
    "krishna", "arjuna", "bhima", "yudhishthira", "draupadi",
    "duryodhana", "karna", "bhishma", "drona", "sahadeva", "nakula",
    "pandu", "kunti", "dhritarashtra", "vidura", "gandhari",
}

# Vocative markers (within ±3 tokens)
VOCATIVE_MARKERS = {"o", "thou", "thee", "hark"}


def normalize_for_lookup(text: str) -> str:
    """Normalize text for dictionary lookups."""
    return re.sub(r"[^a-z0-9]", "", text.lower()).strip()


def load_phase4a1_outputs() -> Tuple[List[Dict], List[Dict]]:
    """Load Phase 4A.1 clean outputs."""
    if not ENTITIES_INPUT.exists():
        raise FileNotFoundError(f"Not found: {ENTITIES_INPUT}")
    if not MENTIONS_INPUT.exists():
        raise FileNotFoundError(f"Not found: {MENTIONS_INPUT}")

    with open(ENTITIES_INPUT, encoding="utf-8") as f:
        entities = json.load(f)
    with open(MENTIONS_INPUT, encoding="utf-8") as f:
        mentions = json.load(f)

    logger.info(f"Loaded Phase 4A.1 outputs: {len(entities)} entities, {len(mentions)} mentions")
    return entities, mentions


def apply_hard_ontology_overrides(
    entities: List[Dict], mentions: List[Dict]
) -> Tuple[Dict[str, str], int]:
    """Apply hard ontology overrides based on canonical names.
    
    Returns:
        (entity_id -> new_type map, count of changes)
    """
    type_overrides = {}
    changes = 0

    # Build entity_id -> canonical_name map
    entity_canon = {}
    for entity in entities:
        entity_id = entity.get("entity_id", "")
        # Try to get canonical from aliases[0] or fall back to entity_id
        aliases = entity.get("aliases", [])
        canonical = aliases[0] if aliases else entity_id
        entity_canon[entity_id] = normalize_for_lookup(canonical)

    # Apply TEXT overrides
    for entity_id, canon_norm in entity_canon.items():
        if canon_norm in HARD_TEXT_ENTITIES:
            type_overrides[entity_id] = "TEXT"
            changes += 1
            logger.debug(f"TEXT override: {entity_id}")

    # Apply PARVA overrides
    for entity_id, canon_norm in entity_canon.items():
        if entity_id in type_overrides:
            continue  # Already overridden
        if any(canon_norm.endswith(suffix) for suffix in ["parva", "ika", "vasika"]):
            type_overrides[entity_id] = "PARVA"
            changes += 1
            logger.debug(f"PARVA override: {entity_id}")

    # Apply RITUAL overrides
    for entity_id, canon_norm in entity_canon.items():
        if entity_id in type_overrides:
            continue
        if canon_norm in HARD_RITUAL_ENTITIES:
            type_overrides[entity_id] = "RITUAL"
            changes += 1
            logger.debug(f"RITUAL override: {entity_id}")

    # Apply MATERIAL overrides
    for entity_id, canon_norm in entity_canon.items():
        if entity_id in type_overrides:
            continue
        if canon_norm in HARD_MATERIAL_ENTITIES:
            type_overrides[entity_id] = "MATERIAL"
            changes += 1
            logger.debug(f"MATERIAL override: {entity_id}")

    logger.info(f"Hard ontology overrides: {changes} type changes")
    return type_overrides, changes


def detect_vocatives(mentions: List[Dict]) -> Tuple[Set[str], int]:
    """Detect mentions that appear in vocative contexts (near O, THOU, etc).
    
    Returns:
        (set of mention indices in vocative context, count)
    """
    vocative_mentions = set()
    count = 0

    for idx, mention in enumerate(mentions):
        context = mention.get("context", "").lower()
        if not context:
            continue

        # Check for vocative markers within context (or nearby in text)
        for marker in VOCATIVE_MARKERS:
            if marker in context:
                vocative_mentions.add(idx)
                count += 1
                break

    logger.info(f"Detected vocative contexts: {count} mentions")
    return vocative_mentions, count


def apply_vocative_titles(
    entities: List[Dict],
    mentions: List[Dict],
    vocative_mention_indices: Set[int],
    type_overrides: Dict[str, str],
) -> Tuple[Dict[str, str], int]:
    """Mark entities that appear in vocative contexts as TITLE.
    
    Returns:
        (updates to type_overrides, count of changes)
    """
    updates = {}
    changes = 0

    # Build entity_id -> mention indices map
    entity_to_mentions = defaultdict(list)
    for idx, mention in enumerate(mentions):
        eid = mention.get("entity_id", "")
        if eid:
            entity_to_mentions[eid].append(idx)

    # Check which entities appear in vocative contexts
    for entity_id, ment_indices in entity_to_mentions.items():
        if entity_id in type_overrides:
            continue  # Already overridden
        vocative_count = sum(1 for i in ment_indices if i in vocative_mention_indices)
        total_count = len(ment_indices)

        # If >30% of mentions are vocative, mark as TITLE
        if total_count > 0 and vocative_count / total_count > 0.3:
            updates[entity_id] = "TITLE"
            changes += 1
            logger.debug(f"Vocative title: {entity_id} ({vocative_count}/{total_count})")

    logger.info(f"Vocative title detection: {changes} entities marked TITLE")
    return updates, changes


def apply_role_locking(
    entities: List[Dict], type_overrides: Dict[str, str]
) -> Tuple[Dict[str, str], int]:
    """Force PERSON type for major characters.
    
    Returns:
        (updates to type_overrides, count of changes)
    """
    updates = {}
    changes = 0

    # Build canonical -> entity_id map for lookup
    canon_to_eid = {}
    for entity in entities:
        eid = entity.get("entity_id", "")
        aliases = entity.get("aliases", [])
        canonical = aliases[0] if aliases else ""
        canon_norm = normalize_for_lookup(canonical)
        canon_to_eid[canon_norm] = eid

    # Apply role locks
    for major_char in MAJOR_CHARACTERS:
        if major_char in canon_to_eid:
            eid = canon_to_eid[major_char]
            if eid not in type_overrides:
                updates[eid] = "PERSON"
                changes += 1
                logger.debug(f"Role lock PERSON: {major_char}")

    logger.info(f"Role locking: {changes} major characters locked to PERSON")
    return updates, changes


def resolve_polysemy(
    entities: List[Dict],
    mentions: List[Dict],
    type_overrides: Dict[str, str],
) -> Tuple[Dict[str, str], int]:
    """Use mention frequency + context to resolve polysemy.
    
    For entities with multiple types in mentions, choose based on:
    1. Frequency (most common type wins)
    2. Context (TITLE in vocatives only)
    
    Returns:
        (updates to type_overrides, count of changes)
    """
    updates = {}
    changes = 0

    # Build entity_id -> mention types map
    entity_mention_types = defaultdict(lambda: defaultdict(int))
    for mention in mentions:
        eid = mention.get("entity_id", "")
        etype = mention.get("type", "UNKNOWN")
        if eid:
            entity_mention_types[eid][etype] += 1

    # Resolve conflicts
    for entity_id, type_counts in entity_mention_types.items():
        if entity_id in type_overrides:
            continue  # Already overridden

        unique_types = set(type_counts.keys())
        if len(unique_types) <= 1:
            continue  # No conflict

        # Pick most frequent
        best_type = max(unique_types, key=lambda t: type_counts[t])
        updates[entity_id] = best_type
        changes += 1
        logger.debug(f"Polysemy resolution {entity_id}: {dict(type_counts)} → {best_type}")

    logger.info(f"Polysemy resolution: {changes} entities resolved")
    return updates, changes


def apply_type_changes(
    entities: List[Dict],
    mentions: List[Dict],
    all_overrides: Dict[str, str],
) -> Tuple[List[Dict], List[Dict], Dict]:
    """Apply all type changes and generate audit trail.
    
    Returns:
        (updated_entities, updated_mentions, audit_trail)
    """
    audit_trail = {}

    # Apply to entities
    updated_entities = []
    for entity in entities:
        entity = entity.copy()
        eid = entity.get("entity_id", "")
        old_type = entity.get("type", "UNKNOWN")

        if eid in all_overrides:
            new_type = all_overrides[eid]
            if new_type != old_type:
                entity["type"] = new_type
                audit_trail[eid] = {"old_type": old_type, "new_type": new_type}

        updated_entities.append(entity)

    # Apply to mentions (update type to match entity's new type)
    eid_to_new_type = {}
    for entity in updated_entities:
        eid = entity.get("entity_id", "")
        etype = entity.get("type", "UNKNOWN")
        eid_to_new_type[eid] = etype

    updated_mentions = []
    for mention in mentions:
        mention = mention.copy()
        eid = mention.get("entity_id", "")
        if eid in eid_to_new_type:
            mention["type"] = eid_to_new_type[eid]
        updated_mentions.append(mention)

    logger.info(f"Type changes applied: {len(audit_trail)} entities changed")
    return updated_entities, updated_mentions, audit_trail


def generate_ontology_stats(
    entities: List[Dict], mentions: List[Dict], audit_trail: Dict
) -> Dict:
    """Generate ontology statistics."""
    # Count by type
    type_counts = defaultdict(int)
    for entity in entities:
        etype = entity.get("type", "UNKNOWN")
        type_counts[etype] += 1

    # Count mentions by type
    mention_type_counts = defaultdict(int)
    for mention in mentions:
        etype = mention.get("type", "UNKNOWN")
        mention_type_counts[etype] += 1

    return {
        "total_entities": len(entities),
        "total_mentions": len(mentions),
        "entity_types": dict(sorted(type_counts.items())),
        "mention_types": dict(sorted(mention_type_counts.items())),
        "type_changes": len(audit_trail),
        "ontology_definition": sorted(list(ONTOLOGY_TYPES)),
    }


def semantic_typing_pipeline() -> None:
    """Execute full semantic typing pipeline."""
    logger.info("=" * 80)
    logger.info("Phase 4A.2: Semantic Typing & Ontology Refinement")
    logger.info("=" * 80)

    # Load Phase 4A.1 outputs
    entities, mentions = load_phase4a1_outputs()

    # Step 1: Hard ontology overrides
    overrides_hard, changes_hard = apply_hard_ontology_overrides(entities, mentions)

    # Step 2: Detect vocatives
    vocative_indices, vocative_count = detect_vocatives(mentions)

    # Step 3: Apply vocative titles
    overrides_vocative, changes_vocative = apply_vocative_titles(
        entities, mentions, vocative_indices, overrides_hard
    )
    overrides_hard.update(overrides_vocative)

    # Step 4: Role locking
    overrides_role, changes_role = apply_role_locking(entities, overrides_hard)
    overrides_hard.update(overrides_role)

    # Step 5: Polysemy resolution
    overrides_poly, changes_poly = resolve_polysemy(entities, mentions, overrides_hard)
    overrides_hard.update(overrides_poly)

    # Step 6: Apply all changes
    all_overrides = overrides_hard
    entities_updated, mentions_updated, audit = apply_type_changes(entities, mentions, all_overrides)

    # Generate statistics
    stats = generate_ontology_stats(entities_updated, mentions_updated, audit)

    # Write outputs
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(ENTITIES_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(entities_updated, f, indent=2)
    logger.info(f"Wrote {len(entities_updated)} entities to {ENTITIES_OUTPUT}")

    with open(MENTIONS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(mentions_updated, f, indent=2)
    logger.info(f"Wrote {len(mentions_updated)} mentions to {MENTIONS_OUTPUT}")

    with open(ONTOLOGY_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Wrote ontology stats to {ONTOLOGY_STATS}")

    with open(TYPE_CHANGES_LOG, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    logger.info(f"Wrote {len(audit)} type changes to {TYPE_CHANGES_LOG}")

    logger.info("=" * 80)
    logger.info(f"Total type changes: {len(audit)}")
    logger.info(f"  Hard overrides: {changes_hard}")
    logger.info(f"  Vocative titles: {changes_vocative}")
    logger.info(f"  Role locking: {changes_role}")
    logger.info(f"  Polysemy resolution: {changes_poly}")
    logger.info("=" * 80)
    logger.info("Ontology distribution:")
    for otype, count in sorted(stats["entity_types"].items()):
        logger.info(f"  {otype}: {count}")
    logger.info("=" * 80)
    logger.info("Phase 4A.2 semantic typing complete.")


if __name__ == "__main__":
    semantic_typing_pipeline()
