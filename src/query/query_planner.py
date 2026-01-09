"""Phase 5A: Query Understanding & Planning.

Deterministically converts a natural language question into a structured
QueryPlan suitable for execution against the event-centric KG. No graph
traversal or LLM calls are performed here.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple


INTENT_EVENT_MAP: Dict[str, List[str]] = {
    "FACT": ["KILL", "DEATH", "BATTLE", "CORONATION", "APPOINTED_AS", "CURSE"],
    "CAUSAL": ["SUPPORTED", "DEFENDED", "VOW", "COMMAND"],
    "TEMPORAL": ["DEATH", "BATTLE", "RETREATED"],
    "MULTI_HOP": ["KILL", "DEATH", "BOON", "CURSE"],
}

# Priority order when multiple intents match
INTENT_PRIORITY = ["CAUSAL", "TEMPORAL", "MULTI_HOP", "FACT"]

# Semantic triggers for MULTI_HOP intent (checked before FACT)
MULTI_HOP_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bbenefit(?:ed|s)?\b", re.IGNORECASE),
    re.compile(r"\bconsequence(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bimpact(?:ed|s)?\b", re.IGNORECASE),
    re.compile(r"\bled to\b", re.IGNORECASE),
    re.compile(r"\bresult(?:ed)? in\b", re.IGNORECASE),
    re.compile(r"\bgained\b", re.IGNORECASE),
    re.compile(r"\badvantage\b", re.IGNORECASE),
]

PRONOUNS: Set[str] = {
    "i",
    "me",
    "we",
    "you",
    "he",
    "she",
    "they",
    "them",
    "him",
    "her",
    "it",
    "his",
    "hers",
    "their",
    "theirs",
    "your",
    "yours",
    "who",
    "whom",
}

ENTITY_TYPE_PRIORITY = ["PERSON", "GROUP", "PLACE", "TIME", "LITERAL"]


@dataclass
class QueryPlan:
    """Structured plan for downstream graph execution."""

    intent: str
    seed_entities: List[str]
    target_event_types: List[str]
    constraints: Dict[str, Any]
    traversal_depth: int
    debug: Dict[str, Any] = field(default_factory=dict)


class IntentClassifier:
    """Rule-based intent classifier (deterministic, debuggable).
    
    Intent priority order (highest to lowest):
        1. CAUSAL (why/because/reason)
        2. TEMPORAL (before/after/during/first/last)
        3. MULTI_HOP (benefit/consequence/impact/result/gain) — checked BEFORE FACT
        4. FACT (who/what/when) — fallback
    """

    RULES: Dict[str, List[re.Pattern]] = {
        "FACT": [re.compile(r"\bwho\b|\bwhom\b|\bwhat\b|\bwhen\b", re.IGNORECASE)],
        "CAUSAL": [re.compile(r"\bwhy\b|\bbecause\b|\breason\b", re.IGNORECASE)],
        "TEMPORAL": [
            re.compile(r"\bbefore\b|\bafter\b|\bduring\b|\bfirst\b|\blast\b", re.IGNORECASE)
        ],
    }

    @classmethod
    def classify(cls, question: str) -> Dict[str, Any]:
        """Classify question intent with semantic trigger check for MULTI_HOP.
        
        MULTI_HOP is checked independently before applying FACT fallback,
        ensuring benefit/consequence questions aren't misclassified as FACT.
        """
        text = question.strip().lower()
        matched: List[Tuple[str, str]] = []

        # First, check for MULTI_HOP semantic triggers (benefit/consequence/impact/result)
        # This must happen BEFORE FACT classification to avoid misclassifying "Who benefited?"
        for pat in MULTI_HOP_PATTERNS:
            if pat.search(text):
                matched.append(("MULTI_HOP", pat.pattern))

        # Then check standard rules
        for intent, patterns in cls.RULES.items():
            for pat in patterns:
                if pat.search(text):
                    matched.append((intent, pat.pattern))

        if matched:
            intents_found = {intent for intent, _ in matched}
            # Apply priority: CAUSAL > TEMPORAL > MULTI_HOP > FACT
            for intent in INTENT_PRIORITY:
                if intent in intents_found:
                    chosen_intent = intent
                    break
        else:
            chosen_intent = "FACT"  # default fallback

        return {"intent": chosen_intent, "matched_rules": matched}


def _build_alias_index(entity_registry: Dict[str, Dict[str, Any]]) -> Dict[str, Tuple[str, str]]:
    """Builds alias -> (canonical_name, entity_type) index from registry."""
    index: Dict[str, Tuple[str, str]] = {}
    for ent in entity_registry.values():
        canonical = ent.get("canonical_name", "").lower()
        etype = ent.get("entity_type") or ent.get("type") or ""
        if canonical:
            index[canonical] = (canonical, etype)
        for alias in ent.get("aliases", []) or []:
            alias_norm = str(alias).lower()
            if alias_norm:
                index[alias_norm] = (canonical, etype)
    return index


def extract_seed_entities(question: str, entity_registry: Dict[str, Dict[str, Any]]) -> List[str]:
    """Extract seed entities present in the question, constrained to known registry entries.

    - Case-insensitive match
    - Prefers PERSON > GROUP > PLACE > TIME > LITERAL when duplicates share surface text
    - Skips pronouns
    """
    text = question.lower()
    alias_index = _build_alias_index(entity_registry)
    matches: Dict[str, Tuple[str, str]] = {}

    def priority(entity_type: str) -> int:
        return ENTITY_TYPE_PRIORITY.index(entity_type) if entity_type in ENTITY_TYPE_PRIORITY else len(ENTITY_TYPE_PRIORITY)

    # Simple token/phrase scan using word-boundary regex for each alias
    for alias, (canonical, etype) in alias_index.items():
        if not alias or alias in PRONOUNS:
            continue
        pattern = rf"\b{re.escape(alias)}\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            # If already matched, keep the higher-priority type
            if canonical in matches:
                _, existing_type = matches[canonical]
                if priority(etype) < priority(existing_type):
                    matches[canonical] = (canonical, etype)
            else:
                matches[canonical] = (canonical, etype)

    # Sort by type priority then by name for determinism
    sorted_hits = sorted(matches.values(), key=lambda x: (priority(x[1]), x[0]))
    return [canonical for canonical, _ in sorted_hits]


def infer_target_event_types(intent: str, question: str) -> Tuple[List[str], Dict[str, Any]]:
    """Determine target event types based on intent and lexical cues."""
    debug: Dict[str, Any] = {"base_map": INTENT_EVENT_MAP.get(intent, [])}
    types = list(INTENT_EVENT_MAP.get(intent, []))

    q_lower = question.lower()
    lexical_hits: List[str] = []
    if re.search(r"\bkilled\b|\bslew\b|\bslain\b", q_lower):
        # Prioritize KILL
        if "KILL" in types:
            types.remove("KILL")
        types.insert(0, "KILL")
        lexical_hits.append("kill/verb")
    if re.search(r"\bcurse(?:d|s)?\b", q_lower):
        # Curse queries should target CURSE events only
        types = ["CURSE"]
        lexical_hits.append("curse/verb")
    if re.search(r"\bwhy\b", q_lower):
        # Causal questions often don't need DEATH unless explicitly implied
        if intent == "CAUSAL" and "DEATH" in types:
            types = [t for t in types if t != "DEATH"]
            lexical_hits.append("drop DEATH for causal why")

    debug["lexical_hits"] = lexical_hits
    debug["final_types"] = types
    return types, debug


def infer_constraints(question: str, intent: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Infer constraints like agent requirement, causal chain, temporal order."""
    q_lower = question.lower()
    constraints = {
        "agent_required": False,
        "temporal_order": None,  # BEFORE | AFTER | DURING | FIRST | LAST
        "causal_chain": False,
    }
    debug: Dict[str, Any] = {}

    if re.search(r"\bkilled\b|\bslew\b", q_lower):
        constraints["agent_required"] = True
        debug["agent_required_reason"] = "kill verb present"
    if "why" in q_lower:
        constraints["causal_chain"] = True
        debug["causal_chain_reason"] = "why present"
    if re.search(r"\bafter\b", q_lower):
        constraints["temporal_order"] = "AFTER"
    elif re.search(r"\bbefore\b", q_lower):
        constraints["temporal_order"] = "BEFORE"
    elif re.search(r"\bduring\b", q_lower):
        constraints["temporal_order"] = "DURING"
    elif re.search(r"\bfirst\b", q_lower):
        constraints["temporal_order"] = "FIRST"
    elif re.search(r"\blast\b", q_lower):
        constraints["temporal_order"] = "LAST"

    if constraints["temporal_order"]:
        debug["temporal_order_reason"] = constraints["temporal_order"]

    return constraints, debug


def infer_traversal_depth(intent: str) -> int:
    """Infer required traversal depth based on intent type.
    
    MULTI_HOP questions (benefit/consequence) require depth=2 to explore
    both the triggering event and its consequences/beneficiaries.
    """
    if intent == "FACT":
        return 1
    if intent == "CAUSAL":
        return 2
    if intent == "TEMPORAL":
        return 2
    if intent == "MULTI_HOP":
        return 2  # Hard-set for consequence/benefit queries
    return 1


def build_query_plan(question: str, entity_registry: Dict[str, Dict[str, Any]]) -> QueryPlan:
    """Main entry: produce a QueryPlan from a natural language question."""
    intent_info = IntentClassifier.classify(question)
    intent = intent_info["intent"]

    seed_entities = extract_seed_entities(question, entity_registry)
    target_event_types, event_debug = infer_target_event_types(intent, question)
    constraints, constraints_debug = infer_constraints(question, intent)
    depth = infer_traversal_depth(intent)

    debug = {
        "intent": intent_info,
        "entities": seed_entities,
        "event_debug": event_debug,
        "constraints_debug": constraints_debug,
    }

    return QueryPlan(
        intent=intent,
        seed_entities=seed_entities,
        target_event_types=target_event_types,
        constraints=constraints,
        traversal_depth=depth,
        debug=debug,
    )


def load_entity_registry(registry_path: str) -> Dict[str, Dict[str, Any]]:
    """Load Phase 4 entity_registry.json.

    Returns empty dict if file is missing; caller may choose to fallback to mocks for tests.
    """
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Phase 4 schema stores entities under the "entities" key
            if isinstance(data, dict) and "entities" in data:
                return data.get("entities", {})
            return data
    except FileNotFoundError:
        return {}


# --------------------------- Inline Tests ---------------------------

def _mock_registry() -> Dict[str, Dict[str, Any]]:
    """Small deterministic registry for inline tests."""
    return {
        "E1": {
            "canonical_name": "karna",
            "entity_type": "PERSON",
            "aliases": ["Radheya", "Vasusena"],
        },
        "E2": {
            "canonical_name": "bhishma",
            "entity_type": "PERSON",
            "aliases": [],
        },
        "E3": {
            "canonical_name": "duryodhana",
            "entity_type": "PERSON",
            "aliases": ["suyodhana"],
        },
        "E4": {
            "canonical_name": "abhimanyu",
            "entity_type": "PERSON",
            "aliases": [],
        },
        "E5": {
            "canonical_name": "drona",
            "entity_type": "PERSON",
            "aliases": [],
        },
    }


def _run_inline_tests() -> None:
    registry = _mock_registry()
    tests = [
        "Who killed Karna?",
        "Why did Bhishma support Duryodhana?",
        "What happened after Abhimanyu's death?",
        "Who benefited from Drona's death?",
    ]

    for q in tests:
        plan = build_query_plan(q, registry)
        print("Question:", q)
        print(plan)
        print("- intent:", plan.intent)
        print("- entities:", plan.seed_entities)
        print("- target_event_types:", plan.target_event_types)
        print("- constraints:", plan.constraints)
        print("- depth:", plan.traversal_depth)
        print("- debug:", plan.debug)
        print("-")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 5A Query Planner")
    parser.add_argument("question", nargs="?", help="Natural language question to plan")
    parser.add_argument(
        "--registry_path",
        type=str,
        default="data/kg/entity_registry.json",
        help="Path to entity_registry.json from Phase 4",
    )
    args = parser.parse_args()

    if args.question:
        # Load real registry if available; fallback to mock only if file missing
        try:
            with open(args.registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except FileNotFoundError:
            registry = _mock_registry()

        plan = build_query_plan(args.question, registry)
        print(plan)
    else:
        _run_inline_tests()
