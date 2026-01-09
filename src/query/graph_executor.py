"""Phase 5B: Graph Query Execution Engine.

Deterministically executes a QueryPlan against the event-centric Mahabharat KG.
Routes execution by intent (FACT, TEMPORAL, CAUSAL, MULTI_HOP) with constraint
enforcement and depth-limited traversal. Returns structured QueryResult with
rich debug traces explaining selection/rejection logic.

No hallucination: returns empty results if no valid paths exist.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class QueryResult:
    """Structured result from graph query execution."""

    query_text: str
    intent: str
    found: bool  # Whether valid results were found
    seed_entities: List[str]  # Original query entities
    matched_events: List[Dict[str, Any]]  # Events that matched (with tier, type, participants)
    matched_entities: List[Dict[str, Any]]  # Entities extracted from matched events
    constraints_applied: List[str]  # Which constraints were active
    traversal_info: Dict[str, Any]  # Depth reached, path count, etc.
    debug_trace: List[str]  # Ordered log explaining why events were selected/rejected
    execution_time_ms: float = 0.0


@dataclass
class GraphState:
    """Internal state during graph traversal."""

    visited_events: Set[str]
    visited_entities: Set[str]
    current_depth: int
    max_depth: int
    paths: List[List[str]]  # List of event ID paths
    event_graph: Dict[str, Any]  # All events indexed by ID
    entity_graph: Dict[str, Any]  # All entities indexed by canonical_name
    edges_by_source: Dict[str, List[Dict[str, Any]]]  # Edges indexed by source entity
    edges_by_target: Dict[str, List[Dict[str, Any]]]  # Edges indexed by target event


class KGLoader:
    """Load KG data from JSON files."""

    @staticmethod
    def load_graphs(
        entities_path: str = "data/kg/entities.json",
        events_path: str = "data/kg/events.json",
        edges_path: str = "data/kg/edges.json",
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
        """Load all KG components.

        Returns:
            (entities_dict, events_dict, edges_list)
        """
        try:
            with open(entities_path) as f:
                entities_data = json.load(f)
            entities = entities_data.get("entities", {})

            with open(events_path) as f:
                events_data = json.load(f)
            events = events_data.get("events", {})

            with open(edges_path) as f:
                edges_data = json.load(f)
            edges = edges_data.get("edges", [])

            return entities, events, edges
        except FileNotFoundError as e:
            raise RuntimeError(f"Failed to load KG files: {e}")

    @staticmethod
    def build_entity_index(entities: Dict[str, Any]) -> Dict[str, str]:
        """Build alias -> entity_id index for case-insensitive lookup.

        Returns:
            {alias_lower: entity_id}
        """
        index: Dict[str, str] = {}
        for entity_id, entity_data in entities.items():
            canonical = entity_data.get("canonical_name", "").lower()
            if canonical:
                index[canonical] = entity_id
            for alias in entity_data.get("aliases", []) or []:
                alias_lower = str(alias).lower()
                if alias_lower:
                    index[alias_lower] = entity_id
        return index

    @staticmethod
    def build_edge_indices(
        edges: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
        """Build edge indices for fast lookup.

        Returns:
            (edges_by_source_entity, edges_by_target_event)
        """
        by_source: Dict[str, List[Dict[str, Any]]] = {}
        by_target: Dict[str, List[Dict[str, Any]]] = {}

        for edge in edges:
            source = edge.get("source_id")
            target = edge.get("target_id")

            if source:
                by_source.setdefault(source, []).append(edge)
            if target:
                by_target.setdefault(target, []).append(edge)

        return by_source, by_target


class QueryExecutor:
    """Main execution engine for structured graph queries."""

    def __init__(
        self,
        entities: Dict[str, Any],
        events: Dict[str, Any],
        edges: List[Dict[str, Any]],
    ):
        self.entities = entities
        self.events = events
        self.edges = edges

        # Build indices
        self.entity_index = KGLoader.build_entity_index(entities)
        self.edges_by_source, self.edges_by_target = KGLoader.build_edge_indices(edges)

    def execute(self, query_plan: Dict[str, Any], query_text: str = "") -> QueryResult:
        """Execute a QueryPlan against the KG.

        Args:
            query_plan: Dict with keys: intent, seed_entities, target_event_types,
                       constraints, traversal_depth
            query_text: Original question text

        Returns:
            QueryResult with matched events/entities and debug trace
        """
        import time

        start_time = time.time()

        if hasattr(query_plan, "__dataclass_fields__"):
            from dataclasses import asdict

            query_plan = asdict(query_plan)

        intent = query_plan.get("intent", "FACT")
        seed_entities = query_plan.get("seed_entities", [])
        target_event_types = query_plan.get("target_event_types", [])
        constraints = query_plan.get("constraints", {})
        max_depth = query_plan.get("traversal_depth", 1)

        debug_trace: List[str] = []
        debug_trace.append(f"[START] Executing {intent} query with max_depth={max_depth}")
        debug_trace.append(f"[SEEDS] seed_entities={seed_entities}")
        debug_trace.append(f"[TARGET] event_types={target_event_types}")
        debug_trace.append(f"[CONSTRAINTS] {constraints}")

        # Route by intent
        if intent == "FACT":
            matched_events, trace = self._execute_fact(seed_entities, target_event_types, constraints, max_depth)
            debug_trace.extend(trace)
        elif intent == "TEMPORAL":
            matched_events, trace = self._execute_temporal(
                seed_entities, target_event_types, constraints, max_depth
            )
            debug_trace.extend(trace)
        elif intent == "CAUSAL":
            matched_events, trace = self._execute_causal(
                seed_entities, target_event_types, constraints, max_depth
            )
            debug_trace.extend(trace)
        elif intent == "MULTI_HOP":
            matched_events, trace = self._execute_multi_hop(
                seed_entities, target_event_types, constraints, max_depth
            )
            debug_trace.extend(trace)
        else:
            matched_events = []
            debug_trace.append(f"[ERROR] Unknown intent: {intent}")

        # Extract entities from matched events
        matched_entities = self._extract_entities_from_events(matched_events)

        debug_trace.append(f"[RESULT] Found {len(matched_events)} events, {len(matched_entities)} entities")

        execution_time = (time.time() - start_time) * 1000

        return QueryResult(
            query_text=query_text,
            intent=intent,
            found=len(matched_events) > 0,
            seed_entities=seed_entities,
            matched_events=matched_events,
            matched_entities=matched_entities,
            constraints_applied=list(constraints.keys()),
            traversal_info={
                "max_depth": max_depth,
                "events_found": len(matched_events),
                "entities_found": len(matched_entities),
            },
            debug_trace=debug_trace,
            execution_time_ms=execution_time,
        )

    def _execute_fact(
        self,
        seed_entities: List[str],
        target_event_types: List[str],
        constraints: Dict[str, Any],
        max_depth: int,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """FACT intent: Direct lookup of events involving seed entities.

        Strategy: Find all events where a seed entity participated,
        filtered by target event types and constraints (agent_required).
        """
        trace: List[str] = []
        matched_events: List[Dict[str, Any]] = []
        matched_event_ids: Set[str] = set()

        trace.append("[FACT] Direct entity lookup")

        # Resolve seed entities to entity IDs
        resolved_entities = self._resolve_entities(seed_entities, trace)
        if not resolved_entities:
            trace.append("[FACT] No seed entities resolved")
            return [], trace

        # Find edges where these entities are sources
        for entity_id in resolved_entities:
            edges = self.edges_by_source.get(entity_id, [])
            trace.append(f"[FACT] Entity {entity_id}: {len(edges)} outgoing edges")

            for edge in edges:
                event_id = edge.get("event_id")
                if not event_id or event_id in matched_event_ids:
                    continue

                event = self.events.get(event_id)
                if not event:
                    trace.append(f"[FACT] Event {event_id} not found")
                    continue

                event_type = event.get("type")
                if target_event_types and event_type not in target_event_types:
                    trace.append(f"[FACT] Event {event_id} type {event_type} not in {target_event_types}")
                    continue

                # Check agent_required constraint
                if constraints.get("agent_required"):
                    # Must have multiple participants (agent + patient)
                    participants = event.get("participants", [])
                    if len(participants) < 2:
                        trace.append(
                            f"[FACT] Event {event_id} agent_required=True but only {len(participants)} participant(s)"
                        )
                        continue

                trace.append(f"[FACT] ✓ Event {event_id} matched ({event_type})")
                event_with_id = dict(event)
                event_with_id.setdefault("event_id", event_id)
                matched_events.append(event_with_id)
                matched_event_ids.add(event_id)

        trace.append(f"[FACT] Total matched: {len(matched_events)} events")
        return matched_events, trace

    def _execute_temporal(
        self,
        seed_entities: List[str],
        target_event_types: List[str],
        constraints: Dict[str, Any],
        max_depth: int,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """TEMPORAL intent: Find events related to seed entities, ordered by temporal constraint.

        Strategy: Find all events with seed entities, then apply temporal ordering
        (BEFORE/AFTER/DURING) if specified. Since no explicit temporal data in KG,
        use event iteration order as proxy (E0 < E1 < E2...).
        """
        trace: List[str] = []
        matched_events: List[Dict[str, Any]] = []
        temporal_order = constraints.get("temporal_order")

        trace.append(f"[TEMPORAL] Lookup with temporal_order={temporal_order}")

        # First, find all events with seed entities (like FACT)
        resolved_entities = self._resolve_entities(seed_entities, trace)
        if not resolved_entities:
            trace.append("[TEMPORAL] No seed entities resolved")
            return [], trace

        seed_events: List[Tuple[str, Dict[str, Any]]] = []
        for entity_id in resolved_entities:
            edges = self.edges_by_source.get(entity_id, [])
            for edge in edges:
                event_id = edge.get("event_id")
                event = self.events.get(event_id)
                if event:
                    event_type = event.get("type")
                    if event_type in target_event_types:
                        seed_events.append((event_id, event))

        trace.append(f"[TEMPORAL] Found {len(seed_events)} seed events")

        # Now apply temporal ordering
        # Since we don't have explicit timestamps, use event ID ordering (E0 < E1 < ...)
        if temporal_order == "AFTER":
            # Return events AFTER seed events (higher event IDs)
            seed_ids = {eid for eid, _ in seed_events}
            max_seed_id = max(seed_ids) if seed_ids else ""
            if max_seed_id:
                max_seed_num = int(max_seed_id[1:])
                for event_id in sorted(self.events.keys()):
                    if event_id.startswith("E"):
                        try:
                            event_num = int(event_id[1:])
                            if event_num > max_seed_num:
                                event = self.events.get(event_id)
                                if event and event.get("type") in target_event_types:
                                    if len(matched_events) < max_depth * 10:  # Limit results
                                        event_with_id = dict(event)
                                        event_with_id.setdefault("event_id", event_id)
                                        matched_events.append(event_with_id)
                                        trace.append(f"[TEMPORAL] ✓ Event {event_id} is AFTER")
                        except ValueError:
                            pass

        elif temporal_order == "BEFORE":
            # Return events BEFORE seed events (lower event IDs)
            seed_ids = {eid for eid, _ in seed_events}
            min_seed_id = min(seed_ids) if seed_ids else ""
            if min_seed_id:
                min_seed_num = int(min_seed_id[1:])
                for event_id in sorted(self.events.keys(), reverse=True):
                    if event_id.startswith("E"):
                        try:
                            event_num = int(event_id[1:])
                            if event_num < min_seed_num:
                                event = self.events.get(event_id)
                                if event and event.get("type") in target_event_types:
                                    if len(matched_events) < max_depth * 10:
                                        event_with_id = dict(event)
                                        event_with_id.setdefault("event_id", event_id)
                                        matched_events.append(event_with_id)
                                        trace.append(f"[TEMPORAL] ✓ Event {event_id} is BEFORE")
                        except ValueError:
                            pass

        else:
            # No temporal constraint, just return seed events
            matched_events = []
            for eid, event in seed_events:
                event_with_id = dict(event)
                event_with_id.setdefault("event_id", eid)
                matched_events.append(event_with_id)
            trace.append(f"[TEMPORAL] No temporal_order constraint, returning {len(matched_events)} seed events")

        trace.append(f"[TEMPORAL] Total matched: {len(matched_events)} events")
        return matched_events, trace

    def _execute_causal(
        self,
        seed_entities: List[str],
        target_event_types: List[str],
        constraints: Dict[str, Any],
        max_depth: int,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """CAUSAL intent: Find events + follow causal chains (depth-limited BFS).

        Strategy: Start from seed entity events, then traverse edges to find
        causally related events. Causal_chain constraint enforces multi-hop.
        """
        trace: List[str] = []
        matched_events: List[Dict[str, Any]] = []
        matched_event_ids: Set[str] = set()

        trace.append("[CAUSAL] Depth-limited causal traversal")

        resolved_entities = self._resolve_entities(seed_entities, trace)
        if not resolved_entities:
            trace.append("[CAUSAL] No seed entities resolved")
            return [], trace

        # BFS traversal
        queue: List[Tuple[str, int, List[str]]] = []  # (entity_id, depth, path)
        visited_entities: Set[str] = set()

        for entity_id in resolved_entities:
            queue.append((entity_id, 0, [entity_id]))
            visited_entities.add(entity_id)

        while queue:
            current_entity, depth, path = queue.pop(0)

            if depth > max_depth:
                continue

            # Find events involving current entity
            edges = self.edges_by_source.get(current_entity, [])
            for edge in edges:
                event_id = edge.get("event_id")
                if event_id in matched_event_ids:
                    continue

                event = self.events.get(event_id)
                if not event:
                    continue

                event_type = event.get("type")
                if event_type in target_event_types:
                    event_with_id = dict(event)
                    event_with_id.setdefault("event_id", event_id)
                    matched_events.append(event_with_id)
                    matched_event_ids.add(event_id)
                    trace.append(f"[CAUSAL] ✓ Depth {depth}: Event {event_id} ({event_type})")

                    # Explore further from this event's other participants
                    if depth < max_depth:
                        participants = event.get("participants", [])
                        for p in participants:
                            if p not in visited_entities:
                                visited_entities.add(p)
                                queue.append((p, depth + 1, path + [p]))
                                trace.append(f"[CAUSAL] → Enqueue entity {p} at depth {depth + 1}")

        trace.append(f"[CAUSAL] Total matched: {len(matched_events)} events")
        return matched_events, trace

    def _execute_multi_hop(
        self,
        seed_entities: List[str],
        target_event_types: List[str],
        constraints: Dict[str, Any],
        max_depth: int,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """MULTI_HOP intent: Find consequence/benefit chains (depth=2 minimum).

        Strategy: For each seed entity, find triggering events (KILL/DEATH/etc),
        then find consequence events (BOON/CURSE/etc) that involve other entities.
        This models: TRIGGER_EVENT → BENEFICIARY_EVENT.
        """
        trace: List[str] = []
        matched_events: List[Dict[str, Any]] = []
        matched_event_ids: Set[str] = set()

        trace.append("[MULTI_HOP] Consequence/benefit chain traversal (depth≥2)")

        resolved_entities = self._resolve_entities(seed_entities, trace)
        if not resolved_entities:
            trace.append("[MULTI_HOP] No seed entities resolved")
            return [], trace

        # Separate target types into triggers vs consequences
        trigger_types = {"KILL", "DEATH", "BATTLE"}
        consequence_types = {"BOON", "CURSE", "COMMAND", "SUPPORTED"}
        actual_targets = set(target_event_types)

        # Phase 1: Find trigger events (KILL/DEATH)
        trigger_events: Dict[str, Dict[str, Any]] = {}
        for entity_id in resolved_entities:
            edges = self.edges_by_source.get(entity_id, [])
            for edge in edges:
                event_id = edge.get("event_id")
                event = self.events.get(event_id)
                if not event:
                    continue

                event_type = event.get("type")
                if event_type in actual_targets and event_type in trigger_types:
                    event_with_id = dict(event)
                    event_with_id.setdefault("event_id", event_id)
                    trigger_events[event_id] = event_with_id
                    trace.append(f"[MULTI_HOP] Phase 1: ✓ Trigger event {event_id} ({event_type})")

        trace.append(f"[MULTI_HOP] Found {len(trigger_events)} trigger events")

        if not trigger_events and max_depth < 2:
            trace.append("[MULTI_HOP] No triggers found and depth < 2, returning empty")
            return [], trace

        # Phase 2: Find consequence events
        # Look for events involving OTHER entities that follow the trigger
        all_participants = set()
        for event in trigger_events.values():
            participants = event.get("participants", [])
            all_participants.update(participants)

        trace.append(f"[MULTI_HOP] Phase 2: Searching for consequences among {len(all_participants)} participants")

        consequence_events: Dict[str, Dict[str, Any]] = {}
        for entity_id in all_participants:
            edges = self.edges_by_source.get(entity_id, [])
            for edge in edges:
                event_id = edge.get("event_id")
                if event_id in trigger_events or event_id in consequence_events:
                    continue

                event = self.events.get(event_id)
                if not event:
                    continue

                event_type = event.get("type")
                if event_type in actual_targets:
                    event_with_id = dict(event)
                    event_with_id.setdefault("event_id", event_id)
                    consequence_events[event_id] = event_with_id
                    trace.append(f"[MULTI_HOP] Phase 2: ✓ Consequence event {event_id} ({event_type})")

        # Combine triggers + consequences
        matched_events = list(trigger_events.values()) + list(consequence_events.values())
        trace.append(
            f"[MULTI_HOP] Total matched: {len(trigger_events)} triggers + {len(consequence_events)} consequences"
        )

        return matched_events, trace

    def _resolve_entities(self, entity_names: List[str], trace: List[str]) -> List[str]:
        """Resolve seed entity names to entity IDs using the index.

        Returns list of valid entity IDs.
        """
        resolved: List[str] = []
        for name in entity_names:
            entity_id = self.entity_index.get(name.lower())
            if entity_id:
                resolved.append(entity_id)
                trace.append(f"[RESOLVE] {name} → {entity_id}")
            else:
                trace.append(f"[RESOLVE] {name} NOT FOUND in KG")
        return resolved

    def _extract_entities_from_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract unique entities mentioned in matched events.

        Returns list of entity dicts with their metadata.
        """
        entity_ids: Set[str] = set()
        for event in events:
            participants = event.get("participants", [])
            entity_ids.update(participants)

        result: List[Dict[str, Any]] = []
        for entity_id in sorted(entity_ids):
            entity = self.entities.get(entity_id)
            if entity:
                result.append(
                    {
                        "entity_id": entity_id,
                        "canonical_name": entity.get("canonical_name"),
                        "type": entity.get("type"),
                        "entity_type": entity.get("type"),
                        "event_count": entity.get("event_count", 0),
                    }
                )
        return result


def execute_query(
    query_plan: Any,  # Can be Dict or QueryPlan dataclass
    query_text: str = "",
    entities_path: str = "data/kg/entities.json",
    events_path: str = "data/kg/events.json",
    edges_path: str = "data/kg/edges.json",
) -> QueryResult:
    """Main entry point: Execute a QueryPlan against the KG.

    Args:
        query_plan: Dict or QueryPlan dataclass with intent, seed_entities,
                   target_event_types, constraints, traversal_depth
        query_text: Original question (optional)
        entities_path, events_path, edges_path: Paths to KG JSON files

    Returns:
        QueryResult with matches and debug trace
    """
    # Convert dataclass to dict if needed
    if hasattr(query_plan, '__dataclass_fields__'):
        from dataclasses import asdict
        query_plan = asdict(query_plan)

    entities, events, edges = KGLoader.load_graphs(entities_path, events_path, edges_path)
    executor = QueryExecutor(entities, events, edges)
    return executor.execute(query_plan, query_text)


# ========================== Inline Tests ==========================

def _mock_query_plans() -> List[Tuple[str, Dict[str, Any]]]:
    """Test query plans from Phase 5A QueryPlanner."""
    return [
        (
            "Who killed Karna?",
            {
                "intent": "FACT",
                "seed_entities": ["karna"],
                "target_event_types": ["KILL", "DEATH", "BATTLE", "CORONATION", "APPOINTED_AS"],
                "constraints": {"agent_required": True, "temporal_order": None, "causal_chain": False},
                "traversal_depth": 1,
            },
        ),
        (
            "Why did Bhishma support Duryodhana?",
            {
                "intent": "CAUSAL",
                "seed_entities": ["bhishma", "duryodhana"],
                "target_event_types": ["SUPPORTED", "DEFENDED", "VOW", "COMMAND"],
                "constraints": {"agent_required": False, "temporal_order": None, "causal_chain": True},
                "traversal_depth": 2,
            },
        ),
        (
            "What happened after Abhimanyu's death?",
            {
                "intent": "TEMPORAL",
                "seed_entities": ["abhimanyu"],
                "target_event_types": ["DEATH", "BATTLE", "RETREATED"],
                "constraints": {"agent_required": False, "temporal_order": "AFTER", "causal_chain": False},
                "traversal_depth": 2,
            },
        ),
        (
            "Who benefited from Drona's death?",
            {
                "intent": "MULTI_HOP",
                "seed_entities": ["drona"],
                "target_event_types": ["KILL", "DEATH", "BOON", "CURSE"],
                "constraints": {"agent_required": False, "temporal_order": None, "causal_chain": False},
                "traversal_depth": 2,
            },
        ),
    ]


def _run_inline_tests() -> None:
    """Execute inline tests against live KG."""
    import os

    print("Loading KG...")
    try:
        entities, events, edges = KGLoader.load_graphs()
        print(f"✓ Loaded: {len(entities)} entities, {len(events)} events, {len(edges)} edges")
    except RuntimeError as e:
        print(f"✗ Failed to load KG: {e}")
        return

    executor = QueryExecutor(entities, events, edges)

    print("\n" + "=" * 80)
    print("RUNNING PHASE 5B INLINE TESTS")
    print("=" * 80)

    for query_text, query_plan in _mock_query_plans():
        print(f"\nQuery: {query_text}")
        print(f"Intent: {query_plan['intent']}")
        print(f"Depth: {query_plan['traversal_depth']}")

        result = executor.execute(query_plan, query_text)

        print(f"\nResult:")
        print(f"  Found: {result.found}")
        print(f"  Events: {result.matched_events.__len__()}")
        print(f"  Entities: {result.matched_entities.__len__()}")
        print(f"  Time: {result.execution_time_ms:.2f}ms")

        if result.matched_events:
            print(f"\nTop 3 matched events:")
            for i, event in enumerate(result.matched_events[:3]):
                print(f"  {i + 1}. {event.get('type')} (tier={event.get('tier')})")
                print(f"     {event.get('sentence')[:80]}...")

        print(f"\nDebug trace (last 10 lines):")
        for line in result.debug_trace[-10:]:
            print(f"  {line}")

        print("-" * 80)


if __name__ == "__main__":
    _run_inline_tests()
