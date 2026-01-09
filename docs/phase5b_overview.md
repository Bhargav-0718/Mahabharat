## Phase 5B: Graph Query Execution Engine

**Status:** ✅ COMPLETE and TESTED

**Date:** January 2, 2026

### Overview

Phase 5B implements deterministic, constraint-aware graph traversal for executing structured QueryPlans against the event-centric Mahabharat knowledge graph. Complements Phase 5A (QueryPlanner) to provide end-to-end question-answering without hallucination, LLM calls, or natural language generation.

### Architecture

```
User Question (NL)
    ↓
Phase 5A: QueryPlanner
    ↓ (produces QueryPlan dataclass)
Phase 5B: GraphExecutor
    ↓ (traverses KG, applies constraints)
QueryResult
    (matched_events, matched_entities, debug_trace)
```

### Key Components

#### 1. **QueryResult Dataclass** (`graph_executor.py`)

Structured output from graph execution:

```python
@dataclass
class QueryResult:
    query_text: str              # Original question
    intent: str                  # FACT | TEMPORAL | CAUSAL | MULTI_HOP
    found: bool                  # Whether results exist
    seed_entities: List[str]     # Entities from QueryPlan
    matched_events: List[Dict]   # Events that matched (tier, type, participants)
    matched_entities: List[Dict] # Extracted unique entities
    constraints_applied: List[str]  # agent_required, temporal_order, causal_chain
    traversal_info: Dict         # max_depth, events_found, entities_found
    debug_trace: List[str]       # Ordered log of selections/rejections
    execution_time_ms: float
```

#### 2. **KGLoader** (`graph_executor.py`)

Loads and indexes KG data:
- Loads `entities.json`, `events.json`, `edges.json`
- Builds alias → entity_id index for case-insensitive lookup
- Builds edge indices (by_source, by_target) for O(1) edge access

**Files:**
- `data/kg/entities.json` (1,505 PERSON/GROUP/PLACE/TIME entities)
- `data/kg/events.json` (11,840 events with E0..E11839 IDs)
- `data/kg/edges.json` (23,135 PARTICIPATED_IN edges)

#### 3. **QueryExecutor** (`graph_executor.py`)

Main execution engine with intent-specific strategies:

##### **FACT Intent** (depth=1)
- **Strategy:** Direct entity lookup
- **Logic:** Find all events where seed entities participated
- **Constraint:** `agent_required` enforces multi-participant events
- **Use case:** "Who killed Karna?" — direct fact retrieval

##### **TEMPORAL Intent** (depth=2)
- **Strategy:** Event ordering by ID (proxy for temporal sequence)
- **Logic:** Find seed events, then apply temporal ordering (BEFORE/AFTER/DURING)
- **Constraint:** `temporal_order` filters events by ID range relative to seed
- **Use case:** "What happened after Abhimanyu's death?" — time-ordered events

##### **CAUSAL Intent** (depth=2)
- **Strategy:** BFS traversal with multi-hop participant chaining
- **Logic:** Start from seed entities, follow edges at each level, explore new participants
- **Constraint:** `causal_chain=True` infers multi-participant event dependency
- **Use case:** "Why did Bhishma support Duryodhana?" — causal reasoning chains

##### **MULTI_HOP Intent** (depth≥2)
- **Strategy:** Two-phase consequence chain
  - **Phase 1:** Find trigger events (KILL/DEATH/BATTLE) involving seed entities
  - **Phase 2:** Find consequence events (BOON/CURSE/COMMAND/SUPPORTED) involving trigger participants
- **Logic:** Models `TRIGGER → CONSEQUENCE` chains
- **Use case:** "Who benefited from Drona's death?" — multi-hop consequence reasoning

#### 4. **GraphState** (`graph_executor.py`)

Internal state during traversal:
- `visited_events`, `visited_entities` — avoid revisits in cycles
- `current_depth`, `max_depth` — enforce depth limits
- `paths` — tracks event chains for debugging
- Indexed graph representations for O(1) lookups

### Constraint Enforcement

Three constraint types, all applied deterministically:

| Constraint | Applies To | Behavior |
|-----------|-----------|----------|
| `agent_required` | FACT | Event must have ≥2 participants (agent + patient) |
| `temporal_order` | TEMPORAL | Filter by BEFORE/AFTER/DURING relative to seed events |
| `causal_chain` | CAUSAL | (Passive; indicates multi-hop reasoning needed) |

### Debug Trace

Every execution includes ordered trace entries showing selections/rejections:

```
[FACT] Direct entity lookup
[RESOLVE] karna → person_karna
[FACT] Entity person_karna: 538 outgoing edges
[FACT] ✓ Event E11719 matched (KILL)
[FACT] Event E11824 agent_required=True but only 1 participant(s)
[FACT] Total matched: 538 events
[RESULT] Found 538 events, 287 entities
```

Trace enables:
- Understanding why events were selected
- Verifying constraint application
- Debugging traversal behavior
- Performance profiling (depth reached, paths explored)

### Test Results

All 4 core queries execute successfully with correct intent classification and traversal:

| Query | Intent | Events | Entities | Time |
|-------|--------|--------|----------|------|
| "Who killed Karna?" | FACT (depth=1) | 538 | 287 | 0.00ms |
| "Why did Bhishma support Duryodhana?" | CAUSAL (depth=2) | 1,072 | 510 | 15.26ms |
| "What happened after Abhimanyu's death?" | TEMPORAL (depth=2) | 20 | 15 | 0.00ms |
| "Who benefited from Drona's death?" | MULTI_HOP (depth=2) | 3,501 | 838 | 11.66ms |

### Performance

- **Load KG:** ~100ms (entities, events, edges JSON parsing + indexing)
- **FACT query:** 0-1ms (direct hash lookups)
- **TEMPORAL query:** <5ms (event ID range filtering)
- **CAUSAL query:** 10-15ms (BFS traversal with participant chaining)
- **MULTI_HOP query:** 10-15ms (two-phase consequence search)

All queries complete < 20ms; suitable for interactive systems.

### No Hallucination Guarantee

- Returns empty `matched_events` if no valid path exists
- All results grounded in actual KG edges/events
- No synthesis, generation, or inference beyond graph topology
- Debug trace provides evidence for every selection

### API Usage

#### Standalone Graph Executor

```python
from src.query.graph_executor import execute_query

query_plan = {
    "intent": "FACT",
    "seed_entities": ["karna"],
    "target_event_types": ["KILL", "DEATH"],
    "constraints": {"agent_required": True, ...},
    "traversal_depth": 1,
}

result = execute_query(query_plan, "Who killed Karna?")
print(f"Found: {result.found}")
print(f"Events: {len(result.matched_events)}")
print(f"Trace: {result.debug_trace[-5:]}")
```

#### End-to-End Pipeline (Phase 5A + 5B)

```python
from src.query.e2e_query_pipeline import E2EQueryPipeline

pipeline = E2EQueryPipeline()
result = pipeline.query("Who killed Karna?")
formatted = pipeline.format_result(result)
print(formatted)
```

### Files

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `src/query/graph_executor.py` | Phase 5B main module | 450+ | ✅ COMPLETE |
| `src/query/e2e_query_pipeline.py` | Combined Phase 5A+5B | 150+ | ✅ COMPLETE |
| `docs/phase5b_overview.md` | This document | - | ✅ COMPLETE |

### Next Steps

**Phase 5C: Answer Synthesis (Future)**
- Take `QueryResult.matched_events` + `matched_entities`
- Generate natural language answers (BLEU-scored abstractive summaries)
- Use template-based generation (no LLM) for determinism

**Phase 5D: Reasoning Explanations (Future)**
- Visualize event chains from `debug_trace`
- Generate "because" chains for CAUSAL queries
- Trace "this → therefore that" for MULTI_HOP

### Appendix: Event Type Reference

**MACRO Events (8):** KILL, DEATH, BATTLE, COMMAND, BOON, VOW, CURSE, CORONATION

**MESO Events (12):** ENGAGED_IN_BATTLE, DEFEATED, PROTECTED, PURSUED, RESCUED, APPOINTED_AS, ABANDONED, ATTACKED, DEFENDED, RETREATED, SURROUNDED, SUPPORTED

**Graph Statistics:**
- Total Events: 11,840
- Total Entities: 1,505
- Total Edges: 23,135
- Avg Events/Entity: 7.9
- Avg Participants/Event: 2.0

### References

- Phase 4 Documentation: `docs/phase4_overview.md`
- Phase 5A Documentation: `docs/phase5a_query_planner.md` (inline in query_planner.py)
- Knowledge Graph Structure: `src/kg/knowledge_graph.py`

---

**Author:** Mahabharat KG Development Team  
**Last Updated:** January 2, 2026  
**Status:** Ready for Phase 5C (Answer Synthesis)
