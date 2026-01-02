# Phase 4 Overview — Event-Centric Knowledge Graph

This document summarizes Phase 4 (event-centric KG), its pipeline, fixes, how to run it, and validation status.

## Goals
- Build an event-first KG: events detected, arguments extracted, entities admitted only from events.
- Enforce clean admission: no pronouns/noise, deduped arguments, unique event IDs.
- Recover places from event context and downgrade conceptual nouns.
- Preserve high recall (11.8k events) while keeping entities clean and supported.

## Inputs and Outputs
- Inputs:
  - data/parsed_text/parsed_pages.jsonl (semantic chunks from Phase 3)
- Primary outputs:
  - data/kg/entities.json (1,505 entities after postprocess)
  - data/kg/events.json (11,840 events)
  - data/kg/edges.json (23,135 edges)
  - data/kg/graph_stats.json
  - data/kg/validation_report.json

## Pipeline Stages
Implementation: [src/kg/phase4_pipeline.py](src/kg/phase4_pipeline.py)

1) Event detection (rule-based)
- [src/kg/event_detector.py](src/kg/event_detector.py)
- 20 event types (8 MACRO, 12 MESO); sentence cleaning removes URLs/paths; rejects micro verbs.

2) Argument extraction
- [src/kg/event_extractor.py](src/kg/event_extractor.py)
- Validates MESO events (multi-actor + tactical verbs); deduplicates arguments by (text, role).
- Hard entity filters: pronouns blocked, max 4 tokens, preposition/verb guards, noise phrases rejected.

3) Entity admission & aliasing
- [src/kg/entity_registry.py](src/kg/entity_registry.py)
- Entities created only from event arguments; alias normalization; admission gate enforces filters.

4) Graph build
- [src/kg/knowledge_graph.py](src/kg/knowledge_graph.py)
- Monotonic event IDs E1..En (no collisions); event nodes admitted even if no edges; edges stored with evidence.

5) Post-processing (Fixes D/E/F)
- [src/kg/phase4_postprocess.py](src/kg/phase4_postprocess.py)
- Fix D: Downgrade conceptual nouns to LITERAL when only abstract-object roles.
- Fix E: Recover places via regex (at/in/field of/near [Place]); whitelist + deconflict with PERSON/GROUP; add OCCURRED_AT edges.
- Fix F: Minimum support threshold (PERSON≥2 events; GROUP/PLACE/TIME≥1; LITERAL unlimited); removes weak entities and their edges.

6) Validation
- [src/kg/kg_validators.py](src/kg/kg_validators.py)
- Checks graph consistency (IDs, edges, alias collisions, counts). Current run: PASSED.

7) Save outputs
- entities.json, events.json, edges.json, graph_stats.json, validation_report.json, entity_registry.json.

## How to Run
```
D:/AI/Mahabharat/.venv/Scripts/python.exe -m src.kg.phase4_pipeline data/parsed_text data/kg
```
Outputs are written to data/kg.

## Current State (2026-01-02)
- Events: 11,840
- Entities: 1,505 (after Fix F)
- Edges: 23,135
- Places recovered (Fix E): Kurukshetra, Indraprastha, Khandavaprastha, India (others filtered)
- Validation: PASSED

## Key Fixes Implemented
- Fix A/B/C (earlier): Event tiers (MACRO/MESO), hard entity filters, event-first admission (≥1 argument).
- Six critical fixes: monotonic event IDs; decouple event admission from edges; sentence decontamination; rejection logging; argument deduplication; tighter entity guard.
- Fix D/E/F (postprocess): Concept downgrading to LITERAL; regex place recovery with whitelist; minimum support thresholds.

## Follow-ups
- Expand vetted place whitelist as more canonical locations are identified.
- Add lightweight stats export per event type/tier for QA.
- Optional: surface LITERAL nodes differently in downstream retrieval/QA.
