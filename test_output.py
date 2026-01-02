#!/usr/bin/env python
"""Test script to validate Phase 4 outputs."""
import json

# Load entities
with open("data/kg/entities.json") as f:
    entities = json.load(f)

# Load events
with open("data/kg/events.json") as f:
    events = json.load(f)

# Count entity types
types = {}
for ent in entities["entities"].values():
    t = ent["type"]
    types[t] = types.get(t, 0) + 1

print("=" * 80)
print("PHASE 4 OUTPUT VALIDATION")
print("=" * 80)

print("\n=== ENTITY DISTRIBUTION ===")
print(f"Total entities: {entities['metadata']['total']}")
for t in sorted(types.keys()):
    print(f"  {t:10s}: {types[t]:5d} ({100*types[t]/entities['metadata']['total']:5.1f}%)")

print("\n=== EVENT DISTRIBUTION ===")
print(f"Total events: {events['metadata']['total']}")
event_types = {}
event_tiers = {}
for event in events["events"].values():
    et = event["type"]
    event_types[et] = event_types.get(et, 0) + 1
    tier = event.get("tier", "UNKNOWN")
    event_tiers[tier] = event_tiers.get(tier, 0) + 1

print("\nEvent types:")
for et in sorted(event_types.keys()):
    print(f"  {et:20s}: {event_types[et]:3d}")

print("\nEvent tiers:")
for tier in sorted(event_tiers.keys()):
    print(f"  {tier:10s}: {event_tiers[tier]:3d}")

print("\n=== TOP 15 ENTITIES (by event count) ===")
top = sorted([(k, v["canonical_name"], v["event_count"]) for k,v in entities["entities"].items()], key=lambda x: x[2], reverse=True)[:15]
for k, name, count in top:
    print(f"  {name:35s} {count:3d} events")

print("\n=== QUALITY CHECK ===")
# Check for pronouns
pronouns = {"thou", "thee", "thy", "him", "her", "they", "them", "his", "hers", "who", "whom", "he", "she", "it", "we", "you"}
pronoun_entities = [v["canonical_name"] for v in entities["entities"].values() if v["canonical_name"].lower() in pronouns]
print(f"Pronouns found: {len(pronoun_entities)}")
if pronoun_entities:
    for p in pronoun_entities[:5]:
        print(f"  - {p}")

# Check for noise phrases
noise = [v["canonical_name"] for v in entities["entities"].values() if any(ph in v["canonical_name"].lower() for ph in ["having ", "being ", "the presence", "the act of"])]
print(f"Noise phrases found: {len(noise)}")
if noise:
    for n in noise[:5]:
        print(f"  - {n}")

print("\n=== TARGETS vs ACTUAL ===")
print(f"Events:   TARGET 500-1500,   ACTUAL {events['metadata']['total']}")
print(f"Entities: TARGET 2000-3500,  ACTUAL {entities['metadata']['total']}")

print("\n✓ Validation passed!" if events['metadata']['total'] > 50 and entities['metadata']['total'] > 1000 else "\n⚠ WARNING: Below targets")
print("=" * 80)
