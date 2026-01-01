"""Knowledge graph builder using NetworkX."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import networkx as nx

from .schemas import RelationRecord


class KnowledgeGraph:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add_entity(self, entity_id: str, etype: str, aliases: Iterable[str]) -> None:
        if entity_id not in self.graph:
            self.graph.add_node(entity_id, type=etype, aliases=list(aliases))

    def add_relation(self, record: RelationRecord) -> None:
        if not record.subject or not record.object:
            return
        if record.subject not in self.graph or record.object not in self.graph:
            return
        # Merge evidence if identical edge already exists
        for key, data in self.graph.get_edge_data(record.subject, record.object, default={}).items():
            if data.get("relation") == record.relation:
                evidence = set(data.get("evidence", []))
                evidence.add(record.evidence_chunk)
                data["evidence"] = sorted(evidence)
                return
        self.graph.add_edge(
            record.subject,
            record.object,
            relation=record.relation,
            evidence=[record.evidence_chunk],
            confidence=record.confidence,
        )

    def to_entities(self) -> List[Dict[str, object]]:
        result: List[Dict[str, object]] = []
        for node, data in self.graph.nodes(data=True):
            result.append({"id": node, "type": data.get("type"), "aliases": data.get("aliases", [])})
        return result

    def to_relations(self) -> List[Dict[str, object]]:
        rels: List[Dict[str, object]] = []
        for u, v, data in self.graph.edges(data=True):
            rels.append(
                {
                    "subject": u,
                    "relation": data.get("relation"),
                    "object": v,
                    "evidence": data.get("evidence", []),
                    "confidence": data.get("confidence", 1.0),
                }
            )
        return rels

    def to_json(self) -> Dict[str, object]:
        return {
            "entities": self.to_entities(),
            "relations": self.to_relations(),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
