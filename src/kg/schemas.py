"""Schemas for Phase 4 KG pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EntityMention:
    text: str
    type: str
    chunk_id: str
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass
class ResolvedEntity:
    mention: str
    entity_id: str
    type: str
    chunk_id: str
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass
class RelationRecord:
    subject: str
    relation: str
    object: str
    evidence_chunk: str
    confidence: float = 1.0


@dataclass
class KGStats:
    entities: int
    relations: int
    edges: int
    nodes: int
    warnings: List[str] = field(default_factory=list)
