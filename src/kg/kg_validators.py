"""Validators for Phase 4 KG outputs."""
from __future__ import annotations

import logging
from typing import Dict, List

import networkx as nx

logger = logging.getLogger(__name__)


def validate_no_self_loops(graph: nx.MultiDiGraph) -> List[str]:
    warnings: List[str] = []
    for u, v, data in graph.edges(data=True):
        if u == v:
            warnings.append(f"Self-loop detected on {u} ({data.get('relation')})")
    return warnings


def validate_entities_exist(graph: nx.MultiDiGraph) -> List[str]:
    warnings: List[str] = []
    for u, v, data in graph.edges(data=True):
        if u not in graph:
            warnings.append(f"Missing subject node {u} for edge {data}")
        if v not in graph:
            warnings.append(f"Missing object node {v} for edge {data}")
    return warnings


def validate_symmetry(graph: nx.MultiDiGraph) -> List[str]:
    warnings: List[str] = []
    for u, v, data in graph.edges(data=True):
        if data.get("relation") == "SIBLING_OF":
            back_edges = graph.get_edge_data(v, u, default={}) or {}
            has_back = any(ed.get("relation") == "SIBLING_OF" for ed in back_edges.values())
            if not has_back:
                warnings.append(f"Missing symmetric sibling edge for {u} <-> {v}")
    return warnings


def validate_required_fields(graph: nx.MultiDiGraph) -> List[str]:
    warnings: List[str] = []
    for u, v, data in graph.edges(data=True):
        if not data.get("relation"):
            warnings.append(f"Edge {u}->{v} missing relation")
        if not data.get("evidence"):
            warnings.append(f"Edge {u}->{v} missing evidence")
    return warnings


def run_validations(graph: nx.MultiDiGraph) -> Dict[str, List[str]]:
    checks = {
        "no_self_loops": validate_no_self_loops(graph),
        "entities_exist": validate_entities_exist(graph),
        "symmetry": validate_symmetry(graph),
        "required_fields": validate_required_fields(graph),
    }
    for name, warns in checks.items():
        for w in warns:
            logger.warning("[%s] %s", name, w)
    return checks
