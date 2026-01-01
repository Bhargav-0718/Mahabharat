"""Deterministic relation extraction using pattern heuristics."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .schemas import RelationRecord, ResolvedEntity


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


class RelationExtractor:
    def __init__(self, patterns_path: Path) -> None:
        self.patterns = self._load_patterns(patterns_path)

    def _load_patterns(self, path: Path) -> Dict[str, object]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _mentions_in_sentence(self, sentence: str, mentions: Sequence[ResolvedEntity]) -> List[Tuple[ResolvedEntity, int, int]]:
        lower = sentence.lower()
        hits: List[Tuple[ResolvedEntity, int, int]] = []
        for m in mentions:
            for match in re.finditer(re.escape(m.mention.lower()), lower):
                hits.append((m, match.start(), match.end()))
        return hits

    def _ordered_relations(
        self,
        sentence: str,
        sent_mentions: List[Tuple[ResolvedEntity, int, int]],
        chunk_id: str,
    ) -> List[RelationRecord]:
        records: List[RelationRecord] = []
        lower = sentence.lower()
        for entry in self.patterns.get("ordered_verbs", []):
            relation = entry.get("relation")
            verbs = entry.get("verbs", [])
            subj_first = bool(entry.get("subject_before_object", True))
            for verb in verbs:
                if verb not in lower:
                    continue
                verb_pos = lower.find(verb)
                for subj, s_start, _ in sent_mentions:
                    for obj, o_start, _ in sent_mentions:
                        if subj.entity_id == obj.entity_id:
                            continue
                        if subj_first and not (s_start <= verb_pos <= o_start):
                            continue
                        if not subj_first and not (o_start <= verb_pos <= s_start):
                            continue
                        records.append(
                            RelationRecord(
                                subject=subj.entity_id,
                                relation=relation,
                                object=obj.entity_id,
                                evidence_chunk=subj.chunk_id,
                            )
                        )
        return records

    def _of_construction(
        self,
        sentence: str,
        sent_mentions: List[Tuple[ResolvedEntity, int, int]],
        chunk_id: str,
    ) -> List[RelationRecord]:
        records: List[RelationRecord] = []
        lower = sentence.lower()
        for entry in self.patterns.get("of_construction", []):
            cue = entry.get("cue", "")
            relation = entry.get("relation")
            direction = entry.get("direction", "parent_after")
            if cue not in lower:
                continue
            cue_pos = lower.find(cue)
            after = [m for m in sent_mentions if m[1] > cue_pos]
            before = [m for m in sent_mentions if m[1] < cue_pos]
            if relation == "PARENT_OF" and direction == "parent_after":
                if not after:
                    continue
                parent = after[0][0]
                children = [m[0] for m in after[1:]] + [m[0] for m in before]
                for child in children:
                    if child.entity_id == parent.entity_id:
                        continue
                    records.append(
                        RelationRecord(
                            subject=parent.entity_id,
                            relation="PARENT_OF",
                            object=child.entity_id,
                            evidence_chunk=chunk_id,
                        )
                    )
            if relation == "SIBLING_OF" and direction == "sibling_after":
                if not after:
                    continue
                anchor = after[0][0]
                siblings = [m[0] for m in after[1:]] + [m[0] for m in before]
                for sib in siblings:
                    if sib.entity_id == anchor.entity_id:
                        continue
                    records.append(
                        RelationRecord(
                            subject=anchor.entity_id,
                            relation="SIBLING_OF",
                            object=sib.entity_id,
                            evidence_chunk=chunk_id,
                        )
                    )
        return records

    def _commander_of(
        self,
        sentence: str,
        sent_mentions: List[Tuple[ResolvedEntity, int, int]],
        chunk_id: str,
    ) -> List[RelationRecord]:
        records: List[RelationRecord] = []
        lower = sentence.lower()
        for entry in self.patterns.get("commander_of", []):
            cue = entry.get("cue", "")
            if cue not in lower:
                continue
            cue_pos = lower.find(cue)
            commanders = [m for m in sent_mentions if m[1] < cue_pos]
            groups = [m for m in sent_mentions if m[1] > cue_pos]
            for cmd in commanders:
                for grp in groups:
                    records.append(
                        RelationRecord(
                            subject=grp[0].entity_id,
                            relation="COMMANDED_BY",
                            object=cmd[0].entity_id,
                            evidence_chunk=chunk_id,
                        )
                    )
        return records

    def extract_relations(self, text: str, chunk_id: str, resolved_mentions: Sequence[ResolvedEntity]) -> List[RelationRecord]:
        if not text or not resolved_mentions:
            return []
        relations: List[RelationRecord] = []
        sentences = split_sentences(text)
        for sent in sentences:
            sent_mentions = self._mentions_in_sentence(sent, resolved_mentions)
            if not sent_mentions:
                continue
            relations.extend(self._ordered_relations(sent, sent_mentions, chunk_id))
            relations.extend(self._of_construction(sent, sent_mentions, chunk_id))
            relations.extend(self._commander_of(sent, sent_mentions, chunk_id))
        return relations
