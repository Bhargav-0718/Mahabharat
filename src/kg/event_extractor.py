"""Extract event arguments (subject, object, etc.) from detected event sentences.

Uses shallow parsing + regex + dependency hinting to extract participants.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import spacy
from .event_detector import DetectedEvent

logger = logging.getLogger(__name__)


# MESO event types (tactical/relational)
MESO_EVENT_TYPES = {
    "ENGAGED_IN_BATTLE",
    "DEFEATED",
    "PROTECTED",
    "PURSUED",
    "RESCUED",
    "APPOINTED_AS",
    "ABANDONED",
    "ATTACKED",
    "DEFENDED",
    "RETREATED",
    "SURROUNDED",
    "SUPPORTED",
    "FORMED_ARRAY_AGAINST",
}

# Tactical verb cues for confidence scoring
TACTICAL_VERBS = [
    "attack", "assail", "assault", "strike", "charged", "fell upon", "made war",
    "defend", "shield", "protect", "hold the line", "stood fast",
    "pursue", "chase", "hunt", "follow",
    "retreat", "withdrew", "fled", "fell back", "turned back",
    "surround", "hemmed in", "closed in", "arrayed", "formed array", "entered the ranks",
    "support", "reinforce", "succour", "cover the retreat",
]


# Hard entity admission filters (FIX B)
PRONOUN_BLOCKLIST = {
    "thou", "thee", "thy", "him", "her", "they", "them",
    "his", "hers", "their", "who", "whom", "whose",
    "he", "she", "it", "we", "you", "i", "me", "my",
}

STOP_PHRASES = {
    "having", "being", "the presence", "the act of",
    "in order to", "which is", "that is", "as well as",
}


def is_valid_entity_candidate(text: str, nlp=None) -> bool:
    """Check if text is a valid entity candidate (FIX B + FIX 6).
    
    Mandatory rejection rules:
    1. Pronoun in blocklist
    2. Text length > 6 tokens (FIX 6: reduced from previous)
    3. Starts with verb (if nlp available)
    4. Contains stop phrases
    5. Only lowercase common words
    6. Punctuation-only or digit-only
    7. URL/filename patterns
    8. Contains generic prepositions/articles as core (FIX 6)
    
    Args:
        text: Candidate entity text
        nlp: Optional spaCy model for POS tagging
        
    Returns:
        True if valid, False if should be rejected
    """
    if not text or len(text) < 2:
        return False
        
    # Normalize for checks
    text_lower = text.lower().strip()
    tokens = text_lower.split()
    
    # Rule 1: Reject pronouns
    if text_lower in PRONOUN_BLOCKLIST:
        return False
    
    # FIX 6: Reject if > 4 tokens (tighter)
    if len(tokens) > 4:
        return False
        
    # Rule 3: Reject if starts with verb (requires nlp)
    if nlp:
        doc = nlp(text)
        if doc and doc[0].pos_ in ("VERB", "AUX"):
            return False
            
    # Rule 4: Reject if contains stop phrases
    for phrase in STOP_PHRASES:
        if phrase in text_lower:
            return False
            
    # Rule 5: Reject if only lowercase common words (no capitalization)
    if text == text_lower and len(tokens) > 1:
        # Check if all tokens are common English words
        common_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "from", "for", "of", "with", "by", "as", "is", "was",
            "are", "were", "been", "be", "have", "has", "had", "do",
            "does", "did", "will", "would", "could", "should", "may",
            "might", "can", "must", "shall",
        }
        if all(token in common_words for token in tokens):
            return False
            
    # Rule 6: Reject punctuation-only or digit-only
    if text_lower.replace(" ", "").isdigit():
        return False
    if not any(c.isalnum() for c in text):
        return False
        
    # Rule 7: Reject URL/filename patterns
    if any(pattern in text_lower for pattern in [".htm", "http", "www", ".com"]):
        return False
    
    # FIX 6: Reject if dominated by prepositions
    prepositions = {"the", "of", "in", "on", "at", "to", "from", "for", "by", "with"}
    if tokens and all(t in prepositions for t in tokens):
        return False
        
    return True


@dataclass
class EventArgument:
    """Represents an extracted event argument."""

    role: str  # subject, object, agent, patient, recipient
    text: str
    start: int
    end: int


@dataclass
class ExtractedEvent:
    """Event with extracted arguments."""

    event_type: str
    sentence: str
    sentence_index: int
    chunk_id: str
    parva: str
    section: str
    arguments: List[EventArgument]  # subject, object, etc.
    tier: str = "MESO"  # "MACRO" or "MESO"


class EventExtractor:
    """Extracts arguments from event sentences using shallow parsing + regex."""

    # Role extraction patterns by event type
    # Format: "role": (pre_pattern, entity_pattern, post_pattern)
    ROLE_PATTERNS = {
        "KILL": {
            "agent": (r"\b(?:who|X)\b", r"[a-z]+(?:\s+of\s+\w+)*", r"(?:\s+kill|\s+slew)"),
            "patient": (r"(?:killed|slew|slain)\s+", r"[a-z]+(?:\s+(?:of|the)\s+\w+)*", r"(?:\s|,|\.|$)"),
        },
        "COMMAND": {
            "agent": (r"(?:, )?", r"[a-z]+(?:\s+\w+)*", r"(?:\s+commanded|\s+ordered)"),
            "patient": (r"(?:commanded|ordered)\s+", r"[a-z]+(?:\s+\w+)*", r"(?:\s+(?:to|not)\s|\s|,|\.|$)"),
        },
        "BATTLE": {
            "agent1": (r"(?:between|with)\s+", r"[a-z]+(?:\s+and\s+|,\s+)*", r"(?:\s+and|\s+,)"),
            "agent2": (r"(?:\s+and|\s+,)\s+", r"[a-z]+(?:\s+\w+)*", r"(?:\s|,|\.|$)"),
        },
        "VOW": {
            "agent": (r"(?:, )?", r"[a-z]+(?:\s+\w+)*", r"(?:\s+vowed|\s+swore)"),
        },
        "BIRTH": {
            "agent": (r"(?:, )?", r"[a-z]+(?:\s+\w+)*", r"(?:\s+was\s+born|was\s+begotten)"),
        },
        "CURSE": {
            "agent": (r"(?:, )?", r"[a-z]+(?:\s+\w+)*", r"(?:\s+cursed)"),
            "patient": (r"(?:cursed)\s+", r"[a-z]+(?:\s+\w+)*", r"(?:\s|,|\.|$)"),
        },
        "BOON": {
            "agent": (r"(?:, )?", r"[a-z]+(?:\s+\w+)*", r"(?:\s+granted)"),
            "recipient": (r"(?:granted)\s+(?:to\s+)?", r"[a-z]+(?:\s+\w+)*", r"(?:\s|,|\.|$)"),
        },
        "EXILE": {
            "agent": (r"(?:, )?", r"[a-z]+(?:\s+\w+)*", r"(?:\s+was\s+exiled|\s+was\s+banished)"),
        },
        "DEATH": {
            "agent": (r"(?:, )?", r"[a-z]+(?:\s+\w+)*", r"(?:\s+died|\s+perished)"),
        },
        # MESO events rely primarily on spaCy extraction; regex kept minimal
        "ATTACKED": {},
        "DEFENDED": {},
        "RETREATED": {},
        "SURROUNDED": {},
        "SUPPORTED": {},
        "FORMED_ARRAY_AGAINST": {},
        "ENGAGED_IN_BATTLE": {},
        "DEFEATED": {},
        "PROTECTED": {},
        "PURSUED": {},
        "RESCUED": {},
        "APPOINTED_AS": {},
        "ABANDONED": {},
    }

    def __init__(self, debug: bool = False):
        """Initialize extractor with spaCy model."""
        self.debug = debug or bool(int(os.getenv("KG_DEBUG_EVENTS", "0")))
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy model not found. Install with: python -m spacy download en_core_web_sm"
            )
            self.nlp = None

    def extract(self, detected_event: DetectedEvent) -> ExtractedEvent:
        """Extract arguments from a detected event.
        
        Args:
            detected_event: Event detected by EventDetector
            
        Returns:
            ExtractedEvent with extracted arguments
        """
        arguments = []

        # Use pattern-based extraction
        patterns = self.ROLE_PATTERNS.get(detected_event.event_type, {})
        sentence = detected_event.sentence

        for role, (pre_pat, entity_pat, post_pat) in patterns.items():
            matches = self._extract_with_pattern(
                sentence, pre_pat, entity_pat, post_pat
            )
            for match_text, start, end in matches:
                clean_text = self._clean_entity_text(match_text)
                # FIX B: Apply hard entity filters
                if clean_text and is_valid_entity_candidate(clean_text, self.nlp):
                    arguments.append(
                        EventArgument(
                            role=role,
                            text=clean_text,
                            start=start,
                            end=end,
                        )
                    )

        # Also try spaCy if available for additional extraction
        if self.nlp and len(arguments) < 3:
            spacy_args = self._extract_with_spacy(detected_event.event_type, sentence)
            arguments.extend(spacy_args)

        # FIX 5: Deduplicate arguments by (text, role)
        seen = set()
        deduped = []
        for arg in arguments:
            key = (arg.text.lower(), arg.role)
            if key not in seen:
                seen.add(key)
                deduped.append(arg)
        arguments = deduped

        # MESO validation: multi-actor + confidence scoring
        if detected_event.event_type in MESO_EVENT_TYPES:
            passed, reason = self._assess_meso_event(detected_event, arguments)
            if not passed:
                self._log_meso_rejection(detected_event, reason)
                arguments = []  # force rejection
            else:
                self._log_meso_accept(detected_event, arguments)

        return ExtractedEvent(
            event_type=detected_event.event_type,
            sentence=sentence,
            sentence_index=detected_event.sentence_index,
            chunk_id=detected_event.chunk_id,
            parva=detected_event.parva,
            section=detected_event.section,
            arguments=arguments,
            tier=detected_event.tier,  # Propagate tier from detection
        )

    def _assess_meso_event(self, detected_event: DetectedEvent, arguments: List[EventArgument]) -> Tuple[bool, str]:
        """Apply multi-actor and confidence rules for MESO events.

        Rules:
        - Require ≥2 PERSON/GROUP or (1 PERSON/GROUP + 1 PLACE)
        - Confidence: +1 multi-actor, +1 tactical verb, +1 place present, -1 short sentence (<8 tokens)
        - Keep only if confidence >= 2
        """
        sentence = detected_event.sentence
        sentence_lower = sentence.lower()
        token_len = len(sentence.split())

        actors, places = self._extract_actor_place_signals(sentence, arguments)

        # Multi-actor requirement
        multi_actor = len(actors) >= 2 or (len(actors) >= 1 and len(places) >= 1)

        # Confidence scoring
        confidence = 0
        if multi_actor:
            confidence += 1
        if self._has_tactical_verb(sentence_lower):
            confidence += 1
        if len(places) >= 1:
            confidence += 1
        if token_len < 8:
            confidence -= 1

        if not multi_actor:
            return False, "no_actors"
        if confidence < 2:
            return False, "low_confidence"
        return True, "accepted"

    def _extract_actor_place_signals(self, sentence: str, arguments: List[EventArgument]) -> Tuple[set, set]:
        """Collect actor/place cues using spaCy when available; fall back to arguments.

        Actors: PERSON/ORG labels
        Places: GPE/LOC/FAC labels
        """
        actors = set()
        places = set()

        if self.nlp:
            doc = self.nlp(sentence)
            for ent in doc.ents:
                if ent.label_ in ("PERSON", "ORG"):
                    actors.add(ent.text.lower())
                elif ent.label_ in ("GPE", "LOC", "FAC"):
                    places.add(ent.text.lower())

        # Fallback to arguments if spaCy misses
        if not actors:
            for arg in arguments:
                actors.add(arg.text.lower())
        return actors, places

    def _has_tactical_verb(self, sentence_lower: str) -> bool:
        """Check if sentence contains a tactical verb cue."""
        for verb in TACTICAL_VERBS:
            if re.search(r"\b" + re.escape(verb) + r"\b", sentence_lower):
                return True
        return False

    def _log_meso_rejection(self, event: DetectedEvent, reason: str) -> None:
        if not self.debug:
            return
        truncated = (event.sentence[:140] + "...") if len(event.sentence) > 140 else event.sentence
        logger.debug(
            {
                "action": "meso_reject",
                "reason": reason,
                "chunk_id": event.chunk_id,
                "event_type": event.event_type,
                "sentence": truncated,
            }
        )

    def _log_meso_accept(self, event: DetectedEvent, arguments: List[EventArgument]) -> None:
        if not self.debug:
            return
        actors = [arg.text for arg in arguments]
        logger.debug(
            {
                "action": "meso_accept",
                "event_type": event.event_type,
                "chunk_id": event.chunk_id,
                "actors": actors,
            }
        )

    def _extract_with_pattern(
        self, text: str, pre_pat: str, entity_pat: str, post_pat: str
    ) -> List[Tuple[str, int, int]]:
        """Extract entity matches using regex patterns.
        
        Returns:
            List of (match_text, start, end)
        """
        full_pattern = f"({pre_pat})({entity_pat})({post_pat})"
        matches = []
        try:
            for match in re.finditer(full_pattern, text, re.IGNORECASE):
                # Group 2 is the entity
                entity_text = match.group(2)
                start = match.start(2)
                end = match.end(2)
                matches.append((entity_text, start, end))
        except re.error as e:
            logger.debug(f"Pattern error: {e}")

        return matches

    def _extract_with_spacy(self, event_type: str, sentence: str) -> List[EventArgument]:
        """Extract entities using spaCy NER.
        
        Returns:
            List of EventArgument
        """
        if not self.nlp:
            return []

        doc = self.nlp(sentence)
        arguments = []

        # Prefer PERSON entities for most events
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE"):
                # FIX B: Apply hard entity filters
                if not is_valid_entity_candidate(ent.text, self.nlp):
                    continue
                    
                role = "agent" if "PERSON" in ent.label_ else "group"
                arguments.append(
                    EventArgument(
                        role=role,
                        text=ent.text,
                        start=ent.start_char,
                        end=ent.end_char,
                    )
                )

        return arguments

    def _clean_entity_text(self, text: str) -> str:
        """Clean extracted entity text.
        
        - Remove leading/trailing punctuation
        - Remove common prepositions/articles
        - Collapse whitespace
        """
        # Remove punctuation
        text = re.sub(r"^[^\w]+|[^\w]+$", "", text)

        # Remove common lead-in words
        text = re.sub(r"^(?:the|a|an|and|or|but|with|of|from|to|for|in|by|at)\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+(?:the|a|an|and|or|but|with|of|from|to|for|in|by|at)$", "", text, flags=re.IGNORECASE)

        # Collapse whitespace
        text = " ".join(text.split())

        return text if text else ""

    def batch_extract(
        self, events: List[DetectedEvent]
    ) -> List[ExtractedEvent]:
        """Extract arguments from multiple events.
        
        Args:
            events: List of detected events
            
        Returns:
            List of extracted events
        """
        extracted = []
        for event in events:
            try:
                extracted_event = self.extract(event)
                extracted.append(extracted_event)
            except Exception as e:
                logger.error(f"Failed to extract event: {e}")
                # Still add the event, just without arguments
                extracted.append(
                    ExtractedEvent(
                        event_type=event.event_type,
                        sentence=event.sentence,
                        sentence_index=event.sentence_index,
                        chunk_id=event.chunk_id,
                        parva=event.parva,
                        section=event.section,
                        arguments=[],
                        tier=event.tier,  # Preserve tier
                    )
                )

        return extracted
