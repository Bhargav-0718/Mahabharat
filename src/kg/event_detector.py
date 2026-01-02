"""Event detection using rule-based patterns.

Detects 9 core event types:
- KILL, COMMAND, BATTLE, VOW, BIRTH, CURSE, BOON, EXILE, DEATH
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# Event keyword patterns (case-insensitive)
# Tier-1: MACRO events (core narrative pivots)
# Tier-2: MESO events (tactical/relational actions)
EVENT_PATTERNS = {
    # === TIER-1: MACRO EVENTS ===
    "KILL": [
        r"\b(?:kill|killed|slay|slew|slain|struck?\s+down|struck?\s+dead|smote|smitten|slaughter|beheaded|beheading|decapitated|murder|murdered)\b",
    ],
    "COMMAND": [
        r"\b(?:command|commanded|led|ordered|instructed|directed|sent|dispatch|deputed)\b",
    ],
    "BATTLE": [
        r"\b(?:battle|fought|fought\s+(?:with|against)|clash|combat|war|duel|skirmish|battle\s+(?:between|of)|fought\s+(?:in|at))\b",
    ],
    "VOW": [
        r"\b(?:vow|vowed|vow(?:ing|ed)|swore|sworn|swear|oath|promise|promised)\b",
    ],
    "CURSE": [
        r"\b(?:curse|cursed|curse\s+(?:upon|on)|cursing|accursed|doomed|condemned|imprecation)\b",
    ],
    "BOON": [
        r"\b(?:boon|granted\s+a\s+boon|boon\s+(?:from|of)|granted|granted\s+(?:to|him)|favour|favor|gift|bestowed)\b",
    ],
    "DEATH": [
        r"\b(?:died|death|perish|perished|fallen|fell|expire|expired|breathed\s+(?:his|her)\s+last|passed\s+away|succumb|succumbed)\b",
    ],
    "CORONATION": [
        r"\b(?:crowned|coronation|anointed|installed\s+as\s+king|made\s+king|enthroned|ascended\s+(?:the\s+)?throne)\b",
    ],

    # === TIER-2: MESO EVENTS ===
    "ENGAGED_IN_BATTLE": [
        r"\b(?:engaged\s+(?:in\s+)?(?:battle|combat|fight)|encountered\s+(?:in\s+)?battle|met\s+(?:in\s+)?battle|joined\s+battle\s+with|stood\s+against|opposed|confronted)\b",
        r"\b(?:advanced\s+against|assailed|attacked|surged\s+against)\b",
    ],
    "DEFEATED": [
        r"\b(?:defeated|overcame|vanquished|routed|conquered|subdued|overpowered|worsted|broke\s+the\s+ranks\s+of)\b",
    ],
    "PROTECTED": [
        r"\b(?:protected|defended|shielded|guarded|covered\s+the\s+retreat\s+of|supported|reinforced|held\s+firm\s+for)\b",
    ],
    "PURSUED": [
        r"\b(?:pursued|chased|followed|hunted|tracked)\b",
    ],
    "RESCUED": [
        r"\b(?:rescued|saved|delivered\s+from|liberated|freed)\b",
    ],
    "APPOINTED_AS": [
        r"\b(?:appointed\s+(?:as)?|installed\s+as|made\s+(?:commander|general|minister|king)|designated\s+as)\b",
    ],
    "ABANDONED": [
        r"\b(?:abandoned|left|forsook|deserted|renounced|retreated\s+before|fled\s+from|fell\s+back\s+before)\b",
    ],
    "ATTACKED": [
        r"\b(?:attacked|assailed|assaulted|struck\s+at|charged\s+at|fell\s+upon|made\s+war\s+upon)\b",
    ],
    "DEFENDED": [
        r"\b(?:defended|shielded|protected|held\s+the\s+line\s+against|stood\s+fast\s+against)\b",
    ],
    "RETREATED": [
        r"\b(?:retreated|fell\s+back|withdrew|fled\s+from|turned\s+back\s+before)\b",
    ],
    "SURROUNDED": [
        r"\b(?:surrounded|encompassed|hemmed\s+in|closed\s+in\s+upon)\b",
    ],
    "SUPPORTED": [
        r"\b(?:supported|reinforced|succoured|came\s+to\s+the\s+aid\s+of|covered\s+the\s+retreat\s+of)\b",
    ],
    "FORMED_ARRAY_AGAINST": [
        r"\b(?:formed\s+(?:an\s+)?array\s+against|arrayed\s+against|drew\s+up\s+against|entered\s+the\s+ranks\s+of)\b",
    ],
}

# Tier classification for each event type
EVENT_TIERS = {
    "KILL": "MACRO",
    "COMMAND": "MACRO",
    "BATTLE": "MACRO",
    "VOW": "MACRO",
    "CURSE": "MACRO",
    "BOON": "MACRO",
    "DEATH": "MACRO",
    "CORONATION": "MACRO",
    "ENGAGED_IN_BATTLE": "MESO",
    "DEFEATED": "MESO",
    "PROTECTED": "MESO",
    "PURSUED": "MESO",
    "RESCUED": "MESO",
    "APPOINTED_AS": "MESO",
    "ABANDONED": "MESO",
    "ATTACKED": "MESO",
    "DEFENDED": "MESO",
    "RETREATED": "MESO",
    "SURROUNDED": "MESO",
    "SUPPORTED": "MESO",
    "FORMED_ARRAY_AGAINST": "MESO",
}

# Micro event verbs to explicitly reject
REJECTED_MICRO_VERBS = {
    "said", "spoke", "replied", "answered", "told", "asked", "questioned",
    "went", "came", "arrived", "departed", "returned", "reached",
    "saw", "looked", "beheld", "witnessed", "observed",
    "thought", "knew", "understood", "remembered", "forgot",
    "stood", "sat", "lay", "arose", "rose",
}

# Compile patterns once
COMPILED_PATTERNS = {
    event_type: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for event_type, patterns in EVENT_PATTERNS.items()
}


@dataclass
class DetectedEvent:
    """Represents a detected event."""

    event_type: str
    sentence: str
    sentence_index: int  # Index within chunk
    chunk_id: str
    parva: str
    section: str
    tier: str  # "MACRO" or "MESO"


class EventDetector:
    """Rule-based event detector."""

    def __init__(self):
        pass
    
    def _clean_sentence(self, sentence: str) -> str:
        """FIX 3: Pre-clean sentence before event detection.
        
        Strip:
        - URLs (http://, .htm, www)
        - File markers (m04037.htm, file:///)
        - Narration prefixes (said,--, continued,--, replied,--)
        - Extra punctuation
        """
        # Remove URLs and file paths
        sentence = re.sub(r'https?://[^\s]+', '', sentence)
        sentence = re.sub(r'file://[^\s]+', '', sentence)
        sentence = re.sub(r'\b[mM]\d+(?:\w+)?\.htm', '', sentence)
        sentence = re.sub(r'\bwww\.\S+', '', sentence)
        
        # Remove narration prefixes with continuation markers
        sentence = re.sub(r'\b(?:said|spoke|replied|answered|continued),?--', '', sentence, flags=re.IGNORECASE)
        
        # Clean up extra punctuation and whitespace
        sentence = re.sub(r'\s+', ' ', sentence)
        sentence = sentence.strip()
        
        return sentence

    def detect_events(
        self,
        text: str,
        chunk_id: str,
        parva: str,
        section: str,
    ) -> List[DetectedEvent]:
        """Detect events in text.
        
        Args:
            text: Chunk text
            chunk_id: Chunk identifier
            parva: Parva name
            section: Section name
            
        Returns:
            List of detected events
        """
        events = []
        sentences = self._split_sentences(text)

        for sent_idx, sentence in enumerate(sentences):
            # FIX 3: Clean sentence before detection
            cleaned = self._clean_sentence(sentence)
            if not cleaned or len(cleaned) < 5:
                continue
            
            # Skip sentences with micro event verbs
            if self._contains_micro_verb(cleaned):
                continue
                
            # Check each event type
            for event_type, patterns in COMPILED_PATTERNS.items():
                for pattern in patterns:
                    if pattern.search(cleaned):
                        tier = EVENT_TIERS.get(event_type, "MESO")
                        events.append(
                            DetectedEvent(
                                event_type=event_type,
                                sentence=cleaned,  # Store cleaned sentence
                                sentence_index=sent_idx,
                                chunk_id=chunk_id,
                                parva=parva,
                                section=section,
                                tier=tier,
                            )
                        )
                        break  # Only record first match per sentence per type
                        
        return events

    def _contains_micro_verb(self, sentence: str) -> bool:
        """Check if sentence contains rejected micro event verbs."""
        sentence_lower = sentence.lower()
        for verb in REJECTED_MICRO_VERBS:
            if re.search(r'\b' + verb + r'\b', sentence_lower):
                return True
        return False

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting."""
        # Split on . ! ? followed by space or end
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
