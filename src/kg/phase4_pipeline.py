"""Phase 4: Event-centric knowledge graph construction.

Pipeline:
1. Load semantic chunks from Phase 3
2. Detect events using rule-based patterns
3. Extract event arguments (subject/object)
4. Create entities ONLY from event participants (admission control)
5. Normalize aliases
6. Build knowledge graph
7. Validate graph
8. Save outputs
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import jsonlines
from tqdm import tqdm

from .alias_resolver import AliasResolver
from .entity_registry import EntityRegistry
from .event_detector import EventDetector, DetectedEvent
from .event_extractor import EventExtractor
from .knowledge_graph import KnowledgeGraph
from .kg_validators import GraphValidator
from .phase4_postprocess import postprocess_graph

logger = logging.getLogger(__name__)


class Phase4Pipeline:
    """Event-centric knowledge graph construction."""

    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.event_detector = EventDetector()
        self.event_extractor = EventExtractor()
        self.entity_registry = EntityRegistry()
        self.graph = KnowledgeGraph(self.entity_registry)

        # Tracking
        self.chunk_count = 0
        self.event_count = 0
        self.event_count_by_type: Dict[str, int] = {}
        # FIX 4: Track rejections
        self.extracted_count = 0
        self.admitted_count = 0
        self.rejected_reasons: Dict[str, int] = {}

    def run(self) -> None:
        """Run the full Phase 4 pipeline."""
        logger.info("=" * 80)
        logger.info("PHASE 4: Event-Centric Knowledge Graph Construction")
        logger.info("=" * 80)

        # Load chunks from Phase 3
        chunks = self._load_chunks()
        logger.info(f"Loaded {len(chunks)} chunks from Phase 3")

        # Stage 1: Detect events
        all_detected = self._detect_events(chunks)
        logger.info(f"Detected {len(all_detected)} events total")
        logger.info(f"Event type breakdown: {self.event_count_by_type}")

        # Stage 2: Extract arguments
        all_extracted = self._extract_arguments(all_detected)
        logger.info(f"Extracted arguments from {len(all_extracted)} events")

        # Stage 3: Build graph
        self._build_graph(all_extracted)
        logger.info(f"Built graph with {self.graph.entity_count()} entities, "
                    f"{self.graph.event_count()} events, "
                    f"{self.graph.edge_count()} edges")
        
        # FIX 4: Log admission summary
        logger.info(f"Event admission: {self.admitted_count}/{self.extracted_count} extracted → admitted")

        # Stage 4: Post-processing (Fixes D, E, F)
        postprocess_graph(self.graph, self.entity_registry)
        logger.info(f"After post-processing: {self.graph.entity_count()} entities, "
                    f"{self.graph.event_count()} events, "
                    f"{self.graph.edge_count()} edges")

        # Stage 5: Validate
        validator = GraphValidator(self.graph)
        if validator.validate():
            logger.info("Graph validation PASSED")
        else:
            logger.warning("Graph validation found errors")

        # Stage 6: Save outputs
        self._save_outputs(validator)
        logger.info(f"Saved outputs to {self.output_dir}")

        logger.info("=" * 80)
        logger.info("Phase 4 COMPLETE")
        logger.info("=" * 80)

    def _load_chunks(self) -> List[Dict]:
        """Load semantic chunks from Phase 3.
        
        Expects: data/parsed_text/parsed_pages.jsonl
        """
        chunk_file = self.input_dir / "parsed_pages.jsonl"
        if not chunk_file.exists():
            logger.warning(f"Chunk file not found: {chunk_file}")
            return []

        chunks = []
        with jsonlines.open(chunk_file) as reader:
            for obj in reader:
                chunks.append(obj)

        return chunks

    def _detect_events(self, chunks: List[Dict]) -> List[DetectedEvent]:
        """Detect events in all chunks.
        
        Args:
            chunks: List of semantic chunks
            
        Returns:
            List of detected events
        """
        all_events = []

        for chunk in tqdm(chunks, desc="Detecting events", unit="chunk"):
            chunk_id = chunk.get("chunk_id", "unknown")
            parva = chunk.get("parva", "unknown")
            section = chunk.get("section", "unknown")
            text = chunk.get("text", "")

            events = self.event_detector.detect_events(text, chunk_id, parva, section)
            all_events.extend(events)

            # Track
            for event in events:
                self.event_count_by_type[event.event_type] = \
                    self.event_count_by_type.get(event.event_type, 0) + 1

        return all_events

    def _extract_arguments(self, events: List[DetectedEvent]) -> List:
        """Extract arguments from detected events.
        
        Args:
            events: List of detected events
            
        Returns:
            List of extracted events with valid arguments
        """
        extracted = []

        for event in tqdm(events, desc="Extracting arguments", unit="event"):
            try:
                ext_event = self.event_extractor.extract(event)
                # FIX C: Discard events with fewer than 1 valid entity
                if len(ext_event.arguments) >= 1:
                    extracted.append(ext_event)
            except Exception as e:
                logger.error(f"Failed to extract arguments from event: {e}")

        logger.info(f"Kept {len(extracted)} events with valid arguments (filtered {len(events) - len(extracted)})")
        return extracted

    def _build_graph(self, events: List) -> None:
        """Build knowledge graph from extracted events.
        
        Args:
            events: List of extracted events
        """
        # FIX 4: Track admissions
        self.extracted_count = len(events)
        for event in tqdm(events, desc="Building graph", unit="event"):
            event_id = f"{event.chunk_id}:{event.sentence_index}"
            self.graph.add_event(event, event_id)
        
        self.admitted_count = self.graph.event_count()
        if self.extracted_count > self.admitted_count:
            logger.info(f"Admission filter: {self.extracted_count - self.admitted_count} events filtered out")

    def _save_outputs(self, validator: GraphValidator) -> None:
        """Save graph and validation report.
        
        Outputs:
        - entities.json: Entity registry
        - events.json: Event list
        - edges.json: Knowledge graph edges
        - graph_stats.json: Summary statistics
        - validation_report.json: Validation results
        """
        # Save graph
        self.graph.save(self.output_dir)

        # Save validation report
        report = validator.get_report()
        with open(self.output_dir / "validation_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Save entity registry metadata
        registry_data = self.entity_registry.to_dict()
        with open(self.output_dir / "entity_registry.json", "w") as f:
            json.dump(registry_data, f, indent=2)

        logger.info(f"Saved {len(self.entity_registry.entities)} entities")
        logger.info(f"Saved {len(self.graph.events)} events")
        logger.info(f"Saved {len(self.graph.edges)} edges")


def main(data_dir: str = "data/parsed_text", output_dir: str = "data/kg"):
    """Run Phase 4 pipeline.
    
    Args:
        data_dir: Directory containing parsed chunks
        output_dir: Directory for KG outputs
    """
    pipeline = Phase4Pipeline(Path(data_dir), Path(output_dir))
    pipeline.run()


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "data/parsed_text"

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "data/kg"

    main(data_dir, output_dir)
