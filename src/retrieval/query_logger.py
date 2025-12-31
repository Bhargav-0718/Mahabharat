import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


class QueryLogger:
    """Append-only JSONL logger for Phase 3 queries."""

    def __init__(self, log_path: Path = Path("data/retrieval/query_log.jsonl")) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, query: str, retrieved_ids: List[str], used_ids: List[str]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "retrieved_chunk_ids": retrieved_ids,
            "used_chunk_ids": used_ids,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
