import hashlib
import statistics
from datetime import datetime
from typing import Dict, List


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_chunk_metadata(
    chunks: List[Dict],
    input_path: str,
    model_name: str,
    tokenizer_name: str,
    token_limits: Dict,
) -> Dict:
    parva_counts: Dict[str, int] = {}
    for chunk in chunks:
        name = chunk.get("parva_name", "Unknown")
        parva_counts[name] = parva_counts.get(name, 0) + 1

    return {
        "created_at": datetime.utcnow().isoformat(),
        "input_hash": sha256_file(input_path),
        "input_file": input_path,
        "chunk_count": len(chunks),
        "parva_chunk_counts": parva_counts,
        "model": model_name,
        "tokenizer": tokenizer_name,
        "token_limits": token_limits,
        "source": "KM Ganguly",
        "language": "English",
    }


def build_chunk_stats(chunks: List[Dict]) -> Dict:
    token_counts = [c.get("token_count", 0) for c in chunks]
    total_tokens = sum(token_counts)
    per_parva: Dict[str, Dict[str, int]] = {}
    for chunk in chunks:
        name = chunk.get("parva_name", "Unknown")
        per_parva.setdefault(name, {"chunks": 0, "tokens": 0})
        per_parva[name]["chunks"] += 1
        per_parva[name]["tokens"] += chunk.get("token_count", 0)

    return {
        "total_chunks": len(chunks),
        "total_tokens": total_tokens,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "avg_tokens": statistics.mean(token_counts) if token_counts else 0,
        "per_parva": per_parva,
    }
