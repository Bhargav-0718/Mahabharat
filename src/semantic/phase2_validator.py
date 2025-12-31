import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime

# Support execution as a script (python src/semantic/phase2_validator.py)
if __package__ is None or __package__ == "":
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.append(str(src_dir))
    from semantic.validators import ChunkValidator
else:
    from .validators import ChunkValidator

logger = logging.getLogger(__name__)


def _load_chunks(path: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def _chunk_stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    token_counts = [int(c.get("token_count", 0)) for c in chunks]
    parva_counts: Dict[str, int] = {}
    for c in chunks:
        name = c.get("parva_name", "Unknown")
        parva_counts[name] = parva_counts.get(name, 0) + 1

    total_tokens = sum(token_counts)
    below_40 = sum(1 for t in token_counts if t < 40)
    below_84 = sum(1 for t in token_counts if 40 <= t < 84)
    below_120 = sum(1 for t in token_counts if 84 <= t < 120)
    above_800 = sum(1 for t in token_counts if t > 800)

    return {
        "chunk_count": len(chunks),
        "total_tokens": total_tokens,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "avg_tokens": total_tokens / len(token_counts) if token_counts else 0.0,
        "per_parva": parva_counts,
        "below_absolute_floor": below_40,
        "below_soft_min": below_84,
        "below_preferred": below_120,
        "above_max": above_800,
    }


def _validate_chunks_file(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    report = {
        "file": str(path),
        "valid": True,
        "errors": [],
        "warnings": [],
        "statistics": {},
    }

    if not path.exists():
        report["valid"] = False
        report["errors"].append(f"File not found: {path}")
        return report, []

    try:
        chunks = _load_chunks(path)
        stats = _chunk_stats(chunks)
        report["statistics"] = stats

        try:
            ChunkValidator.validate_chunks(chunks, min_tokens=120, max_tokens=800)
        except Exception as exc:  # Keep report and continue
            report["valid"] = False
            report["errors"].append(f"Chunk validation failed: {exc}")

    except Exception as exc:
        report["valid"] = False
        report["errors"].append(f"Error reading chunks: {exc}")
        chunks = []

    return report, chunks


def _validate_metadata(path: Path, expected_count: int) -> Dict[str, Any]:
    report = {
        "file": str(path),
        "valid": True,
        "errors": [],
        "warnings": [],
        "statistics": {},
    }

    if not path.exists():
        report["valid"] = False
        report["errors"].append(f"File not found: {path}")
        return report

    try:
        with path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        chunk_count = meta.get("chunk_count")
        model = meta.get("model")
        tokenizer = meta.get("tokenizer")
        token_limits = meta.get("token_limits", {})

        if chunk_count is None:
            report["valid"] = False
            report["errors"].append("Missing chunk_count in metadata")
        elif expected_count and chunk_count != expected_count:
            report["warnings"].append(
                f"chunk_count mismatch: metadata={chunk_count}, actual={expected_count}"
            )

        report["statistics"] = {
            "chunk_count": chunk_count,
            "model": model,
            "tokenizer": tokenizer,
            "token_limits": token_limits,
            "parvas_with_chunks": len(meta.get("parva_chunk_counts", {})),
        }

    except Exception as exc:
        report["valid"] = False
        report["errors"].append(f"Error reading metadata: {exc}")

    return report


def _validate_chunk_stats(path: Path, expected: Dict[str, Any]) -> Dict[str, Any]:
    report = {
        "file": str(path),
        "valid": True,
        "errors": [],
        "warnings": [],
        "statistics": {},
    }

    if not path.exists():
        report["valid"] = False
        report["errors"].append(f"File not found: {path}")
        return report

    try:
        with path.open("r", encoding="utf-8") as f:
            stats = json.load(f)
        report["statistics"] = stats

        expected_count = expected.get("chunk_count")
        expected_tokens = expected.get("total_tokens")
        expected_min = expected.get("min_tokens")
        expected_max = expected.get("max_tokens")

        if expected_count and stats.get("total_chunks") != expected_count:
            report["warnings"].append(
                f"total_chunks mismatch: stats={stats.get('total_chunks')}, actual={expected_count}"
            )
        if expected_tokens and stats.get("total_tokens") != expected_tokens:
            report["warnings"].append(
                f"total_tokens mismatch: stats={stats.get('total_tokens')}, actual={expected_tokens}"
            )
        if expected_min and stats.get("min_tokens") != expected_min:
            report["warnings"].append(
                f"min_tokens mismatch: stats={stats.get('min_tokens')}, actual={expected_min}"
            )
        if expected_max and stats.get("max_tokens") != expected_max:
            report["warnings"].append(
                f"max_tokens mismatch: stats={stats.get('max_tokens')}, actual={expected_max}"
            )

    except Exception as exc:
        report["valid"] = False
        report["errors"].append(f"Error reading chunk_stats: {exc}")

    return report


def _validate_embeddings(path: Path, expected_count: int, expected_dim: int) -> Dict[str, Any]:
    report = {
        "file": str(path),
        "valid": True,
        "errors": [],
        "warnings": [],
        "statistics": {},
    }

    if not path.exists():
        report["valid"] = False
        report["errors"].append(f"File not found: {path}")
        return report

    try:
        with path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        model = manifest.get("model")
        dimension = manifest.get("dimension")
        count = manifest.get("count")
        chunks = manifest.get("chunks", [])

        # Count fallbacks if needed
        count_from_list = len(chunks)
        effective_count = count if count is not None else count_from_list

        if expected_count and effective_count != expected_count:
            report["warnings"].append(
                f"Embedding count mismatch: manifest={effective_count}, chunks={expected_count}"
            )

        # Sample dimension check without scanning all
        sample_dim = None
        if chunks:
            emb = chunks[0].get("embedding", [])
            sample_dim = len(emb)
            if dimension and sample_dim != dimension:
                report["errors"].append(
                    f"Embedding dimension mismatch: declared={dimension}, sample={sample_dim}"
                )
                report["valid"] = False
            if expected_dim and sample_dim != expected_dim:
                report["warnings"].append(
                    f"Embedding dimension differs from expected: sample={sample_dim}, expected={expected_dim}"
                )

        report["statistics"] = {
            "model": model,
            "dimension": dimension,
            "count": effective_count,
            "sample_dimension": sample_dim,
        }

    except Exception as exc:
        report["valid"] = False
        report["errors"].append(f"Error reading embedding_manifest: {exc}")

    return report


def _validate_checkpoint(path: Path, expected_parvas: int = 18) -> Dict[str, Any]:
    report = {
        "file": str(path),
        "valid": True,
        "errors": [],
        "warnings": [],
        "statistics": {},
    }

    if not path.exists():
        report["warnings"].append("Checkpoint not found (resume still possible but not tracked)")
        return report

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        processed = data.get("processed_parvas", [])
        report["statistics"] = {"processed_parvas": processed, "intermediate_chunks": len(data.get("intermediate_chunks", []))}
        if expected_parvas and len(processed) != expected_parvas:
            report["warnings"].append(
                f"Checkpoint has {len(processed)} parvas processed, expected {expected_parvas}"
            )
    except Exception as exc:
        report["valid"] = False
        report["errors"].append(f"Error reading checkpoint: {exc}")

    return report


def run_validation(
    chunks_path: Path = Path("data/semantic_chunks/chunks.jsonl"),
    metadata_path: Path = Path("data/semantic_chunks/chunk_metadata.json"),
    stats_path: Path = Path("data/semantic_chunks/chunk_stats.json"),
    embeddings_path: Path = Path("data/semantic_chunks/embedding_manifest.json"),
    checkpoint_path: Path = Path("data/semantic_chunks/parva_checkpoint.json"),
    report_path: Path = Path("phase2_validation_report.json"),
) -> Dict[str, Any]:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

    chunks_report, chunks = _validate_chunks_file(chunks_path)
    chunk_stats_values = chunks_report.get("statistics", {})

    metadata_report = _validate_metadata(metadata_path, chunk_stats_values.get("chunk_count"))
    stats_report = _validate_chunk_stats(stats_path, chunk_stats_values)
    embeddings_report = _validate_embeddings(
        embeddings_path,
        expected_count=chunk_stats_values.get("chunk_count", 0),
        expected_dim=768,
    )
    checkpoint_report = _validate_checkpoint(checkpoint_path)

    overall_valid = all(
        section.get("valid", False)
        for section in [chunks_report, metadata_report, stats_report, embeddings_report, checkpoint_report]
    )

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "chunks": chunks_report,
        "metadata": metadata_report,
        "stats": stats_report,
        "embeddings": embeddings_report,
        "checkpoint": checkpoint_report,
        "overall_valid": overall_valid,
    }

    try:
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.error("Failed to write validation report: %s", exc)

    return report


def main() -> None:
    run_validation()


if __name__ == "__main__":
    main()
