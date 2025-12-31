
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm
from transformers import AutoTokenizer

# Support execution as a script (python src/semantic/phase2_pipeline.py)
if __package__ is None or __package__ == "":
    # Add the src/ directory to sys.path so that "import semantic" works
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.append(str(src_dir))
    from semantic.embedder import Embedder
    from semantic.metadata_builder import build_chunk_metadata, build_chunk_stats, sha256_file
    from semantic.semantic_chunker import SemanticChunker
    from semantic.validators import ChunkValidator
else:
    from .embedder import Embedder
    from .metadata_builder import build_chunk_metadata, build_chunk_stats, sha256_file
    from .semantic_chunker import SemanticChunker
    from .validators import ChunkValidator

logger = logging.getLogger(__name__)


def load_structure(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("mahabharata", {}).get("parvas", [])


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_chunks(path: Path) -> List[Dict]:
    chunks: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def load_embedding_manifest(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_embedding_manifest(model_name: str, dimension: int, embeddings: List[List[float]], chunk_ids: List[str]) -> Dict:
    records = []
    for cid, emb in zip(chunk_ids, embeddings):
        records.append({"chunk_id": cid, "embedding": emb})
    return {
        "model": model_name,
        "dimension": dimension,
        "count": len(records),
        "chunks": records,
    }


def outputs_exist(output_dir: Path) -> bool:
    required = [
        output_dir / "chunks.jsonl",
        output_dir / "chunk_metadata.json",
        output_dir / "chunk_stats.json",
        output_dir / "embedding_manifest.json",
    ]
    return all(p.exists() for p in required)


def load_parva_checkpoint(path: Path) -> Dict:
    if not path.exists():
        return {"processed_parvas": [], "intermediate_chunks": []}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"processed_parvas": [], "intermediate_chunks": []}


def save_parva_checkpoint(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_checkpoint(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_checkpoint(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def is_up_to_date(checkpoint: Dict, input_hash: str, model_name: str) -> bool:
    return (
        checkpoint.get("input_hash") == input_hash
        and checkpoint.get("model_name") == model_name
        and checkpoint.get("status") == "complete"
    )


def run_pipeline(
    input_file: str,
    output_dir: str,
    model_name: str,
    dry_run: bool,
    validate_only: bool,
    verbose: bool,
    force: bool,
    similarity_threshold: float,
) -> None:
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input structure not found: {input_file}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_path / "phase2_checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path)
    input_hash = sha256_file(str(input_path))

    if validate_only:
        logger.info("Running validation-only mode")
        _validate_outputs(output_path, input_file)
        return

    if outputs_exist(output_path) and is_up_to_date(checkpoint, input_hash, model_name) and not force:
        logger.info("Outputs are up-to-date; skipping recomputation")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedder = Embedder(model_name=model_name)

    # Cap max_tokens at tokenizer's model_max_length to avoid indexing errors
    tokenizer_max = getattr(tokenizer, 'model_max_length', 8192)
    safe_max = min(800, tokenizer_max - 50)  # Leave buffer for special tokens
    safe_min = 120
    safe_target = 450

    logger.info(f"Token limits: min={safe_min}, target={safe_target}, max={safe_max} (model supports up to {tokenizer_max})")

    chunker = SemanticChunker(
        tokenizer=tokenizer,
        embedder=embedder,
        target_tokens=safe_target,
        min_tokens=safe_min,
        max_tokens=safe_max,
        similarity_threshold=similarity_threshold,
    )

    logger.info("Loading structure and chunking...")
    parvas = load_structure(str(input_path))
    logger.info(f"Processing {len(parvas)} Parvas...")
    
    parva_checkpoint_path = output_path / "parva_checkpoint.json"
    parva_checkpoint = load_parva_checkpoint(parva_checkpoint_path)
    processed_parva_numbers = set(parva_checkpoint.get("processed_parvas", []))
    intermediate_chunks = parva_checkpoint.get("intermediate_chunks", [])
    
    for parva in tqdm(parvas, desc="Chunking Parvas", unit="parva"):
        parva_num = parva.get("parva_number")
        if parva_num in processed_parva_numbers:
            logger.info(f"Skipping Parva {parva_num} (already processed)")
            continue
        logger.info(f"Processing Parva {parva_num}: {parva.get('parva_name')}")
        parva_chunks = chunker._chunk_parva(parva)
        intermediate_chunks.extend(parva_chunks)
        processed_parva_numbers.add(parva_num)
        parva_checkpoint["processed_parvas"] = sorted(list(processed_parva_numbers))
        parva_checkpoint["intermediate_chunks"] = intermediate_chunks
        save_parva_checkpoint(parva_checkpoint_path, parva_checkpoint)
        logger.info(f"Saved checkpoint: {len(intermediate_chunks)} total chunks so far")
    
    chunks = intermediate_chunks

    ChunkValidator.validate_chunks(chunks, min_tokens=120, max_tokens=800)
    ChunkValidator.log_stats(chunks)

    logger.info(f"Starting embedding generation for {len(chunks)} chunks...")

    texts = [c["text"] for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings_np = embedder.embed_texts(texts)
    ChunkValidator.validate_embeddings(chunks, embeddings_np)

    embeddings = [emb.tolist() for emb in embeddings_np]
    manifest = build_embedding_manifest(embedder.model_name, embedder.dimension, embeddings, [c["chunk_id"] for c in chunks])

    metadata = build_chunk_metadata(
        chunks=chunks,
        input_path=str(input_path),
        model_name=embedder.model_name,
        tokenizer_name=tokenizer.name_or_path,
        token_limits={"target": 450, "min": 120, "max": 800},
    )
    stats = build_chunk_stats(chunks)

    if dry_run:
        logger.info("Dry run enabled; outputs will not be written")
        return

    logger.info("Writing outputs to %s", output_path)
    write_jsonl(output_path / "chunks.jsonl", chunks)
    logger.info(f"Wrote chunks.jsonl")
    write_json(output_path / "chunk_metadata.json", metadata)
    logger.info(f"Wrote chunk_metadata.json")
    write_json(output_path / "chunk_stats.json", stats)
    logger.info(f"Wrote chunk_stats.json")
    write_json(output_path / "embedding_manifest.json", manifest)
    logger.info(f"Wrote embedding_manifest.json")

    checkpoint = {
        "input_hash": input_hash,
        "model_name": model_name,
        "status": "complete",
        "outputs": {
            "chunks": str(output_path / "chunks.jsonl"),
            "metadata": str(output_path / "chunk_metadata.json"),
            "stats": str(output_path / "chunk_stats.json"),
            "embedding_manifest": str(output_path / "embedding_manifest.json"),
        },
    }
    save_checkpoint(checkpoint_path, checkpoint)
    logger.info("Phase 2 pipeline complete: %s chunks", len(chunks))


def _validate_outputs(output_path: Path, input_file: str) -> None:
    chunks_path = output_path / "chunks.jsonl"
    manifest_path = output_path / "embedding_manifest.json"
    if not chunks_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Required output files for validation are missing")

    chunks = load_chunks(chunks_path)
    ChunkValidator.validate_chunks(chunks, min_tokens=120, max_tokens=800)

    manifest = load_embedding_manifest(manifest_path)
    embeddings = manifest.get("chunks", [])
    ChunkValidator.validate_embeddings(chunks, embeddings)
    ChunkValidator.log_stats(chunks)
    logger.info("Validation-only completed successfully")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: Semantic Chunking and Embeddings")
    parser.add_argument("--input", default="data/parsed_text/mahabharata_structure.json", help="Path to Phase 1 structure JSON")
    parser.add_argument("--output-dir", default="data/semantic_chunks", help="Output directory for chunks and embeddings")
    parser.add_argument("--model", default="jinaai/jina-embeddings-v2-base-en", help="SentenceTransformer model name")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing outputs")
    parser.add_argument("--validate-only", action="store_true", help="Validate existing outputs")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if outputs are up-to-date")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--similarity-threshold", type=float, default=0.35, help="Cosine similarity threshold for chunk splitting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        input_file=args.input,
        output_dir=args.output_dir,
        model_name=args.model,
        dry_run=args.dry_run,
        validate_only=args.validate_only,
        verbose=args.verbose,
        force=args.force,
        similarity_threshold=args.similarity_threshold,
    )


if __name__ == "__main__":
    main()
