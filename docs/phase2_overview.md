# Phase 2 Overview - Semantic Chunking and Embeddings

This document captures what Phase 2 does, why we made specific choices, and how to run/validate it.

## Goals
- Convert Phase 1 structure into semantic chunks sized for retrieval (target ~450 tokens, min 120 preferred, max 800).
- Generate deterministic embeddings using a free, long-context model (8192 token limit) with clean logging and resume.
- Preserve small but meaningful chunks via soft minimums; avoid crashes on edge cases.

## Inputs and Outputs
- Input: data/parsed_text/mahabharata_structure.json (from Phase 1)
- Primary outputs (data/semantic_chunks/):
  - chunks.jsonl (8,632 chunks)
  - chunk_metadata.json (counts, model info, token limits)
  - chunk_stats.json (token stats per Parva)
  - embedding_manifest.json (8,632 embeddings, 768-dim, ~191 MB)
  - parva_checkpoint.json (resume state per Parva)
  - phase2_checkpoint.json (input/model hash, completion status)

## Pipeline Stages
Implementation: [src/semantic/phase2_pipeline.py](src/semantic/phase2_pipeline.py)

1) Load & configure
- Loads tokenizer and model: jinaai/jina-embeddings-v2-base-en (768 dims, 8192 token support).
- Sets token limits: min 120 (preferred), soft minimum 84, absolute floor 40, target 450, max 800.

2) Semantic chunking
- [src/semantic/semantic_chunker.py](src/semantic/semantic_chunker.py)
- Paragraph normalization: [src/semantic/paragraph_normalizer.py](src/semantic/paragraph_normalizer.py) removes nav/header noise and collapses whitespace.
- For each Parva/Section:
  - Paragraphs over max are sentence-split to stay under 800 tokens.
  - Paragraphs embed incrementally; cosine similarity (< 0.35) or target/max triggers a chunk boundary.
  - Small chunks attempt backward/forward merge if under max; otherwise accepted if above floor 40 (soft below 120).
  - Deterministic IDs: P{parva:02d}-S{section:03d}-C{chunk:03d}.

3) Validation & metadata
- [src/semantic/validators.py](src/semantic/validators.py) enforces required fields, duplicate detection, max-token hard limit, and soft minimums (warn 40-83, warn 84-119, pass >=120; error <40).
- [src/semantic/metadata_builder.py](src/semantic/metadata_builder.py) builds chunk_metadata.json, chunk_stats.json, and hashes.

4) Embeddings
- [src/semantic/embedder.py](src/semantic/embedder.py) wraps SentenceTransformer for batch encode.
- Embedding manifest stores model, dimension, and per-chunk vectors.

## Checkpoint/Resume
- Parva-level resume: data/semantic_chunks/parva_checkpoint.json stores processed_parvas and intermediate chunks; reruns skip completed Parvas.
- Completion checkpoint: data/semantic_chunks/phase2_checkpoint.json stores input hash, model name, and status to skip reruns unless --force is used.

## Key Heuristics
- Similarity split: cosine < 0.35 starts a new chunk (after embedding current paragraph).
- Token targets: aim 450; stop adding when exceeding max 800.
- Soft minimum: prefer >=120, allow 84-119 (silent), warn 40-83, fail <40.
- Long paragraph handling: sentence-based splitting with fallback hard word splits to respect 800 cap.

## Current Results (2025-12-31)
- Chunks: 8,632
- Tokens: min 48, max 800, avg ~477 (total 4,117,459)
- Embeddings: 8,632 vectors, 768-dim, model jinaai/jina-embeddings-v2-base-en
- Coverage: all 18 Parvas processed
- Files: chunks.jsonl, chunk_metadata.json, chunk_stats.json, embedding_manifest.json, parva_checkpoint.json, phase2_checkpoint.json

## How to Run
- From repo root (venv active):
  - Full pipeline: `D:/AI/Mahabharat/.venv/Scripts/python.exe src/semantic/phase2_pipeline.py`
  - Validate existing outputs only: `D:/AI/Mahabharat/.venv/Scripts/python.exe src/semantic/phase2_pipeline.py --validate-only`
  - Force recompute: add `--force`
  - Adjust similarity threshold (default 0.35): `--similarity-threshold 0.33` (example)

## Validation
- validate-only path checks presence of chunks.jsonl and embedding_manifest.json, validates chunk fields, max tokens, soft minimums, duplicate IDs, and embedding counts/dimensions.
- chunk_stats.json reports min/max/avg tokens and totals; chunk_metadata.json echoes model and token limits.

## Known Caveats / Follow-ups
- embedding_manifest.json is large (~191 MB); avoid loading fully unless needed.
- A few chunks fall below preferred minimum (120) but above floor (48) by design to preserve content.
- If input structure changes, checkpoints invalidate and rerun is required (or use --force).
