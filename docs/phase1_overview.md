# Phase 1 Overview — PDF Parsing and Structural Extraction

This document captures what Phase 1 does, why we made specific choices, and how to run/validate it.

## Goals
- Parse the KM Ganguly Mahabharata PDF into clean page-level text.
- Extract the canonical 18 Parvas with SECTION boundaries and paragraphs.
- Make iteration fast: reuse cached parsed pages and support checkpoint/resume.

## Inputs and Outputs
- Input PDF: data/raw_pdf/MahabharataOfVyasa-EnglishTranslationByKMGanguli.pdf
- Primary outputs:
  - Parsed pages: data/parsed_text/parsed_pages.jsonl (5818 pages)
  - Structured document: data/parsed_text/mahabharata_structure.json (18 Parvas)
  - Checkpoint: data/parsed_text/phase1_checkpoint.json
  - Validation report: phase1_validation_report.json

## Pipeline Stages
Implementation: [src/ingestion/phase1_pipeline.py](src/ingestion/phase1_pipeline.py)

1) PDF Parsing (Stage 1)
- Uses [src/ingestion/pdf_parser.py](src/ingestion/pdf_parser.py) to read the PDF and produce JSONL pages.
- Each page record includes page_number, text, word_count, has_section_marker, has_parva_marker.
- If data/parsed_text/parsed_pages.jsonl already exists, Stage 1 is skipped and cached pages are loaded.

2) Section Extraction (Stage 2)
- Uses [src/ingestion/section_extractor.py](src/ingestion/section_extractor.py) to build a Parva -> Section -> Paragraph hierarchy.
- Steps inside the extractor:
  - Frontmatter cutoff: scan pages >= 50 for the first BOOK header, drop earlier pages.
  - Combine remaining pages to a single string.
  - BOOK header detection:
    - Multi-line pattern with optional “Part one of three” lines between BOOK and PARVA.
    - Inline pattern for colon style (BOOK X: <Name> Parva).
    - Deduplicate: keep only the first header per book number to handle multi-part PDFs (Books 12/13 parts).
  - SECTION header detection:
    - Matches classic SECTION N and lines like “The Mahabharata, Book X: <Parva>: Section N”.
    - Deduplicate section numbers to avoid splitting on repeated page headers.
  - Paragraph extraction:
    - Normalize single newlines to spaces, split on blank lines, with sentence fallback for very long single paragraphs.
  - Narrative filter:
    - Keep sections once real narrative appears (length/mixed-case heuristic); prevents pure navigation stubs.
  - Fallback: if a Parva has no SECTION markers, emit a single FULL_TEXT section.

## Checkpoint/Resume
- Checkpoint file: data/parsed_text/phase1_checkpoint.json
- Behavior:
  - If parsed_pages.jsonl exists, Stage 1 is marked complete and skipped.
  - If mahabharata_structure.json exists and checkpoint says section_extraction_complete, Stage 2 is skipped.
  - Hashes of PDF and outputs are stored to detect changes.

## Key Heuristics and Regexes (section_extractor.py)
- BOOK (multi-line): allows up to three intervening lines between BOOK and PARVA to catch part headers.
- BOOK (inline): BOOK <n>: <NAME> PARVA
- SECTION: matches SECTION <num> or “The Mahabharata … Section <num>”.
- Deduping:
  - Keep only first BOOK per number.
  - Keep only first SECTION per section_number.
- TOC noise removal: drop “Table of Contents Index …” and “Downloaded from:” lines without removing SECTION headers.

## Current Results (latest run)
- Parvas: 18
- Sections: 2,103
- Paragraphs: 110,733
- Section counts (examples): Adi 236, Sabha 79, Vana 313, Karna 95, Shanti 364, Anushasan 167, Svargarohanika 6.
- See [data/parsed_text/mahabharata_structure.json](data/parsed_text/mahabharata_structure.json) for full counts.

## How to Run
- From repo root (venv active):
  - Phase 1 pipeline: `D:/AI/Mahabharat/.venv/Scripts/python.exe src/ingestion/phase1_pipeline.py`
  - Validation: `D:/AI/Mahabharat/.venv/Scripts/python.exe src/ingestion/phase1_validator.py`
- Both commands reuse cached parsed_pages.jsonl; only Section Extraction reruns if checkpoint is reset.

## Validation (phase1_validator.py)
- parsed_pages checks: file readability, required fields, optional marker presence warnings, sequential pages, basic stats.
- structure checks: exactly 18 Parvas (hard), duplicate Parva/section detection (hard), canonical name warnings, section_count vs actual, empty-section warnings, totals.
- Output: phase1_validation_report.json plus console summary.

## Known Caveats / Follow-ups
- Navigation text can appear inside early paragraphs of sections; if needed, add stricter trimming for TOC lines inside sections.
- Sentence-level splitting is minimal; could refine for better paragraph balance in very long paragraphs.
- If PDF changes, hashes in the checkpoint will invalidate resume and force re-parse.

## Where to Start for Phase 2
- Use mahabharata_structure.json as the source for context-unit building and embeddings.
- The cached parsed_pages.jsonl remains the single source of raw text if chunking needs to be revisited.
