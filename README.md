# Mahabharata-SemRAG: Contextual, Source-Grounded Question Answering System

## Overview

A **hallucination-resistant, temporally-aware, source-grounded QA system** over the complete Mahabharata (KM Ganguly Translation) using a **Semantic Retrieval-Augmented Generation (SemRAG)** architecture.

### Key Features

- **Narrative Continuity**: Preserves story phases and temporal context
- **Source-Grounded**: Every answer cites Parva + Section with verbatim passages
- **Temporal Reasoning**: Supports queries across 8 story phases
- **Graph-First Retrieval**: Knowledge graph precedes semantic search and LLM
- **Hallucination-Resistant**: LLM strictly constrained to retrieved content

## Project Structure

```
mahabharata-semrag/
├── data/
│   ├── raw_pdf/           # KM Ganguly Mahabharata PDF (input)
│   ├── parsed_text/       # Extracted, structured text
│   └── context_units/     # JSONL Context Unit store
├── src/
│   ├── ingestion/         # PDF parsing and extraction
│   ├── structuring/       # Text segmentation, Context Unit building
│   ├── graph/             # Knowledge graph construction
│   ├── embeddings/        # OpenAI embedding integration
│   ├── retrieval/         # Graph + semantic retrieval
│   └── generation/        # Answer formatting
├── app/                   # Streamlit UI
├── configs/               # story_phases.yaml
└── README.md
```

## Core Concepts

### Context Unit
The smallest retrievable unit: **1–3 consecutive paragraphs within a Section** expressing a single narrative fact.

### Story Phases
Temporal markers for narrative reasoning:
- Origins
- Rise of the Pandavas
- Exile (with sub-phases: DiceGame, ForestYears, Agyatvasa)
- Prelude to War
- Kurukshetra War (Days grouped)
- Immediate Aftermath
- Post-War Instruction
- Withdrawal from the World

### SemRAG Pipeline

```
User Query
 → Entity Resolution (NER, aliases)
 → Story-Phase Filtering
 → Community Filtering
 → Graph Traversal
 → Context Unit Retrieval
 → Semantic Re-ranking
 → Answer Generation (LLM, constrained)
```

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### Phase 1: Ingestion
```bash
python src/ingestion/pdf_parser.py --pdf data/raw_pdf/mahabharata.pdf
```

### Phase 2: Structuring
```bash
python src/structuring/paragraph_splitter.py
python src/structuring/context_unit_builder.py
```

### Phase 3: Graph Construction
```bash
python src/graph/graph_builder.py
```

### Phase 4: Embeddings
```bash
python src/embeddings/openai_embedder.py
```

### Phase 5: Retrieval & Generation
```bash
streamlit run app/streamlit_app.py
```

## Output Format

```
Query: Where did Arjun go during the Agyatvasa?

Answer:
During the Agyatvasa, Arjuna went to the kingdom of Virata, where he lived
in disguise as a eunuch named Brihannala.

Retrieved from:
Parva: Virata Parva
Section: Section X

Contextual Passage:
"Arjuna, assuming the guise of a eunuch named Brihannala, entered the city
of Virata. Skilled in music and dance, he lived there unnoticed..."
```

## Evaluation

Answers are validated if:
- The quoted passage explicitly supports the answer
- Parva + Section match the passage source
- Removing the passage invalidates the answer

## Technologies

- **PDF Processing**: pdfplumber, PyMuPDF
- **NER**: spaCy
- **Knowledge Graph**: NetworkX
- **Community Detection**: Louvain/Leiden
- **Embeddings**: OpenAI text-embedding-3-large
- **LLM**: OpenAI GPT models
- **Vector Store**: FAISS
- **UI**: Streamlit

## License

As per KM Ganguly translation copyright and project licensing.
