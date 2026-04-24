# Production-Grade Legal Document RAG Pipeline

> Artikate Studio · AI / ML / LLM Engineer Assessment

---

## Live System Walkthrough (Optional Bonus)

**Loom / screen recording link:**  
`[Link For Demonstration on Loom](https://www.loom.com/share/40bdf0fc4fa14061a465824497218b24)]`

This recording can show:
- A live query from a fresh terminal
- Retrieved chunks alongside the generated answer and source citations
- The evaluation harness computing Precision@3
- A refusal case for insufficient context
- Brief narration of design choices

---

## Overview

This project implements a production-grade Retrieval-Augmented Generation (RAG) pipeline for legal documents.

The system answers precise questions over PDF contracts / NDAs and returns:
- a generated answer,
- exact source citations with document name and page number,
- and a confidence score.

Hallucinated answers are mitigated using:
1. a context coverage gate,
2. a strict grounding prompt,
3. and post-generation faithfulness scoring.

**Evaluation result:** `Precision@3 = 0.800 (8/10)`

---

## Repository Structure

```text
section_02_RAG/
├── DESIGN.md              # Architecture decisions, trade-offs, scaling answer, evaluation
├── README.md              # Project overview and usage
├── .gitignore
├── ingest.py              # PDF ingestion → chunking → embedding → FAISS index
├── pipeline.py            # Core RAG pipeline with pipeline.query(...)
├── evaluate.py            # Evaluation harness with 10 QA pairs
├── query_cli.py           # Interactive live query terminal
├── test_refusal.py        # Refusal / insufficient-context demonstration
├── .env                   # GROQ_API_KEY (not committed)
├── data/
│   └── pdfs/              # Sample legal PDFs used for testing
├── vectorstore/           # Generated FAISS index + metadata
└── results/               # Generated evaluation outputs
```

---

## Deliverables Covered

This repository includes all required deliverables for Section 02:

- `DESIGN.md` covering:
  - chunking strategy and rationale,
  - embedding model choice and rationale,
  - vector store choice and rationale,
  - retrieval strategy and rationale,
  - hallucination mitigation strategy,
  - scaling changes for 50,000 documents,
  - evaluation result and failure analysis.

- Full working pipeline:
  - document ingestion
  - chunking
  - embedding
  - vector store
  - retrieval
  - generation with source citation

- Self-contained evaluation harness:
  - 10 manually written QA pairs
  - Precision@3 computation
  - generated CSV and JSON results

---

## Setup

### 1) Create and activate environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install pymupdf sentence-transformers faiss-cpu groq rank_bm25 python-dotenv numpy tqdm
```

### 3) Add Groq API key

Create a `.env` file in the project root:

```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

### 4) Add PDF documents

Place your sample PDF files in:

```bash
data/pdfs/
```

The current implementation was tested on 3 NDA-related PDF documents.

---

## Running the Project

### Step 1 — Ingest documents

This parses PDFs, creates legal-aware chunks, embeds them, and writes the FAISS index plus metadata.

```bash
python3 ingest.py
```

Expected outcome:

```text
✅ Ingestion complete!
Documents : 3
Chunks    : 13
Dimensions: 1024
```

### Step 2 — Run an interactive live query

This is the best way to demo the system from a fresh terminal.

```bash
python3 query_cli.py
```

Example question:

```text
Ask a question > What is the notice period for termination in the NDA?
```

### Step 3 — Run the pipeline programmatically

```python
from pipeline import LegalRAGPipeline

pipeline = LegalRAGPipeline()

result = pipeline.query(
    question="What is the notice period in the NDA with Vendor X?"
)

print(result)
```

### Minimum required interface

```python
{
    "answer": str,
    "sources": [
        {
            "document": str,
            "page": int,
            "chunk": str
        }
    ],
    "confidence": float
}
```

---

## Evaluation

Run the evaluation harness:

```bash
python3 evaluate.py
```

This runs 10 manually written question–answer pairs and computes Precision@3.

Expected output:

```text
Precision@3 = 8/10 = 0.800
Saved CSV   : results/evaluation_results.csv
Saved JSON  : results/evaluation_summary.json
```

### Metric used

**Precision@3** checks whether the correct source document and page appear in the top 3 retrieved chunks for each question. This is a standard retrieval-focused way to evaluate whether the evidence needed for grounded generation is surfaced early enough. [web:6][web:31]

---

## Refusal / Hallucination Demo

To show the system refusing unsupported questions:

```bash
python3 test_refusal.py
```

Example refusal-style query:

```text
What is the limitation of liability clause above 1 crore rupees for Vendor Acme Corp?
```

Expected behavior:
- answer refusal,
- confidence near 0.0,
- empty or insufficient sources.

This demonstrates that the pipeline prefers refusal over unsupported generation, which is critical in legal QA. [web:16][web:18]

---

## Architecture Summary

| Component | Choice | Reason |
|---|---|---|
| PDF Parsing | PyMuPDF (`fitz`) | Page-accurate extraction, fast, good for legal PDFs |
| Chunking | Hierarchical sliding window (800 tokens, 200 overlap) | Preserves clause context and section continuity |
| Embedding | `BAAI/bge-large-en-v1.5` | Strong retrieval quality, local/private, good trade-off |
| Vector Store | FAISS | Fast, local, zero-cost, sufficient at this scale |
| Retrieval | BM25 + Dense + Cross-Encoder reranking | Combines exact term match with semantic retrieval |
| Hallucination Control | Coverage gate + grounding prompt + faithfulness scoring | Prevents unsupported answers |
| LLM | Groq Llama 3.x | Fast generation with deterministic settings |

All detailed trade-off reasoning is documented in `DESIGN.md`.

---

## Notes

- The current test corpus is intentionally small and semantically homogeneous (3 NDA-style documents), which creates some retrieval ambiguity across near-duplicate clauses.
- The two evaluation misses were attribution-style misses, not pure hallucination failures.
- The scaling plan for 50,000 documents is included in `DESIGN.md`.

---

## Quick Demo Commands

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Ingest documents
python3 ingest.py

# 3. Ask live questions
python3 query_cli.py

# 4. Run evaluation
python3 evaluate.py

# 5. Test refusal
python3 test_refusal.py
```

---
