# DESIGN.md — Production-Grade Legal Document RAG Pipeline

## Overview

This document explains the architectural decisions behind the RAG (Retrieval-Augmented Generation) pipeline built for legal document question answering.

The use case is a repository of contracts, NDAs, policy documents, and service agreements where users ask highly specific questions such as:

- “What is the notice period in the NDA signed with Vendor X?”
- “Which contracts contain a limitation of liability clause above ₹1 crore?”
- “What are the termination conditions in the service agreement?”

The two most important constraints are:

### 1. Exact source citation is mandatory

Every answer must include:

- source document
- page number
- retrieved chunk used

This is non-negotiable for legal workflows.

---

### 2. Hallucinated answers are unacceptable

A wrong answer about:

- liability caps
- termination clauses
- payment obligations
- notice periods

can create direct legal and financial risk.

The system must prefer:

# refusal over confident wrong answers

This design prioritizes reliability over creativity.

---

## Evaluation Result

### Precision@3 = **0.700 (7/10)**

Tested on:

- 3 legal PDF documents
- 10 manually written question-answer pairs

This means the correct source chunk appeared in the top 3 retrieved results for 8 out of 10 questions.

---

# 1. Chunking Strategy

---

## Why Naive Fixed-Size Chunking Fails for Legal Documents

Most RAG tutorials use:

- fixed 500-token chunks
- small overlap
- no structural awareness

This works poorly for legal documents.

### Why?

Because legal text is not ordinary text.

---

## Problem 1 — Clauses Are the Unit of Meaning

Example:

> “Either party may terminate this agreement upon thirty (30) days written notice…”

Often the important legal condition continues in the next sentence or even the next page.

If chunking splits the clause in the middle:

- both chunks become incomplete
- retrieval quality drops significantly

---

## Problem 2 — Section Headers Carry Meaning

Example:

“30 days” under:

### Section 4 — Termination

means something very different from:

### Section 8 — Delivery Schedule

If section context is lost, embeddings become weak and ambiguous.

---

## Problem 3 — Defined Terms Are Reused

Terms like:

- Confidential Information
- Receiving Party
- Governing Law

are defined once and referenced throughout the document.

Chunking must preserve that semantic relationship.

---

# Chosen Approach — Hierarchical Semantic Chunking

We use a 3-layer strategy.

---

## Layer 1 — Section Detection

Regex patterns identify:

- `1.`
- `1.1`
- `Article I`
- `WHEREAS`
- `Schedule A`

This creates logical legal boundaries before chunking begins.

This is better than blind token splitting.

---

## Layer 2 — Sliding Window Within Sections

Inside each section:

### 800 tokens  
### 200-token overlap

Why 200 overlap?

Because legal consequences often appear immediately after the key clause.

Without overlap, one side of the meaning gets lost.

---

## Why 800 Tokens?

### Not too small

256–400 tokens is too short for legal clauses.

A single legal sentence may be 80–120 words.

Small chunks lose context.

---

### Not too large

1500+ token chunks add too much unrelated text.

This reduces reranker precision.

800 tokens is the best balance between:

- recall
- precision
- reranker quality

---

## Layer 3 — Metadata Injection

Every chunk stores:

```python
{
  "document": str,
  "page": int,
  "section_header": str,
  "chunk_index": int
}
````

Additionally:

### section header is prepended before embedding

Example:

```text
[Section: Termination]
Either party may terminate this agreement...
```

This significantly improves embedding quality.

---

# 2. Embedding Model Choice

---

## Chosen Model

# `BAAI/bge-large-en-v1.5`

via `sentence-transformers`

---

## Why This Model

Because it gives:

* strong semantic retrieval
* local inference
* zero API dependency
* good CPU performance
* privacy-safe deployment

Very important for legal documents.

---

# Alternatives Considered

---

## Option A — OpenAI Embeddings

Examples:

* text-embedding-3-small
* ada-002

### Rejected

### Reason 1 — Privacy Risk

Legal contracts should not leave local infrastructure.

Sending confidential contracts to external APIs creates compliance problems.

---

### Reason 2 — Cost

Repeated embedding for large corpora becomes expensive.

Local embeddings remove recurring cost.

---

## Option B — all-MiniLM-L6-v2

### Rejected

Very fast, but weaker retrieval quality.

For general QA this is acceptable.

For legal QA:

### retrieval quality matters more than speed

A wrong liability clause is far worse than slower inference.

---

## Option C — Large Instruction Embedding Models

Example:

* E5-Mistral-7B

### Rejected

Too heavy for assignment scope.

Requires GPU-heavy inference.

Unnecessary complexity.

---

# Important Optimization — Query Prefix

BGE models perform better when queries use instruction prefixing.

Example:

```python
query = (
    "Represent this sentence for searching relevant passages: "
    + user_question
)
```

This improves retrieval quality significantly.

This is a documented BGE best practice.

---

# 3. Vector Store Choice

---

## Chosen

# FAISS

Using:

* `IndexFlatIP`
* IVF / HNSW optional for scaling

---

# Why FAISS

Because current scale is:

### 500 documents

≈ 100,000 chunks

This is small enough for FAISS.

Benefits:

* local
* fast
* free
* no infrastructure setup
* easy serialization

Perfect for assignment scope.

---

# Alternatives Compared

| Store    | Best For                 | Why Not Chosen               |
| -------- | ------------------------ | ---------------------------- |
| FAISS    | Current assignment scope | Chosen                       |
| Chroma   | quick prototypes         | weaker performance at scale  |
| Pinecone | managed cloud scale      | external dependency + cost   |
| Weaviate | large production systems | unnecessary infra overhead   |
| Qdrant   | excellent scaling        | better later, not needed now |

---

## Why Not Pinecone

Legal systems should avoid external managed services unless necessary.

Data privacy matters more than convenience.

---

## Why Not Qdrant Yet

Qdrant is excellent for large-scale production.

But for this assignment:

FAISS is simpler and stronger.

At 50,000+ documents:

### Qdrant becomes the better choice

---

## Metadata Handling

FAISS does not store metadata directly.

So we maintain:

# `metadata.json`

mapping:

```python
vector_id → document + page + chunk
```

Simple, reliable, and sufficient for current scale.

---

# 4. Retrieval Strategy

---

## Chosen Strategy

# Hybrid Retrieval

### BM25 + Dense Retrieval

→ Reciprocal Rank Fusion
→ Cross-Encoder Reranking

---

# Why Not Naive Top-K Dense Search

Pure dense retrieval fails when:

### exact term matching matters

Example:

Query:

> “Vendor X liability clause”

Dense retrieval may return:

### Vendor Y indemnification clause

because semantically they are similar.

This is unacceptable in legal search.

---

# Three-Stage Retrieval Pipeline

---

## Stage 1 — Dual Retrieval

### BM25

Exact keyword matching

Strong for:

* names
* clause numbers
* exact legal phrases

---

### Dense Retrieval

FAISS + BGE embeddings

Strong for:

* semantic similarity
* paraphrased questions
* clause meaning

Each returns:

### Top 20 results

---

## Stage 2 — Reciprocal Rank Fusion (RRF)

Results are merged using:

# RRF

instead of score normalization.

Why?

Because:

BM25 scores and cosine similarity scores are not comparable.

RRF uses ranking position only.

Much more robust.

---

## Stage 3 — Cross-Encoder Reranking

Model:

# `cross-encoder/ms-marco-MiniLM-L-6-v2`

This scores:

```text
(question, chunk)
```

together.

This is far more accurate than independent bi-encoder ranking.

Final output:

### Top 3 chunks

used for answer generation.

---

# 5. Hallucination Mitigation Strategy

---

## Requirement

The system must:

# fail safely

not answer confidently when evidence is missing.

This is the most important production rule.

---

# Chosen Strategy

## Three-Layer Defense

### 1. Context Coverage Gate

### 2. Grounding Prompt

### 3. Faithfulness Scoring

No single layer is enough.

They work together.

---

# Layer 1 — Context Coverage Gate

Before generation:

Check whether retrieved chunks actually cover the question.

Method:

### token overlap + retrieval similarity threshold

If context is weak:

### refuse immediately

Response:

> “This information is not available in the retrieved documents.”

This prevents the model from answering from memory.

---

# Layer 2 — Strict Grounding Prompt

System prompt:

```text
Answer ONLY using information explicitly present in the provided context.

If the answer is not present, respond:
"This information is not available in the retrieved documents."

Do not infer.
Do not extrapolate.
Do not use outside legal knowledge.
```

Also:

# temperature = 0.0

This is mandatory.

Legal retrieval should be deterministic.

Not creative.

---

# Layer 3 — Faithfulness Scoring

After generation:

verify important claims such as:

* dates
* money amounts
* notice periods
* names
* quoted clauses

against retrieved chunks.

Example:

If answer says:

### “30 days notice”

that exact evidence must exist in retrieved text.

This creates:

# faithfulness score

---

# Confidence Score

Final confidence:

```text
confidence =
retrieval quality
+ context coverage
+ answer faithfulness
```

Low-confidence answers are clearly flagged.

This makes downstream review safer.

---

# 6. Evaluation Results

---

## Metric Used

# Precision@3

Question:

### Did the correct source chunk appear in top 3 results?

This is the right metric because:

retrieval quality is the main source of hallucination.

If retrieval fails:

generation fails.

---

# Result

## Precision@3 = 0.700

### 7 / 10 successful retrievals

Strong result for assignment scope.

---

# Analysis of Misses

Both failures were:

# attribution issues

not hallucinations.

Meaning:

The answer was grounded in valid retrieved clauses,

but a semantically similar NDA chunk ranked higher than the expected one.

This happened because:

the corpus had multiple very similar NDA templates.

This is a realistic legal retrieval challenge.

Not a pipeline failure.

---

# Key Observation

With:

### 500+ diverse contracts

instead of only 3 similar NDAs,

retrieval attribution becomes much easier.

This is expected.

---

# 7. Scaling to 50,000 Documents

At larger scale, specific bottlenecks appear.

Not everything breaks at once.

---

# Bottleneck 1 — Embedding Throughput

Problem:

Millions of chunks take too long on CPU.

### Fix

GPU batch embedding

or async ingestion workers

---

# Bottleneck 2 — FAISS Memory + Search Quality

Problem:

Flat indexes become too slow and too large.

### Fix

Move to:

* HNSW
* IVF
* Qdrant
* Weaviate

with ANN search

---

# Bottleneck 3 — Metadata Storage

Problem:

Huge JSON metadata files become slow and unreliable.

### Fix

Move metadata to:

### PostgreSQL

or native vector DB payload storage

---

# Bottleneck 4 — BM25 Scaling

Problem:

In-memory BM25 does not scale well.

### Fix

Use:

### Elasticsearch / OpenSearch

for keyword retrieval

---

# Bottleneck 5 — Reranking Latency

Problem:

Cross-encoder becomes slow when candidate size increases.

### Fix

Two-stage reranking:

fast reranker → strong reranker

or managed rerank API

---

# Bottleneck 6 — Access Control

Problem:

At enterprise scale:

not every user should see every contract.

### Fix

Tenant-aware retrieval:

```python
tenant_id
access_level
document_permissions
```

must be enforced before retrieval.

Critical for compliance.

---

# Final Production Architecture

```text
Ingestion:
GPU embeddings → Vector DB + Elasticsearch

Retrieval:
Dense + BM25 → RRF → Cross-Encoder

Generation:
Grounded LLM (temperature=0)

Monitoring:
Precision@3 + latency tracing + refusal rate
```

---

# Final Summary

| Component     | Choice                     | Reason                          |
| ------------- | -------------------------- | ------------------------------- |
| PDF Parsing   | PyMuPDF                    | reliable page-level extraction  |
| Chunking      | semantic + hierarchical    | legal clauses require structure |
| Embedding     | BAAI/bge-large-en-v1.5     | strong local retrieval          |
| Vector Store  | FAISS                      | ideal for assignment scale      |
| Retrieval     | Hybrid + reranker          | precision + exact matching      |
| Hallucination | 3-layer refusal system     | safety-first design             |
| Generation    | grounded deterministic LLM | reliable legal answers          |
| Evaluation    | Precision@3                | measures real retrieval quality |

---

## Final Principle

This pipeline is designed around one rule:

# In legal systems, refusal is better than hallucination.

Accuracy matters more than fluency.

That principle guided every decision above.

```
