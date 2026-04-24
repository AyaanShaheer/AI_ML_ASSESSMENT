# Section 01 — Diagnose a Failing LLM Pipeline

## System
GPT-4o-powered customer support chatbot

## Reported Issues
1. Incorrect product pricing answers
2. Replies switching to English for Hindi/Arabic users
3. Response latency increased from 1.2s to 8–12s over two weeks

---

# Problem 1 — Incorrect Product Pricing

## Symptom

The chatbot confidently returns incorrect product prices even though no prompt or model changes were made after launch.

---

## Investigation Process

### Step 1 — Check Prompt and Context

I first checked whether current pricing information was being passed into the prompt or retrieval context.

Result:
No pricing information was present in the system prompt or request context.

The model was answering pricing questions without access to any live pricing source.

---

### Step 2 — Check Temperature

Temperature was set to 0.7.

I tested repeated pricing queries for the same product and found that the same incorrect price appeared consistently.

This rules out temperature as the main cause because temperature affects randomness, not repeated deterministic wrong answers.

**Ruled out**

---

### Step 3 — Check Prompt Regression

No prompt changes were made after deployment.

**Ruled out**

---

### Step 4 — Check Knowledge Cutoff

GPT-4o cannot know current pricing unless pricing is retrieved at runtime.

Product pricing changes frequently, so relying on model memory causes stale or fabricated answers.

There was no retrieval layer (RAG) connected to pricing documents.

---

## Root Cause

### Retrieval failure

This is primarily a retrieval issue.

The model had no access to current pricing data and generated plausible but incorrect answers from training knowledge.

This is not mainly a prompt issue or a temperature issue.

The absence of retrieval is the core problem.

---

## Fix

### Add RAG for pricing queries

Before answering pricing questions:

- retrieve pricing information from the latest pricing source
- restrict answers to retrieved context only
- refuse to answer if pricing is not found

Example system rule:

> Use only retrieved pricing context.
> If no verified pricing exists, respond:
> “I do not have confirmed pricing for that product right now.”

### Secondary safeguard

Lower temperature for pricing-related queries (`temperature=0.1`) to reduce confident guessing.

---

# Problem 2 — Language Switching (Hindi / Arabic → English)

## Symptom

Users write in Hindi or Arabic, but the assistant sometimes replies in English.

Testing originally happened only in English.

---

## Investigation Process

### Step 1 — Check Model Capability

GPT-4o supports Hindi and Arabic well.

This is not a multilingual capability problem.

**Ruled out**

---

### Step 2 — Check API Configuration

There is no API parameter that forces response language.

Language is inferred from prompt structure.

**Ruled out**

---

### Step 3 — Review System Prompt Architecture

The system prompt was long and written entirely in English.

The user message was often short (for example 1–2 Hindi sentences).

In this setup, the system prompt dominates behavior.

Without an explicit language rule, the model defaults to English because English is the strongest context signal.

---

## Root Cause

### System prompt language dominance

The assistant followed the language of the system prompt instead of the user message.

This is a prompt architecture issue.

Not a model failure.

---

## Fix

Add this rule at the top of the system prompt:

> Always respond in the same language as the user’s most recent message unless the user explicitly requests another language.
>
> Do not translate unless asked.
>
> Preserve the same script (Hindi, Arabic, etc.).

This must be placed at the beginning of the system prompt so it has highest priority.

### Optional safeguard

Use automatic language detection (`langdetect`) before sending the request and inject the detected language into the system prompt.

This makes the behavior more reliable.

---

# Problem 3 — Latency Increased from 1.2s to 8–12s

## Symptom

Latency gradually increased over two weeks without any code changes.

This gradual pattern suggests scaling problems rather than a sudden regression.

---

## Investigation Process

### First Step — Add Tracing

Before guessing causes, I would break latency into:

- embedding time
- retrieval time
- LLM generation time
- post-processing time

This identifies which component slowed down.

Tracing should always happen before optimisation.

---

## Possible Root Causes

---

### Cause A — Vector Store Growth

If retrieval uses a flat FAISS index (`IndexFlatL2`), search time grows linearly with document count.

As more users and documents are added, retrieval becomes slower.

This is the first issue I would investigate because it matches gradual degradation very closely.

### Fix

Move to approximate nearest neighbor search:

- FAISS IVF / HNSW
- or managed vector DB like Pinecone / Qdrant / Weaviate

---

### Cause B — Context Window Growth

If old conversation history is continuously appended, prompts become much larger over time.

More tokens = slower generation.

### Fix

Use:

- sliding window history
- summarisation of older conversations
- hard token limits

---

### Cause C — Missing Embedding Cache

If embeddings are recomputed for repeated queries without caching, latency increases significantly under higher traffic.

### Fix

Use Redis or in-memory caching for repeated embedding requests.

---

### Cause D — API Rate Limiting / Queueing

As user volume increases, OpenAI API requests may hit rate limits.

Requests then queue or retry, creating major delay without code changes.

### Fix

- concurrency control
- retry with backoff
- provider tier upgrade

---

## Priority

I would investigate vector store growth first because it best matches a gradual performance decline over time as the indexed data grows.

---

# Post-Mortem Summary (Non-Technical)

We identified three separate issues affecting the chatbot after launch.

First, incorrect pricing happened because the chatbot had no live connection to our pricing system. It was generating answers from older training knowledge instead of verified pricing data. We are fixing this by connecting the chatbot to a live pricing source and making it refuse pricing questions when verified data is missing.

Second, users writing in Hindi or Arabic were sometimes receiving English replies. This happened because the chatbot’s internal instructions were written in English and were stronger than the user’s language signal. We updated the system prompt so the assistant always responds in the same language as the user.

Third, response times increased because the retrieval system slowed down as more documents and users were added. We are improving this by upgrading the search index, reducing unnecessary conversation history, and adding caching.

All three issues were caused by system design and infrastructure, not by the AI model itself.