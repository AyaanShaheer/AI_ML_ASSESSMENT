# Question C — On-Premise LLM Deployment

The requirement is a fully offline deployment for a defence/government client with:

* no external API calls
* 2× NVIDIA A100 80GB GPUs
* response time under 3 seconds
* 500-token input prompts

This is a production inference problem where latency, reliability, and data isolation matter more than maximum model size.

---

## Step 1 — Model Selection

I would evaluate these open-source models:

### Primary Candidates

* Llama 3 70B Instruct
* Mixtral 8x7B Instruct
* Qwen 2.5 72B Instruct
* Mistral Large (if licensing permits)

### Why Not Smaller 7B Models

7B models are faster, but for defence/government workflows:

* factual precision
* instruction reliability
* long-context reasoning

matter more than minimal latency.

70B-class models are a better production baseline.

My first choice would be:

# Llama 3 70B Instruct

because it offers:

* strong instruction following
* excellent enterprise adoption
* stable inference tooling
* reliable offline deployment support

---

## Step 2 — VRAM Calculation

### FP16 Memory Estimate

70B parameters × 2 bytes

≈ **140GB VRAM**

Plus:

* KV cache
* attention buffers
* runtime overhead

Real deployment requires:

### ~160–180GB VRAM

This exceeds a single GPU.

But with:

### 2 × A100 80GB

Total:

### 160GB VRAM

this becomes feasible using tensor parallelism.

---

## Step 3 — Quantisation Strategy

I would use:

# 4-bit AWQ or GPTQ quantisation

instead of full FP16.

Why:

* major VRAM reduction
* lower latency
* minimal accuracy loss for inference

Example:

70B at 4-bit

≈ **35–45GB**

This allows:

* faster inference
* larger KV cache
* higher throughput
* safer deployment margin

For production:

### AWQ is preferred

because it preserves instruction quality better than naive quantization.

---

## Step 4 — Serving Stack

I would choose:

# vLLM

for primary serving.

### Why vLLM

Because it provides:

* paged attention
* efficient KV cache management
* tensor parallelism
* strong throughput
* production-ready serving APIs

Better than:

### llama.cpp

which is excellent for smaller edge deployments but not ideal for A100 multi-GPU serving.

Better than:

### TensorRT-LLM

for this assignment because TensorRT offers maximum performance but significantly higher deployment complexity.

For a defence client:

### reliability + maintainability > maximum benchmark speed

So vLLM is the best balance.

---

## Step 5 — Expected Throughput

Requirement:

### under 3 seconds

for:

### 500-token input

Assume:

* 500 input tokens
* 150–250 output tokens

With:

### Llama 3 70B 4-bit on 2×A100

practical throughput is roughly:

### 80–150 tokens/sec

depending on concurrency and batching.

This gives:

### Time-to-first-token:

~0.5–1.0 sec

### Full response:

~2–3 seconds

which satisfies the requirement.

This is realistic production performance.

---

## Step 6 — Security + Offline Constraints

Because this is a defence environment:

I would also enforce:

* air-gapped deployment
* local model weights only
* local embedding + retrieval
* audit logging
* restricted admin access
* encrypted model storage
* offline package mirrors for updates

No telemetry or outbound network access should exist.

This is often more important than raw model speed.

---

## Limitations

This approach still has tradeoffs:

### Quantisation reduces some reasoning quality

especially for highly specialized domain tasks.

### Fine-tuning may still be needed

for:

* classified terminology
* domain-specific workflows
* military/legal internal procedures

### Large reranking pipelines may be required

if retrieval quality becomes the bottleneck instead of generation.

Model serving alone does not solve system quality.

---

## Final Architecture

```text
User Query
   ↓
Local Retrieval Layer (optional RAG)
   ↓
vLLM Server
   ↓
Llama 3 70B Instruct (4-bit AWQ)
   ↓
2× A100 80GB GPUs
   ↓
Offline Response (<3 sec target)
```

This balances:

* accuracy
* security
* latency
* maintainability

for real-world government deployment.

