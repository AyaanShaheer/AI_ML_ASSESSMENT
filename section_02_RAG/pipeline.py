"""
pipeline.py — Legal RAG Pipeline
Supports: pipeline.query(question) -> {answer, sources, confidence}
"""

import os
import json
import re
import pickle
import numpy as np
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = Path("/mnt/d/ai-ml-assessment/section_02_RAG")
VS_DIR           = BASE_DIR / "vectorstore"
FAISS_INDEX_PATH = VS_DIR / "index.faiss"
METADATA_PATH    = VS_DIR / "metadata.json"
BM25_CORPUS_PATH = VS_DIR / "bm25_corpus.pkl"

# ── Config ────────────────────────────────────────────────────────────────────
DENSE_TOP_K      = 20
BM25_TOP_K       = 20
RERANK_TOP_N     = 30
FINAL_TOP_K      = 3
RRF_K            = 60
MIN_CONFIDENCE   = 0.25
COVERAGE_THRESH  = 0.15

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

SYSTEM_PROMPT = """You are a precise legal document analyst.

STRICT RULES:
1. Answer ONLY using information explicitly stated in the provided context chunks.
2. If the answer is not present in the context, respond EXACTLY with:
   "This information is not available in the retrieved documents."
3. Do NOT infer, extrapolate, or use outside legal knowledge.
4. Always cite which document and section your answer comes from.
5. For numerical values (amounts, days, percentages), quote them exactly as written.
6. Keep answers concise and factual. No preamble."""


class LegalRAGPipeline:

    def __init__(self):
        print("Initializing Legal RAG Pipeline...")

        print("  [1/4] Loading BGE-large embedding model...")
        self.embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

        print("  [2/4] Loading cross-encoder reranker...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        print("  [3/4] Loading FAISS index...")
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        print("  [4/4] Loading metadata and BM25 index...")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        with open(BM25_CORPUS_PATH, "rb") as f:
            corpus = pickle.load(f)
        self.bm25 = BM25Okapi(corpus)

        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        print("Pipeline ready.\n")

    def _embed_query(self, question):
        text = BGE_QUERY_PREFIX + question
        vec = self.embed_model.encode([text], normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    def _dense_retrieve(self, query_vec, top_k):
        scores, indices = self.index.search(query_vec, top_k)
        return indices[0].tolist()

    def _bm25_retrieve(self, question, top_k):
        tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', question).lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return top_indices.tolist()

    def _rrf_fuse(self, dense_ids, bm25_ids):
        scores = {}
        for rank, idx in enumerate(dense_ids):
            scores[idx] = scores.get(idx, 0) + 1.0 / (rank + RRF_K)
        for rank, idx in enumerate(bm25_ids):
            scores[idx] = scores.get(idx, 0) + 1.0 / (rank + RRF_K)
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_ids[:RERANK_TOP_N]

    def _rerank(self, question, candidate_ids):
        valid_ids = [i for i in candidate_ids if i < len(self.metadata)]
        pairs = [(question, self.metadata[i]["chunk"]) for i in valid_ids]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:FINAL_TOP_K]

    def _context_coverage(self, question, chunks):
        q_tokens = set(re.sub(r'[^a-zA-Z0-9]', ' ', question).lower().split())
        c_tokens = set(re.sub(r'[^a-zA-Z0-9]', ' ', " ".join(chunks)).lower().split())
        if not q_tokens:
            return 0.0
        return len(q_tokens & c_tokens) / len(q_tokens)

    def _faithfulness_score(self, answer, chunks):
        if "not available in the retrieved documents" in answer.lower():
            return 1.0

        combined = " ".join(chunks).lower()

        numbers     = re.findall(r'\b\d+\b', answer)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer)
        quoted      = re.findall(r'"([^"]+)"', answer)

        claims = numbers + proper_nouns + quoted
        if not claims:
            return 0.8

        verified = sum(1 for c in claims if c.lower() in combined or c in combined)
        return verified / len(claims)

    def _compute_confidence(self, top_rerank_score, coverage, faithfulness):
        normalized_rerank = 1.0 / (1.0 + np.exp(-top_rerank_score * 0.3))
        confidence = (
            0.40 * normalized_rerank +
            0.30 * min(coverage, 1.0) +
            0.30 * faithfulness
        )
        return round(float(np.clip(confidence, 0.0, 1.0)), 3)

    def _generate(self, question, context_chunks_meta):
        context_text = ""
        for i, m in enumerate(context_chunks_meta):
            context_text += (
                f"\n--- Chunk {i+1} | "
                f"Document: {m['document']} | "
                f"Page: {m['page']} | "
                f"Section: {m['section']} ---\n"
                f"{m['chunk']}\n"
            )

        user_message = (
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            f"Answer (cite the document name and page number in your answer):"
        )

        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    def query(self, question):
        """
        Main query interface.

        Returns:
        {
            'answer'    : str,
            'sources'   : [{'document': str, 'page': int, 'chunk': str}],
            'confidence': float
        }
        """
        # Stage 1 — Dual retrieval
        query_vec = self._embed_query(question)
        dense_ids = self._dense_retrieve(query_vec, DENSE_TOP_K)
        bm25_ids  = self._bm25_retrieve(question, BM25_TOP_K)

        # Stage 2 — RRF fusion
        fused_ids = self._rrf_fuse(dense_ids, bm25_ids)

        # Stage 3 — Cross-encoder reranking
        reranked  = self._rerank(question, fused_ids)

        # Gather top chunk metadata
        top_chunks_meta  = [self.metadata[idx] for idx, _ in reranked if idx < len(self.metadata)]
        top_chunk_texts  = [m["chunk"] for m in top_chunks_meta]
        top_rerank_score = reranked[0][1] if reranked else 0.0

        # Hallucination Gate — coverage check
        coverage = self._context_coverage(question, top_chunk_texts)

        if coverage < COVERAGE_THRESH:
            return {
                "answer":     "This information is not available in the retrieved documents.",
                "sources":    [],
                "confidence": 0.0,
            }

        # Generation
        answer = self._generate(question, top_chunks_meta)

        # Force Confidence = 0.0 for explicit refusal
        if "not available in the retrieved documents" in answer.lower():
            return {
                "answer":     answer,
                "sources":    [],
                "confidence": 0.0,
            }

        # Post-generation faithfulness check
        faithfulness = self._faithfulness_score(answer, top_chunk_texts)

        # Confidence score
        confidence = self._compute_confidence(top_rerank_score, coverage, faithfulness)

        # Low-confidence warning
        if confidence < MIN_CONFIDENCE and "not available" not in answer.lower():
            answer += (
                "\n\nWarning: Low confidence score. "
                "Please verify this answer against the source document directly."
            )

        sources = [
            {
                "document": m["document"],
                "page":     m["page"],
                "chunk":    m["chunk"],
            }
            for m in top_chunks_meta
        ]

        return {
            "answer":     answer,
            "sources":    sources,
            "confidence": confidence,
        }


if __name__ == "__main__":
    pipeline = LegalRAGPipeline()

    test_questions = [
        "What is the notice period for termination in the NDA?",
        "What constitutes Confidential Information according to the agreement?",
        "What are the obligations of the Receiving Party?",
        "What is the governing law mentioned in the agreement?",
        "What happens to confidential information after termination of the agreement?",
    ]

    for q in test_questions:
        print("\n" + "=" * 65)
        print(f"QUESTION: {q}")
        result = pipeline.query(q)
        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nCONFIDENCE: {result['confidence']}")
        print(f"\nSOURCES ({len(result['sources'])}):")
        for s in result["sources"]:
            print(f"  Document : {s['document']}")
            print(f"  Page     : {s['page']}")
            print(f"  Excerpt  : {s['chunk'][:150].strip()}...")
            print()