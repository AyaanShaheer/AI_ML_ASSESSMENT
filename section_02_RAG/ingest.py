"""
ingest.py — Legal Document Ingestion Pipeline
Parses PDFs → chunks → embeds → stores in FAISS
"""

import os
import json
import re
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

import fitz  # pymupdf
from sentence_transformers import SentenceTransformer
import faiss

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/mnt/d/ai-ml-assessment/section_02_RAG")
PDF_DIR     = BASE_DIR / "data" / "pdfs"
VS_DIR      = BASE_DIR / "vectorstore"
VS_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_PATH = VS_DIR / "index.faiss"
METADATA_PATH    = VS_DIR / "metadata.json"
BM25_CORPUS_PATH = VS_DIR / "bm25_corpus.pkl"

# ── Chunking Config ───────────────────────────────────────────────────────────
CHUNK_SIZE    = 800   # tokens (approx chars / 4)
CHUNK_OVERLAP = 200
CHUNK_CHARS   = CHUNK_SIZE * 4        # ~3200 chars
OVERLAP_CHARS = CHUNK_OVERLAP * 4     # ~800 chars

# Legal section header patterns
SECTION_PATTERN = re.compile(
    r'^\s*('
    r'\d+[\.\)]\s+[A-Z]'           # "1. DEFINITIONS"
    r'|\d+\.\d+[\.\)]?\s+[A-Z]'   # "1.1 Confidential"
    r'|Article\s+[IVX\d]+'         # "Article IV"
    r'|WHEREAS'
    r'|Schedule\s+[A-Z\d]'
    r'|Exhibit\s+[A-Z\d]'
    r'|SECTION\s+\d+'
    r')',
    re.MULTILINE
)

# ── Embedding Model ───────────────────────────────────────────────────────────
print("Loading embedding model (BGE-large)...")
EMBED_MODEL = SentenceTransformer("BAAI/bge-large-en-v1.5")
EMBED_DIM   = 1024  # BGE-large output dimension


def extract_pages(pdf_path: Path) -> list[dict]:
    """Extract text per page with page numbers."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:
            pages.append({
                "page": page_num + 1,
                "text": text
            })
    doc.close()
    return pages


def detect_section_header(text: str) -> str | None:
    """Return the first section header found in text, or None."""
    match = SECTION_PATTERN.search(text)
    if match:
        # grab up to 80 chars from that point as the header
        start = match.start()
        header_line = text[start:start+80].split('\n')[0].strip()
        return header_line[:80]
    return None


def chunk_document(pages: list[dict], doc_name: str) -> list[dict]:
    """
    Hierarchical chunking:
    1. Detect section headers as semantic anchors
    2. Sliding window within sections (CHUNK_CHARS / OVERLAP_CHARS)
    3. Prepend section header to each chunk for embedding
    """
    chunks = []
    current_section = "Preamble"

    # Concatenate all pages, tracking page boundaries
    # Build a list of (char_offset, page_number) for lookup
    full_text = ""
    page_offsets = []  # (start_char, page_number)

    for p in pages:
        page_offsets.append((len(full_text), p["page"]))
        full_text += p["text"] + "\n\n"

    def get_page_for_offset(offset: int) -> int:
        page = 1
        for start, pnum in page_offsets:
            if offset >= start:
                page = pnum
            else:
                break
        return page

    # Slide through the full text
    pos = 0
    chunk_idx = 0

    while pos < len(full_text):
        end = min(pos + CHUNK_CHARS, len(full_text))
        chunk_text = full_text[pos:end]

        # Update section header if detected in this window
        detected = detect_section_header(chunk_text)
        if detected:
            current_section = detected

        # Determine page number (use the page of the start offset)
        page_num = get_page_for_offset(pos)

        # Build the enriched chunk text (header prepended as context)
        enriched = f"[Section: {current_section}]\n{chunk_text.strip()}"

        chunks.append({
            "document": doc_name,
            "page": page_num,
            "section": current_section,
            "chunk_index": chunk_idx,
            "chunk": chunk_text.strip(),
            "enriched_chunk": enriched,
        })

        chunk_idx += 1
        # Advance by (CHUNK_CHARS - OVERLAP_CHARS) for sliding window
        pos += (CHUNK_CHARS - OVERLAP_CHARS)

    return chunks


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """
    Embed enriched chunks using BGE-large.
    BGE docs: prepend instruction for document side = empty string.
    """
    texts = [c["enriched_chunk"] for c in chunks]
    embeddings = EMBED_MODEL.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity = dot product
    )
    return np.array(embeddings, dtype=np.float32)


def build_bm25_corpus(chunks: list[dict]) -> list[list[str]]:
    """Tokenize chunks for BM25 (simple whitespace + lowercase)."""
    corpus = []
    for c in chunks:
        tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', c["chunk"]).lower().split()
        corpus.append(tokens)
    return corpus


def ingest():
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}")
        return

    print(f"\nFound {len(pdf_files)} PDF(s): {[f.name for f in pdf_files]}")

    all_chunks = []

    for pdf_path in pdf_files:
        print(f"\n── Processing: {pdf_path.name}")
        pages = extract_pages(pdf_path)
        print(f"   Pages extracted: {len(pages)}")

        chunks = chunk_document(pages, pdf_path.name)
        print(f"   Chunks created:  {len(chunks)}")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all docs: {len(all_chunks)}")

    # ── Embed ─────────────────────────────────────────────────────────────────
    print("\nEmbedding chunks...")
    embeddings = embed_chunks(all_chunks)
    print(f"Embeddings shape: {embeddings.shape}")

    # ── Build FAISS Index (IVF-Flat) ──────────────────────────────────────────
    n_vectors = len(all_chunks)
    nlist = max(1, min(100, n_vectors // 10))  # adaptive nlist

    print(f"\nBuilding FAISS IVF-Flat index (nlist={nlist})...")
    quantizer = faiss.IndexFlatIP(EMBED_DIM)  # Inner Product (cosine on normalized vecs)

    if n_vectors >= nlist * 39:  # FAISS requires >= 39 * nlist training points
        index = faiss.IndexIVFFlat(quantizer, EMBED_DIM, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
    else:
        # Fallback to flat index for small corpora
        print("   (Small corpus: using IndexFlatIP)")
        index = faiss.IndexFlatIP(EMBED_DIM)

    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"FAISS index saved → {FAISS_INDEX_PATH}  ({index.ntotal} vectors)")

    # ── Save Metadata ─────────────────────────────────────────────────────────
    metadata = []
    for i, c in enumerate(all_chunks):
        metadata.append({
            "id": i,
            "document": c["document"],
            "page": c["page"],
            "section": c["section"],
            "chunk": c["chunk"],
        })

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved    → {METADATA_PATH}")

    # ── Save BM25 Corpus ──────────────────────────────────────────────────────
    bm25_corpus = build_bm25_corpus(all_chunks)
    with open(BM25_CORPUS_PATH, "wb") as f:
        pickle.dump(bm25_corpus, f)
    print(f"BM25 corpus saved → {BM25_CORPUS_PATH}")

    print("\n✅ Ingestion complete!")
    print(f"   Documents : {len(pdf_files)}")
    print(f"   Chunks    : {len(all_chunks)}")
    print(f"   Dimensions: {EMBED_DIM}")


if __name__ == "__main__":
    ingest()