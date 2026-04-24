from pipeline import LegalRAGPipeline

pipeline = LegalRAGPipeline()

# Questions the pipeline SHOULD answer
good_questions = [
    "What is the notice period for termination in the NDA?",
    "What are the obligations of the Receiving Party?",
]

# Questions the pipeline SHOULD REFUSE (not in any document)
refusal_questions = [
    "What is the limitation of liability clause above 1 crore rupees?",
    "What is the payment schedule for Vendor Acme Corp?",
    "What is the arbitration clause for disputes above 50 lakh?",
]

print("\n" + "="*60)
print("NORMAL QUERIES (should answer with citations)")
print("="*60)
for q in good_questions:
    result = pipeline.query(q)
    print(f"\nQ: {q}")
    print(f"A: {result['answer'][:200]}...")
    print(f"Confidence: {result['confidence']}")
    if result['sources']:
        print(f"Source: {result['sources'][0]['document']} | Page {result['sources'][0]['page']}")

print("\n" + "="*60)
print("REFUSAL QUERIES (should refuse — not in documents)")
print("="*60)
for q in refusal_questions:
    result = pipeline.query(q)
    print(f"\nQ: {q}")
    print(f"A: {result['answer']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Sources: {len(result['sources'])}")
