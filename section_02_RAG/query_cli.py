"""
query_cli.py — Live interactive query terminal for the Legal RAG Pipeline
Run: python3 query_cli.py
"""

from pipeline import LegalRAGPipeline

def main():
    print("\n" + "="*65)
    print("  LEGAL DOCUMENT RAG PIPELINE — Interactive Query Terminal")
    print("="*65)
    print("  Documents loaded: Govt_NDA.pdf | Mutual_NDA.pdf | NDA_Sample.pdf")
    print("  Type your question and press Enter. Type 'exit' to quit.")
    print("="*65 + "\n")

    pipeline = LegalRAGPipeline()

    while True:
        try:
            question = input("Ask a question > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        print("\nSearching...\n")
        result = pipeline.query(question)

        print(f"ANSWER:")
        print(f"  {result['answer']}")
        print(f"\nCONFIDENCE: {result['confidence']}")

        if result['sources']:
            print(f"\nSOURCES:")
            for i, src in enumerate(result['sources'], 1):
                print(f"  [{i}] {src['document']}  |  Page {src['page']}")
                print(f"      \"{src['chunk'][:180].strip()}...\"")
        else:
            print("\nSOURCES: None — query was refused due to insufficient context.")

        print("\n" + "-"*65 + "\n")

if __name__ == "__main__":
    main()
