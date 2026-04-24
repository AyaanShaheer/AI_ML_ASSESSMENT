import json
import csv
from pathlib import Path

from pipeline import LegalRAGPipeline

BASE_DIR = Path("/mnt/d/ai-ml-assessment/section_02_RAG")
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

QA_PAIRS = [
    {
        "id": 1,
        "question": "What is the notice period for termination in the NDA?",
        "expected_answer": "thirty (30) days written notice",
        "expected_doc": "Mutual_NDA.pdf",
        "expected_page": 2,
    },
    {
        "id": 2,
        "question": "What constitutes Confidential Information according to the agreement?",
        "expected_answer": "information or material that has or could have commercial value or other utility",
        "expected_doc": "NDA_Sample.pdf",
        "expected_page": 1,
    },
    {
        "id": 3,
        "question": "What are the obligations of the Receiving Party?",
        "expected_answer": "hold and maintain confidential information in strictest confidence",
        "expected_doc": "NDA_Sample.pdf",
        "expected_page": 1,
    },
    {
        "id": 4,
        "question": "What happens to confidential information after termination of the agreement?",
        "expected_answer": "obligations continue for two (2) years after termination",
        "expected_doc": "Mutual_NDA.pdf",
        "expected_page": 2,
    },
    {
        "id": 5,
        "question": "What is the governing law of the agreement?",
        "expected_answer": "laws of India",
        "expected_doc": "Govt_NDA.pdf",
        "expected_page": 5,
    },
    {
        "id": 6,
        "question": "What is the duration of the confidentiality obligation?",
        "expected_answer": "until the confidential information no longer qualifies as a trade secret or written release",
        "expected_doc": "NDA_Sample.pdf",
        "expected_page": 1,
    },
    {
        "id": 7,
        "question": "What must the Receiving Party do if the Disclosing Party requests return of materials?",
        "expected_answer": "return all records notes and tangible materials immediately",
        "expected_doc": "NDA_Sample.pdf",
        "expected_page": 1,
    },
    {
        "id": 8,
        "question": "What is the Need to Know clause in the NDA?",
        "expected_answer": "restrict disclosure to employees and consultants with a need to know",
        "expected_doc": "Govt_NDA.pdf",
        "expected_page": 5,
    },
    {
        "id": 9,
        "question": "Can the Receiving Party disclose confidential information to third parties?",
        "expected_answer": "not without prior written approval of the Disclosing Party",
        "expected_doc": "Mutual_NDA.pdf",
        "expected_page": 2,
    },
    {
        "id": 10,
        "question": "What happens if any provision of the agreement is held invalid or unenforceable?",
        "expected_answer": "the provision shall be modified to the extent necessary to make it valid and enforceable",
        "expected_doc": "Govt_NDA.pdf",
        "expected_page": 6,
    },
]


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def hit_at_3(result_sources, expected_doc, expected_page):
    for source in result_sources[:3]:
        if (
            source["document"] == expected_doc
            and int(source["page"]) == int(expected_page)
        ):
            return 1
    return 0


def main():
    pipeline = LegalRAGPipeline()

    rows = []
    hits = 0

    print("\nRunning evaluation on 10 QA pairs...\n")

    for qa in QA_PAIRS:
        result = pipeline.query(qa["question"])
        hit = hit_at_3(result["sources"], qa["expected_doc"], qa["expected_page"])
        hits += hit

        row = {
            "id": qa["id"],
            "question": qa["question"],
            "expected_doc": qa["expected_doc"],
            "expected_page": qa["expected_page"],
            "predicted_answer": result["answer"],
            "predicted_confidence": result["confidence"],
            "top1_doc": result["sources"][0]["document"] if len(result["sources"]) > 0 else "",
            "top1_page": result["sources"][0]["page"] if len(result["sources"]) > 0 else "",
            "top2_doc": result["sources"][1]["document"] if len(result["sources"]) > 1 else "",
            "top2_page": result["sources"][1]["page"] if len(result["sources"]) > 1 else "",
            "top3_doc": result["sources"][2]["document"] if len(result["sources"]) > 2 else "",
            "top3_page": result["sources"][2]["page"] if len(result["sources"]) > 2 else "",
            "hit@3": hit,
        }
        rows.append(row)

        print(f"Q{qa['id']}: hit@3 = {hit} | confidence = {result['confidence']}")
        print(f"  Question: {qa['question']}")
        print(f"  Expected: {qa['expected_doc']} page {qa['expected_page']}")
        for i, src in enumerate(result["sources"], start=1):
            print(f"  Top{i}: {src['document']} page {src['page']}")
        print()

    precision_at_3 = hits / len(QA_PAIRS)

    summary = {
        "total_questions": len(QA_PAIRS),
        "hits_at_3": hits,
        "precision_at_3": round(precision_at_3, 3),
    }

    csv_path = RESULTS_DIR / "evaluation_results.csv"
    json_path = RESULTS_DIR / "evaluation_summary.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print(f"Precision@3 = {hits}/{len(QA_PAIRS)} = {precision_at_3:.3f}")
    print(f"Saved CSV   : {csv_path}")
    print(f"Saved JSON  : {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()