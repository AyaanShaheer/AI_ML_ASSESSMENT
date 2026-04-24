# src/evaluate.py
#
# Evaluation script for Section 03.
# Runs on the manually written held-out eval set (120 examples, 24 per class).
# Reports: overall accuracy, per-class F1, full confusion matrix, avg latency.
# Saves a plain-text results file to outputs/eval_results.txt.
#
# Run from section_03_classifier/:
#   python src/evaluate.py

import json
import sys
import time
from pathlib import Path

# Allow import of classifier from sibling src/ directory
sys.path.insert(0, str(Path(__file__).parent))

from classifier import TicketClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).parent.parent
EVAL_PATH  = ROOT / "data"    / "eval_data.json"
MODEL_PATH = ROOT / "outputs" / "classifier.pkl"
OUT_DIR    = ROOT / "outputs"

CLASSES = sorted(["billing", "technical_issue", "feature_request", "complaint", "other"])


# ── Helpers ────────────────────────────────────────────────────────────────────

def print_confusion_matrix(cm, labels: list[str]) -> str:
    """Render confusion matrix as a plain-text table (no pandas required)."""
    col_w = 16
    lines = []

    # Header row
    header = f"{'':20s}" + "".join(f"{l[:col_w]:>{col_w}}" for l in labels)
    lines.append(header)
    lines.append("-" * len(header))

    # Data rows
    for i, row_label in enumerate(labels):
        row = f"{row_label:20s}" + "".join(f"{val:>{col_w}}" for val in cm[i])
        lines.append(row)

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    # 1. Load eval set
    if not EVAL_PATH.exists():
        print(f"ERROR: eval_data.json not found at {EVAL_PATH}")
        print("Run:  python data/generate_eval.py")
        sys.exit(1)

    with open(EVAL_PATH) as f:
        raw = json.load(f)

    X_eval = [d["text"]  for d in raw]
    y_true = [d["label"] for d in raw]

    # 2. Load model
    if not MODEL_PATH.exists():
        print(f"ERROR: classifier.pkl not found at {MODEL_PATH}")
        print("Run:  python src/classifier.py")
        sys.exit(1)

    clf = TicketClassifier().load(MODEL_PATH)

    # 3. Run inference + measure total wall-clock time
    t_start  = time.perf_counter()
    results  = clf.predict_batch(X_eval)
    total_ms = (time.perf_counter() - t_start) * 1_000

    y_pred      = [r["label"]      for r in results]
    latencies   = [r["latency_ms"] for r in results]

    # 4. Compute metrics
    accuracy    = accuracy_score(y_true, y_pred)
    report_str  = classification_report(y_true, y_pred, target_names=CLASSES, digits=3)
    cm          = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cm_str      = print_confusion_matrix(cm, CLASSES)
    avg_lat_ms  = sum(latencies) / len(latencies)
    max_lat_ms  = max(latencies)

    # 5. Print to terminal
    sep = "=" * 65

    print(sep)
    print("SECTION 03 — CLASSIFIER EVALUATION RESULTS")
    print(sep)
    print(f"  Eval set        : {len(X_eval)} manually written examples ({len(X_eval)//len(CLASSES)} per class)")
    print(f"  Accuracy        : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  Avg latency     : {avg_lat_ms:.2f}ms per ticket")
    print(f"  Max latency     : {max_lat_ms:.2f}ms per ticket")
    print(f"  Total time      : {total_ms:.1f}ms for {len(X_eval)} tickets")
    print()
    print("Per-class Precision / Recall / F1:")
    print(report_str)
    print("Confusion Matrix  (rows = true label, cols = predicted label):")
    print(cm_str)

    # 6. Identify the two most confused class pairs
    print()
    print("Most confused class pairs:")
    off_diag = []
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i != j and cm[i][j] > 0:
                off_diag.append((cm[i][j], CLASSES[i], CLASSES[j]))
    off_diag.sort(reverse=True)
    for count, true_lbl, pred_lbl in off_diag[:4]:
        print(f"  {true_lbl:20s} → predicted as {pred_lbl:20s} : {count} time(s)")

    # 7. Save results to file
    OUT_DIR.mkdir(exist_ok=True)
    out_file = OUT_DIR / "eval_results.txt"

    with open(out_file, "w") as f:
        f.write("SECTION 03 — CLASSIFIER EVALUATION RESULTS\n")
        f.write(sep + "\n")
        f.write(f"Eval set    : {len(X_eval)} manually written examples\n")
        f.write(f"Accuracy    : {accuracy:.4f}  ({accuracy*100:.1f}%)\n")
        f.write(f"Avg latency : {avg_lat_ms:.2f}ms per ticket\n")
        f.write(f"Max latency : {max_lat_ms:.2f}ms per ticket\n\n")
        f.write("Classification Report:\n")
        f.write(report_str + "\n")
        f.write("Confusion Matrix (rows=true, cols=predicted):\n")
        f.write(cm_str + "\n")

    print(f"\nResults saved → {out_file}")


if __name__ == "__main__":
    run()