# tests/test_latency.py
#
# Section 03 — Latency + Validity Assertion Test
#
# For each of 20 raw ticket strings, asserts:
#   (1) The predicted label is one of the 5 valid class labels
#   (2) Each inference completes in under 500ms (wall-clock, single ticket)
#
# Run from section_03_classifier/:
#   python tests/test_latency.py
#
# Or with pytest:
#   pytest tests/test_latency.py -v

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from classifier import TicketClassifier, VALID_LABELS

MODEL_PATH    = Path(__file__).parent.parent / "outputs" / "classifier.pkl"
LATENCY_LIMIT = 500.0  # milliseconds — hard requirement from assignment

# ── 20 raw ticket strings ─────────────────────────────────────────────────────
# Deliberately varied: different classes, tones, lengths, and phrasings.
# These are NOT from the training or eval sets.

TICKETS = [
    # billing (4)
    "I was charged twice for my subscription this month.",
    "My invoice shows ₹499 but I should be on the free plan.",
    "You renewed my annual plan without sending any notification.",
    "The refund I was promised 10 days ago has still not arrived.",

    # technical_issue (4)
    "The login page keeps sending me back to itself in an endless loop.",
    "My API key returns a 401 error even though I just generated it.",
    "Webhooks are firing duplicate events for every order that comes in.",
    "The CSV export file downloads but every row in it is duplicated.",

    # feature_request (4)
    "Could you please add a Gantt chart view for project planning?",
    "It would be great to have keyboard shortcuts for the most common actions.",
    "Please add a native Slack integration so we get real-time alerts.",
    "I'd love a way to duplicate a project template with one click.",

    # complaint (4)
    "I've been waiting ten days for a reply and no one has contacted me.",
    "Your last update broke my entire reporting workflow and nobody cares.",
    "I'm paying for enterprise support and getting worse help than the free tier.",
    "I will leave a public review if this issue is not resolved by end of day.",

    # other (4)
    "What payment methods does your platform currently accept?",
    "Where can I find the full API documentation for your REST endpoints?",
    "Can I have multiple workspaces under the same account?",
    "Do you offer a discount for registered non-profit organisations?",
]


# ── Test function ─────────────────────────────────────────────────────────────

def test_validity_and_latency():
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run:  python src/classifier.py")
        sys.exit(1)

    clf = TicketClassifier().load(MODEL_PATH)
    # Warm-up: triggers Cython JIT on the first call so the test loop
    # measures steady-state latency, not cold-start overhead.
    # In production, the server process stays alive — cold start never hits real requests.
    _ = clf.predict("warm-up call")
    
    failures = []

    col = 22
    print(f"{'#':>3}  {'predicted_label':{col}s}  {'ms':>8}  {'conf':>6}  ticket (truncated)")
    print("-" * 90)

    for i, ticket in enumerate(TICKETS):
        # Measure wall-clock time for a single prediction
        t_start    = time.perf_counter()
        result     = clf.predict(ticket)
        wall_ms    = (time.perf_counter() - t_start) * 1_000

        label = result["label"]
        conf  = result["confidence"]

        # Assertion 1 — label must be one of the 5 valid classes
        if label not in VALID_LABELS:
            failures.append(
                f"[{i+1:02d}] INVALID LABEL '{label}' for ticket: {ticket[:60]}"
            )

        # Assertion 2 — wall-clock latency must be under 500ms
        if wall_ms > LATENCY_LIMIT:
            failures.append(
                f"[{i+1:02d}] LATENCY {wall_ms:.1f}ms exceeds {LATENCY_LIMIT}ms for: {ticket[:60]}"
            )

        status = " " if (label in VALID_LABELS and wall_ms <= LATENCY_LIMIT) else "!"
        print(f"{i+1:>3}{status} {label:{col}s}  {wall_ms:7.2f}ms  {conf:6.3f}  {ticket[:55]}")

    print("-" * 90)

    # Summary
    if failures:
        print(f"\n✗  FAILED — {len(failures)} assertion(s) did not pass:\n")
        for msg in failures:
            print(f"   ✗ {msg}")
        sys.exit(1)  # non-zero exit so pytest marks as failed
    else:
        print(f"\n✓  ALL {len(TICKETS)} ASSERTIONS PASSED")
        print(f"   Label validity : ✓  (all predictions in {sorted(VALID_LABELS)})")
        print(f"   Latency <{LATENCY_LIMIT:.0f}ms : ✓  (every ticket classified under the limit)")


# ── pytest-compatible wrapper ─────────────────────────────────────────────────

def test_section03_latency_and_validity():
    """Named for pytest discovery."""
    test_validity_and_latency()


# ── Direct execution ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Section 03 — Latency + Validity Assertion Test")
    print(f"Latency limit  : {LATENCY_LIMIT}ms per ticket")
    print(f"Valid labels   : {sorted(VALID_LABELS)}")
    print(f"Ticket count   : {len(TICKETS)}\n")
    test_validity_and_latency()