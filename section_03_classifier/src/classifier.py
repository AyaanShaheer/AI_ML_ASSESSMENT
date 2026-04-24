# src/classifier.py
#
# TF-IDF (1-3 ngrams) + Logistic Regression classifier
# for customer support ticket categorisation into 5 classes:
#   billing | technical_issue | feature_request | complaint | other
#
# ── Model choice justification (full version in ANSWERS.md) ──────────────────
#
# Constraint: <500ms per ticket on a single CPU server.
#
# DistilBERT on CPU:    150–400ms per ticket → borderline, no headroom for burst
# GPT-4o-mini via API:  800ms–3s per ticket  → hard fails the constraint
# TF-IDF + LR:          1–5ms per ticket     → 100x under the limit
#
# With only 1,000 training examples, fine-tuned DistilBERT does not
# meaningfully outperform a well-tuned LR. Transformers show their accuracy
# advantage with 10k+ labelled examples. For this dataset size and these
# latency constraints, LR is the correct production choice.
#
# Throughput: 2,880 tickets/day = 0.033/sec required.
#             LR delivers ~200/sec → 6,000× headroom.

import json
import pickle
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ── Constants ─────────────────────────────────────────────────────────────────

VALID_LABELS = frozenset({"billing", "technical_issue", "feature_request", "complaint", "other"})

ROOT       = Path(__file__).parent.parent
DATA_PATH  = ROOT / "data"  / "train_data.json"
MODEL_PATH = ROOT / "outputs" / "classifier.pkl"


# ── Classifier ────────────────────────────────────────────────────────────────

class TicketClassifier:
    """
    Wraps a sklearn Pipeline (TF-IDF → LogisticRegression).
    Public API:
        train()        — fit on training data and return self
        predict(text)  — single ticket → {label, confidence, latency_ms}
        save(path)     — serialise pipeline to disk
        load(path)     — deserialise pipeline from disk and return self
    """

    def __init__(self):
        self.pipeline: Pipeline | None = None

    # ── Internal build ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_pipeline() -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 3),      # unigrams, bigrams, trigrams
                max_features=20_000,     # cap vocabulary for memory efficiency
                sublinear_tf=True,       # log(1 + tf) to dampen high-freq terms
                min_df=2,                # drop terms in only one document
                strip_accents="unicode",
                analyzer="word",
            )),
            ("clf", LogisticRegression(
                C=5.0,                   # moderate regularisation
                max_iter=1_000,
                solver="lbfgs",
                random_state=42,
            )),
        ])

    # ── Train ──────────────────────────────────────────────────────────────────

    def train(self, data_path: Path = DATA_PATH) -> "TicketClassifier":
        """Load training data, fit pipeline, return self."""
        with open(data_path) as f:
            data = json.load(f)

        X = [d["text"]  for d in data]
        y = [d["label"] for d in data]

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X, y)
        return self

    # ── Predict ────────────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Classify a single ticket string.
        Returns:
            label       (str)   — one of the 5 valid classes
            confidence  (float) — probability of the predicted class (0–1)
            latency_ms  (float) — wall-clock inference time in milliseconds
        """
        if self.pipeline is None:
            raise RuntimeError("Model not ready — call train() or load() first.")

        t_start    = time.perf_counter()
        label      = self.pipeline.predict([text])[0]
        proba      = self.pipeline.predict_proba([text])[0]
        latency_ms = (time.perf_counter() - t_start) * 1_000

        return {
            "label":      label,
            "confidence": round(float(max(proba)), 4),
            "latency_ms": round(latency_ms, 3),
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Classify a list of tickets one by one (preserves per-ticket latency)."""
        return [self.predict(t) for t in texts]

    # ── Persist ────────────────────────────────────────────────────────────────

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)
        print(f"Model saved → {path}")

    def load(self, path: Path = MODEL_PATH) -> "TicketClassifier":
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)
        return self


# ── Entrypoint: train + save + quick sanity check ─────────────────────────────

if __name__ == "__main__":
    print("Training classifier...")

    clf = TicketClassifier()
    t0  = time.perf_counter()
    clf.train()
    elapsed = (time.perf_counter() - t0) * 1_000

    print(f"Training complete in {elapsed:.0f}ms")
    clf.save()

    # Sanity checks — one clear example per class
    sanity = [
        ("I was charged twice this month.",                    "billing"),
        ("The CSV export button does absolutely nothing.",     "technical_issue"),
        ("Please add a dark mode option to the app.",          "feature_request"),
        ("Your support is terrible — 5 days and no reply.",   "complaint"),
        ("What are your business hours for support?",          "other"),
    ]

    print("\nSanity checks:")
    all_ok = True
    for text, expected in sanity:
        r  = clf.predict(text)
        ok = r["label"] == expected
        if not ok:
            all_ok = False
        sym = "✓" if ok else "✗"
        print(f"  {sym}  [{r['label']:20s}]  {r['latency_ms']:5.2f}ms  |  {text}")

    if all_ok:
        print("\n✓ All sanity checks passed.")
    else:
        print("\n✗ Some sanity checks failed — review training data.")