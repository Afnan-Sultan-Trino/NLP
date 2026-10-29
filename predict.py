# src/predict.py
"""Predict sentiment for Bangla text.

Usage:
    python src/predict.py "এই ফোনের ক্যামেরা অসাধারণ"
    python src/predict.py            # interactive mode
"""
import sys
from pathlib import Path

import joblib

from preprocess import normalize

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "bangla_sentiment.joblib"

LABEL_BN = {
    "positive": "ইতিবাচক (positive)",
    "negative": "নেতিবাচক (negative)",
    "neutral": "নিরপেক্ষ (neutral)",
}


def load_model():
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run `python src/train.py` first.")
    return joblib.load(MODEL_PATH)


def predict(text: str, model=None) -> dict:
    model = model or load_model()
    clean = normalize(text)
    if not clean:
        raise ValueError("Input is empty after normalisation.")
    proba = model.predict_proba([clean])[0]
    classes = model.classes_
    best = int(proba.argmax())
    return {
        "label": classes[best],
        "confidence": float(proba[best]),
        "scores": {c: float(p) for c, p in zip(classes, proba)},
    }


def _print(text: str, result: dict) -> None:
    print(f"\nText : {text}")
    print(f"Label: {LABEL_BN.get(result['label'], result['label'])}")
    print(f"Conf : {result['confidence']:.3f}")
    for label, score in sorted(result["scores"].items(), key=lambda kv: -kv[1]):
        bar = "#" * int(round(score * 20))
        print(f"  {label:<9} {score:.3f} {bar}")


def main() -> None:
    model = load_model()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        _print(text, predict(text, model))
        return

    print("Bangla sentiment classifier (type 'exit' to quit)")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"exit", "quit"}:
            break
        if not text:
            continue
        _print(text, predict(text, model))


if __name__ == "__main__":
    main()