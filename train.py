# src/train.py
"""Train a Bangla sentiment classifier (TF-IDF + Logistic Regression)."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from preprocess import normalize

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "bangla_sentiment.csv"
MODEL_PATH = ROOT / "models" / "bangla_sentiment.joblib"


def build_pipeline() -> Pipeline:
    """Word n-grams + character n-grams -> Logistic Regression.

    Character n-grams matter a lot for Bangla because it is highly
    agglutinative: the same root shows up with many suffixes.
    """
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                 min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                 min_df=1, sublinear_tf=True)),
    ])
    clf = LogisticRegression(max_iter=1000, C=5.0, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"])
    df["clean"] = df["text"].map(normalize)
    df = df[df["clean"].str.len() > 0]
    return df


def show_top_features(pipe: Pipeline, top_n: int = 6) -> None:
    """Print the most discriminative n-grams per class (nice for the README)."""
    feats = pipe.named_steps["features"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    print("\nTop n-grams per class:")
    for i, cls in enumerate(clf.classes_):
        top = clf.coef_[i].argsort()[-top_n:][::-1]
        pretty = [feats[j].replace("word__", "").replace("char__", "") for j in top]
        print(f"  {cls:<9}: {', '.join(pretty)}")


def main() -> None:
    df = load_data()
    X, y = df["clean"], df["label"]
    print(f"Loaded {len(df)} examples | class balance: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    print(f"\nHold-out accuracy: {accuracy_score(y_test, preds):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, preds, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, preds, labels=pipe.classes_))

    cv = cross_val_score(build_pipeline(), X, y, cv=5, scoring="accuracy")
    print(f"\n5-fold CV accuracy: {cv.mean():.3f} +/- {cv.std():.3f}  {cv.round(3)}")

    show_top_features(pipe)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()