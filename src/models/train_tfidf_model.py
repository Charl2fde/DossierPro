"""Benchmark several TF-IDF + classical ML classifiers on the sentiment label.

Uses `text_classical` (lemmatized, stopwords removed -- see
`src/data/preprocess.py`), since TF-IDF benefits from that normalization,
unlike the Transformer model which uses the raw `text` column instead
(see `train_transformer_model.py`).

The target is `sentiment` (negative/neutral/positive, derived from `rating`),
chosen over the raw 1-5 rating because it generalizes better across business
sectors and because two rating classes (2, 3 stars) were never learnable in
practice at ~1-2% of the data each (see notebooks/03_modeling.ipynb). The
sentiment distribution is still imbalanced (88% positive), so
`class_weight="balanced"` is used where supported, and the primary
evaluation metric is macro F1 / the per-class confusion matrix rather than
accuracy, which would be dominated by the majority class.

Several classifiers are trained on the exact same TF-IDF features, to give
an evidence-based justification for picking the final production model
(XLM-RoBERTa, see `train_transformer_model.py`) over a classical baseline,
rather than asserting it without comparison.
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
RANDOM_STATE = 42

# Each entry trains on identical TF-IDF features (see build_pipeline), so any
# difference in score is attributable to the classifier alone. MultinomialNB
# has no `class_weight` option -- TF-IDF features are non-negative so it's
# usable, but the imbalance isn't corrected for it (kept anyway, as a classic
# "naive" baseline to contrast against the others).
CLASSIFIERS = {
    "random_forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
    ),
    "logistic_regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    ),
    "linear_svm": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
    "naive_bayes": MultinomialNB(),
}


def load_split(name: str) -> pd.DataFrame:
    """Load a train/val/test split, treating empty `text_classical` as an empty string.

    Args:
        name: One of "train", "val", "test".

    Returns:
        The loaded DataFrame.
    """
    df = pd.read_csv(PROCESSED_DATA_DIR / f"{name}.csv", low_memory=False)
    df["text_classical"] = df["text_classical"].fillna("")
    return df


def build_pipeline(classifier) -> Pipeline:
    """Build a TF-IDF + classifier sklearn pipeline.

    Args:
        classifier: An unfitted sklearn classifier instance.

    Returns:
        An untrained sklearn Pipeline.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
                ),
            ),
            ("classifier", classifier),
        ]
    )


def evaluate(pipeline: Pipeline, df: pd.DataFrame, split_name: str) -> dict:
    """Evaluate a fitted pipeline on a split and log a classification report.

    Args:
        pipeline: A fitted sklearn Pipeline.
        df: The split to evaluate on.
        split_name: Name used for logging (e.g. "val", "test").

    Returns:
        A dict with macro F1, accuracy, and the confusion matrix.
    """
    labels = ["negative", "neutral", "positive"]
    predictions = pipeline.predict(df["text_classical"])
    report = classification_report(
        df["sentiment"], predictions, output_dict=True, zero_division=0
    )
    macro_f1 = f1_score(df["sentiment"], predictions, average="macro", zero_division=0)
    matrix = confusion_matrix(df["sentiment"], predictions, labels=labels)

    logger.info(
        "[%s] accuracy=%.3f macro_f1=%.3f", split_name, report["accuracy"], macro_f1
    )
    logger.info(
        "[%s] classification report:\n%s",
        split_name,
        classification_report(df["sentiment"], predictions, zero_division=0),
    )
    logger.info(
        "[%s] confusion matrix (rows=true, cols=pred, order=%s):\n%s",
        split_name,
        labels,
        matrix,
    )

    return {
        "accuracy": report["accuracy"],
        "macro_f1": macro_f1,
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
    }


def main() -> None:
    """Train every classifier in CLASSIFIERS and evaluate each on val and test."""
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    for name, classifier in CLASSIFIERS.items():
        logger.info("=== Training tfidf_%s on %d rows... ===", name, len(train_df))
        pipeline = build_pipeline(classifier)
        pipeline.fit(train_df["text_classical"], train_df["sentiment"])

        val_metrics = evaluate(pipeline, val_df, f"{name}/val")
        test_metrics = evaluate(pipeline, test_df, f"{name}/test")

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = WEIGHTS_DIR / f"tfidf_{name}.pkl"
        joblib.dump(pipeline, model_path)
        logger.info("Saved pipeline to %s", model_path)

        metrics_path = WEIGHTS_DIR / f"tfidf_{name}_metrics.json"
        metrics_path.write_text(
            json.dumps({"val": val_metrics, "test": test_metrics}, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
