"""Fine-tune XLM-RoBERTa on the sentiment label (negative/neutral/positive).

Uses the raw `text` column (not the TF-IDF-oriented `text_classical`):
Transformers rely on punctuation and function words for context, and lose
accuracy if stopwords/inflection are stripped beforehand -- unlike the
TF-IDF + Random Forest baseline (see `train_tfidf_model.py`).

Model choice: XLM-RoBERTa instead of the CamemBERT/RoBERTa named in the
project brief, because the corpus was deliberately scoped to 15 Latin-script
languages (see project decisions in Step 2/EDA) -- CamemBERT (French-only)
and RoBERTa (English-only) would only cover a fraction of the data.
XLM-RoBERTa is the multilingual counterpart of the same architecture
family, so it stays consistent with the spirit of the brief.

No GPU is available in this environment, so fine-tuning runs on a
stratified subsample of the training split (`TRAIN_SUBSAMPLE_SIZE`) to keep
training time practical on CPU. Validation and test stay full-size and
identical to the TF-IDF model's, so the two models remain comparable.

The target is `sentiment` rather than the raw 1-5 rating (see
`src/data/preprocess.py`): it generalizes better across business sectors
and the rating-level model never learned classes 2/3-star (~1-2% of data
each, see notebooks/03_modeling.ipynb).
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

os.environ.setdefault("WANDB_DISABLED", "true")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights" / "xlm_roberta"
MODEL_NAME = "xlm-roberta-base"
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}
NUM_LABELS = len(SENTIMENT_LABELS)
# 64 tokens covers the vast majority of reviews (median length is ~4 words, see EDA);
# kept short deliberately -- CPU-only training, and attention cost grows quadratically
# with sequence length. Benchmarked: an earlier attempt at max_length=128 with 16k
# rows / 3 epochs projected to ~3h total (measured at ~3.7s/step); this config targets
# ~45-60 min instead.
MAX_LENGTH = 64
# Doubled vs. the single-business validation run (8,000): with the merged
# multi-business dataset the "neutral" class still sits at only ~2.8% of
# rows, and the smaller subsample left it with too few absolute examples
# (~136) to learn at all (0% precision/recall, see notebooks/03_modeling.ipynb).
TRAIN_SUBSAMPLE_SIZE = 16_000
RANDOM_STATE = 42


class ReviewsDataset(torch.utils.data.Dataset):
    """Minimal torch Dataset wrapping pre-tokenized encodings and labels."""

    def __init__(self, encodings: dict, labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class WeightedLossTrainer(Trainer):
    """Trainer using a class-weighted cross-entropy loss.

    The default Trainer loss is unweighted, which is fine for balanced
    data but catastrophic here: on a first run without weighting, the
    model collapsed to predicting only the two majority-adjacent classes
    (1 and 5) and scored zero precision/recall on classes 2, 3 and 4 --
    despite a deceptively high accuracy (0.87) driven entirely by the
    class imbalance. The Random Forest baseline did not have this problem
    because `class_weight="balanced"` was set there from the start.
    """

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device)
        )
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def load_split(name: str) -> pd.DataFrame:
    """Load a train/val/test split.

    Args:
        name: One of "train", "val", "test".

    Returns:
        The loaded DataFrame.
    """
    return pd.read_csv(PROCESSED_DATA_DIR / f"{name}.csv", low_memory=False)


def subsample_train(train_df: pd.DataFrame, size: int) -> pd.DataFrame:
    """Take a stratified subsample of the training split for CPU-feasible fine-tuning.

    Args:
        train_df: The full training split.
        size: Target number of rows.

    Returns:
        A stratified subsample of approximately `size` rows.
    """
    subsample, _ = train_test_split(
        train_df,
        train_size=size,
        stratify=train_df["sentiment"],
        random_state=RANDOM_STATE,
    )
    return subsample


def tokenize(tokenizer: AutoTokenizer, texts: list[str]) -> dict:
    """Tokenize a batch of texts with padding/truncation.

    Args:
        tokenizer: A HuggingFace tokenizer.
        texts: List of raw review texts.

    Returns:
        A dict of tensors ("input_ids", "attention_mask", ...).
    """
    return tokenizer(
        texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """Compute accuracy and macro F1 for the Trainer's evaluation loop.

    Args:
        eval_pred: Predictions and labels from the Trainer.

    Returns:
        A dict with "accuracy" and "macro_f1".
    """
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    return {
        "accuracy": (predictions == labels).mean(),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def evaluate_and_report(
    trainer: Trainer,
    dataset: ReviewsDataset,
    true_sentiments: pd.Series,
    split_name: str,
) -> dict:
    """Run predictions on a split and log a full classification report.

    Args:
        trainer: The fitted Trainer.
        dataset: The tokenized dataset (labels are 0..NUM_LABELS-1).
        true_sentiments: The original sentiment strings, for reporting.
        split_name: Name used for logging (e.g. "val", "test").

    Returns:
        A dict with macro F1, accuracy, and the confusion matrix.
    """
    predicted_ids = np.argmax(trainer.predict(dataset).predictions, axis=1)
    predictions = [SENTIMENT_LABELS[i] for i in predicted_ids]
    report = classification_report(
        true_sentiments, predictions, output_dict=True, zero_division=0
    )
    macro_f1 = f1_score(true_sentiments, predictions, average="macro", zero_division=0)
    matrix = confusion_matrix(true_sentiments, predictions, labels=SENTIMENT_LABELS)

    logger.info(
        "[%s] accuracy=%.3f macro_f1=%.3f", split_name, report["accuracy"], macro_f1
    )
    logger.info(
        "[%s] classification report:\n%s",
        split_name,
        classification_report(true_sentiments, predictions, zero_division=0),
    )
    logger.info(
        "[%s] confusion matrix (rows=true, cols=pred, order=%s):\n%s",
        split_name,
        SENTIMENT_LABELS,
        matrix,
    )

    return {
        "accuracy": report["accuracy"],
        "macro_f1": macro_f1,
        "labels": SENTIMENT_LABELS,
        "confusion_matrix": matrix.tolist(),
    }


def main() -> None:
    """Fine-tune XLM-RoBERTa and evaluate it on val and test."""
    train_df = subsample_train(load_split("train"), TRAIN_SUBSAMPLE_SIZE)
    val_df = load_split("val")
    test_df = load_split("test")
    logger.info(
        "Fine-tuning on %d rows (stratified subsample of the train split)",
        len(train_df),
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    train_dataset = ReviewsDataset(
        tokenize(tokenizer, train_df["text"].fillna("").tolist()),
        train_df["sentiment"].map(LABEL_TO_ID).tolist(),
    )
    val_dataset = ReviewsDataset(
        tokenize(tokenizer, val_df["text"].fillna("").tolist()),
        val_df["sentiment"].map(LABEL_TO_ID).tolist(),
    )
    test_dataset = ReviewsDataset(
        tokenize(tokenizer, test_df["text"].fillna("").tolist()),
        test_df["sentiment"].map(LABEL_TO_ID).tolist(),
    )

    training_args = TrainingArguments(
        output_dir=str(WEIGHTS_DIR / "checkpoints"),
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,  # lowered from the 5e-5 default for stability with weighted loss
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=RANDOM_STATE,
    )

    balanced_weights = compute_class_weight(
        "balanced",
        classes=np.arange(NUM_LABELS),
        y=train_df["sentiment"].map(LABEL_TO_ID).to_numpy(),
    )
    # Full inverse-frequency weighting destabilized training in the earlier 5-class
    # experiment (loss never decreased, model collapsed to a single class -- see
    # notebooks/03_modeling.ipynb). Sqrt-scaled weights trade off some
    # imbalance-correction for training stability; kept here for the same reason.
    class_weights = torch.tensor(np.sqrt(balanced_weights), dtype=torch.float32)
    logger.info("Class weights (%s): %s", SENTIMENT_LABELS, class_weights.tolist())

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    val_metrics = evaluate_and_report(trainer, val_dataset, val_df["sentiment"], "val")
    test_metrics = evaluate_and_report(
        trainer, test_dataset, test_df["sentiment"], "test"
    )

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    model.config.id2label = {i: label for i, label in enumerate(SENTIMENT_LABELS)}
    model.config.label2id = LABEL_TO_ID
    trainer.save_model(str(WEIGHTS_DIR))
    tokenizer.save_pretrained(str(WEIGHTS_DIR))
    logger.info("Saved model and tokenizer to %s", WEIGHTS_DIR)

    metrics_path = WEIGHTS_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps({"val": val_metrics, "test": test_metrics}, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
