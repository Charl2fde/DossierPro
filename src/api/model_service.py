"""Loads the fine-tuned XLM-RoBERTa model and exposes prediction methods.

The model is loaded once (see `main.py`'s lifespan handler) and reused across
requests -- reloading a ~1.1GB checkpoint per request would be far too slow.
"""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.api.schemas import BatchSentimentSummary, SentimentPrediction

MODEL_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "models" / "weights" / "xlm_roberta"
)
MAX_LENGTH = (
    64  # must match the value used during training (see train_transformer_model.py)
)


class ModelService:
    """Wraps the tokenizer and fine-tuned model for inference."""

    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        """Load the tokenizer and model from disk.

        Args:
            model_dir: Directory containing the saved model and tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        # id2label is saved with the checkpoint (see train_transformer_model.py),
        # ordered by class index -- read from it instead of a hardcoded list so
        # the API always matches whatever model is actually loaded.
        self.id2label = [
            self.model.config.id2label[i]
            for i in range(len(self.model.config.id2label))
        ]

    def _to_prediction(self, probabilities: list[float]) -> SentimentPrediction:
        """Build a SentimentPrediction from a single softmax probability vector.

        Args:
            probabilities: Per-class probabilities, ordered like `self.id2label`.

        Returns:
            The predicted sentiment, its confidence, and the full distribution.
        """
        predicted_index = max(range(len(probabilities)), key=lambda i: probabilities[i])
        return SentimentPrediction(
            sentiment=self.id2label[predicted_index],
            confidence=probabilities[predicted_index],
            probabilities={
                self.id2label[i]: probabilities[i] for i in range(len(probabilities))
            },
        )

    def predict(self, text: str) -> SentimentPrediction:
        """Predict the sentiment for a single review.

        Args:
            text: Raw review text.

        Returns:
            The predicted sentiment, its confidence, and the full probability distribution.
        """
        inputs = self.tokenizer(
            text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1).squeeze(0).tolist()
        return self._to_prediction(probabilities)

    def predict_batch(self, texts: list[str]) -> list[SentimentPrediction]:
        """Predict the sentiment for a batch of reviews in a single forward pass.

        Args:
            texts: Raw review texts.

        Returns:
            One SentimentPrediction per input text, in the same order.
        """
        inputs = self.tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1).tolist()
        return [self._to_prediction(row) for row in probabilities]


def summarize(predictions: list[SentimentPrediction]) -> BatchSentimentSummary:
    """Aggregate a batch of per-review predictions into an overall summary.

    Args:
        predictions: Per-review sentiment predictions.

    Returns:
        Counts, percentages, and an overall score in [-1, 1] for the batch.
    """
    total = len(predictions)
    negative_count = sum(1 for p in predictions if p.sentiment == "negative")
    neutral_count = sum(1 for p in predictions if p.sentiment == "neutral")
    positive_count = sum(1 for p in predictions if p.sentiment == "positive")

    return BatchSentimentSummary(
        total=total,
        negative_count=negative_count,
        neutral_count=neutral_count,
        positive_count=positive_count,
        negative_pct=negative_count / total,
        neutral_pct=neutral_count / total,
        positive_pct=positive_count / total,
        overall_score=(positive_count - negative_count) / total,
    )
