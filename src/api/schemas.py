"""Pydantic schemas for the prediction API."""

from pydantic import BaseModel, Field

SENTIMENT_LABELS = ("negative", "neutral", "positive")


class ReviewRequest(BaseModel):
    """A single customer review submitted for sentiment prediction."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Texte brut de l'avis client, dans n'importe quelle langue supportée.",
    )


class SentimentPrediction(BaseModel):
    """Predicted sentiment for a review, with the full probability distribution."""

    sentiment: str = Field(
        ..., description="Sentiment prédit : 'negative', 'neutral' ou 'positive'."
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Probabilité du modèle pour le sentiment prédit."
    )
    probabilities: dict[str, float] = Field(
        ..., description="Probabilité pour chaque sentiment possible."
    )


class BatchReviewsRequest(BaseModel):
    """A batch of reviews (e.g. a whole company's dataset) submitted for analysis."""

    reviews: list[str] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Textes bruts des avis, jusqu'à 500 par appel.",
    )


class BatchSentimentSummary(BaseModel):
    """Aggregate sentiment breakdown across a batch of reviews."""

    total: int = Field(..., description="Nombre d'avis analysés.")
    negative_count: int = Field(..., description="Nombre d'avis négatifs.")
    neutral_count: int = Field(..., description="Nombre d'avis neutres.")
    positive_count: int = Field(..., description="Nombre d'avis positifs.")
    negative_pct: float = Field(..., description="Part d'avis négatifs (0 à 1).")
    neutral_pct: float = Field(..., description="Part d'avis neutres (0 à 1).")
    positive_pct: float = Field(..., description="Part d'avis positifs (0 à 1).")
    overall_score: float = Field(
        ...,
        ge=-1,
        le=1,
        description=(
            "Score global de sentiment entre -1 et 1 : "
            "(positive_count - negative_count) / total."
        ),
    )


class BatchPredictionResponse(BaseModel):
    """Per-review predictions plus the aggregate summary for a batch."""

    predictions: list[SentimentPrediction]
    summary: BatchSentimentSummary
