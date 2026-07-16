"""FastAPI application exposing the review sentiment analysis model."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request

from src.api.model_service import ModelService, summarize
from src.api.schemas import (
    BatchPredictionResponse,
    BatchReviewsRequest,
    ReviewRequest,
    SentimentPrediction,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once on startup and keep it in app state for reuse."""
    app.state.model_service = ModelService()
    yield


app = FastAPI(
    title="API d'analyse de sentiment des avis clients",
    description=(
        "Prédit le sentiment (négatif/neutre/positif) d'un avis client, dans "
        "n'importe quelle langue supportée, pour n'importe quelle entreprise. "
        "Permet aussi d'analyser un dataset d'avis complet en une fois."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


def get_model_service(request: Request) -> ModelService:
    """Dependency returning the model service loaded at startup."""
    return request.app.state.model_service


@app.get(
    "/health",
    summary="Vérification de l'état du serveur",
    description="Vérifie simplement que l'API est en ligne et répond.",
)
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=SentimentPrediction,
    summary="Prédire le sentiment d'un avis",
    description="Analyse un seul avis client et renvoie son sentiment (négatif/neutre/positif).",
)
def predict(
    payload: ReviewRequest, model_service: ModelService = Depends(get_model_service)
) -> SentimentPrediction:
    """Predict the sentiment for a single review.

    Note: intentionally a sync `def`, not `async def` -- model inference is
    CPU-bound and blocking; FastAPI runs sync endpoints in a thread pool so
    one slow prediction doesn't block the event loop for other requests.
    """
    return model_service.predict(payload.text)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Analyser un dataset d'avis complet",
    description=(
        "Analyse jusqu'à 500 avis en une fois et renvoie le détail par avis "
        "ainsi qu'un résumé global (ex : '78% positif, score global 0.62'). "
        "Pensé pour une entreprise qui fournit tout son dataset d'avis et "
        "veut un verdict d'ensemble plutôt qu'une seule prédiction."
    ),
)
def predict_batch(
    payload: BatchReviewsRequest,
    model_service: ModelService = Depends(get_model_service),
) -> BatchPredictionResponse:
    """Predict the sentiment for a batch of reviews and summarize the results.

    Intended for a company handing over its full reviews dataset and wanting
    an overall verdict (e.g. "78% positive, overall score 0.62") rather than
    a single per-review prediction.
    """
    predictions = model_service.predict_batch(payload.reviews)
    return BatchPredictionResponse(
        predictions=predictions, summary=summarize(predictions)
    )
