"""Tests for the FastAPI prediction endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

SENTIMENT_LABELS = {"negative", "neutral", "positive"}


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A TestClient that runs the app's lifespan, loading the real model once for all tests."""
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_sentiment(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "Great app, love it!"})
    assert response.status_code == 200

    body = response.json()
    assert body["sentiment"] in SENTIMENT_LABELS
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"].keys()) == SENTIMENT_LABELS
    assert pytest.approx(sum(body["probabilities"].values()), abs=1e-3) == 1.0


def test_predict_handles_negative_review(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"text": "Terrible service, my account was frozen for no reason."},
    )
    assert response.status_code == 200
    assert response.json()["sentiment"] in SENTIMENT_LABELS


def test_predict_handles_non_english_review(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"text": "Une application incroyable, je recommande vivement !"},
    )
    assert response.status_code == 200
    assert response.json()["sentiment"] in SENTIMENT_LABELS


def test_predict_rejects_empty_text(client: TestClient) -> None:
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_rejects_missing_text(client: TestClient) -> None:
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_batch_returns_summary(client: TestClient) -> None:
    response = client.post(
        "/predict/batch",
        json={
            "reviews": [
                "Great app, love it!",
                "Terrible service, my account was frozen for no reason.",
                "It's okay, does the job.",
            ]
        },
    )
    assert response.status_code == 200

    body = response.json()
    assert len(body["predictions"]) == 3
    for prediction in body["predictions"]:
        assert prediction["sentiment"] in SENTIMENT_LABELS

    summary = body["summary"]
    assert summary["total"] == 3
    assert (
        summary["negative_count"] + summary["neutral_count"] + summary["positive_count"]
        == 3
    )
    assert -1.0 <= summary["overall_score"] <= 1.0


def test_predict_batch_rejects_empty_list(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"reviews": []})
    assert response.status_code == 422
