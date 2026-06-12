"""Tests for the FastAPI HTTP endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from config import settings


@pytest.fixture(autouse=True)
def _disable_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable real LLM calls so tests run deterministically offline."""
    monkeypatch.setattr(settings, "openrouter_api_key", "")


@pytest.fixture
def client() -> TestClient:
    """Provide a FastAPI test client."""
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    """GET /health should return status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_triage_endpoint_returns_expected_fields(client: TestClient) -> None:
    """POST /triage should return a fully populated triage response."""
    response = client.post(
        "/triage",
        json={"message": "وصلني المنتج مكسور ومعطوب من الصندوق", "customer_id": "CUST-001"},
    )
    assert response.status_code == 200

    body = response.json()
    for field in (
        "run_id",
        "intent",
        "intent_confidence",
        "urgency",
        "sentiment",
        "dialect",
        "entities",
        "requires_human",
        "routed_team",
        "draft_response_ar",
        "reasoning_trace",
        "tools_used",
        "latency_ms",
        "est_cost_usd",
    ):
        assert field in body

    assert body["intent"] == "damaged_product"
    assert isinstance(body["reasoning_trace"], list)
    assert len(body["reasoning_trace"]) > 0


def test_triage_endpoint_rejects_missing_customer_id(client: TestClient) -> None:
    """POST /triage without customer_id should return a validation error."""
    response = client.post("/triage", json={"message": "مرحبا"})
    assert response.status_code == 422


def test_get_runs_returns_list(client: TestClient) -> None:
    """GET /runs should return a list (possibly empty if DB is unavailable)."""
    response = client.get("/runs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
