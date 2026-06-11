"""Pydantic models shared across the agent and API layers."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    """Incoming request payload for the `/triage` endpoint."""

    message: str = Field(..., min_length=0, description="Raw customer message text.")
    customer_id: str = Field(..., min_length=1, description="Unique customer identifier.")


class Entities(BaseModel):
    """Structured entities extracted from a customer message."""

    order_ids: list[str] = Field(default_factory=list)
    phone_numbers: list[str] = Field(default_factory=list)
    emails: list[str] = Field(default_factory=list)


class ReasoningStep(BaseModel):
    """A single step in the orchestrator's reasoning trace."""

    step: int
    tool: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int


class TriageResponse(BaseModel):
    """Response payload returned by the `/triage` endpoint."""

    run_id: UUID
    intent: str
    intent_confidence: float
    urgency: str
    sentiment: str
    dialect: str
    entities: Entities
    requires_human: bool
    routed_team: str
    draft_response_ar: str
    reasoning_trace: list[ReasoningStep]
    tools_used: list[str]
    latency_ms: int
    est_cost_usd: float


class TriageRunRecord(TriageResponse):
    """A persisted triage run, including request metadata."""

    customer_id: str
    message: str
    created_at: datetime


class HealthResponse(BaseModel):
    """Response payload for the `/health` endpoint."""

    status: str = "ok"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
