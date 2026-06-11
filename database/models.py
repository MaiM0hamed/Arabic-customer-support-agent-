"""SQLAlchemy ORM models for orders and triage runs."""

import uuid
from datetime import datetime

from sqlalchemy import ARRAY, JSON, Boolean, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""


class Order(Base):
    """Represents a customer order used for order-lookup tooling."""

    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    items: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    total_amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="SAR")
    tracking_number: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expected_delivery_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class TriageRun(Base):
    """Represents a single triage run produced by the orchestrator."""

    __tablename__ = "triage_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    message: Mapped[str] = mapped_column(String, nullable=False)

    intent: Mapped[str] = mapped_column(String(64), nullable=False)
    intent_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    urgency: Mapped[str] = mapped_column(String(16), nullable=False)
    sentiment: Mapped[str] = mapped_column(String(16), nullable=False)
    dialect: Mapped[str] = mapped_column(String(32), nullable=False)

    entities: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    requires_human: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    routed_team: Mapped[str] = mapped_column(String(64), nullable=False)
    draft_response_ar: Mapped[str] = mapped_column(String, nullable=False, default="")

    reasoning_trace: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    tools_used: Mapped[list] = mapped_column(ARRAY(String), nullable=False, default=list)

    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    est_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
