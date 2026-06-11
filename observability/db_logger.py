"""Persists triage runs to the PostgreSQL `triage_runs` table."""

import logging
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

from agent.models import TriageResponse
from database.models import TriageRun
from database.postgres import get_session

logger = logging.getLogger(__name__)


def persist_triage_run(customer_id: str, message: str, result: TriageResponse) -> None:
    """Persist a completed triage run to the database.

    Args:
        customer_id: Identifier of the customer who sent the message.
        message: The original (sanitized) customer message.
        result: The `TriageResponse` produced by the orchestrator.

    Raises:
        SQLAlchemyError: Never raised; database errors are caught and logged.
    """
    try:
        with get_session() as session:
            run = TriageRun(
                id=result.run_id,
                customer_id=customer_id,
                message=message,
                intent=result.intent,
                intent_confidence=result.intent_confidence,
                urgency=result.urgency,
                sentiment=result.sentiment,
                dialect=result.dialect,
                entities=result.entities.model_dump(),
                requires_human=result.requires_human,
                routed_team=result.routed_team,
                draft_response_ar=result.draft_response_ar,
                reasoning_trace=[step.model_dump() for step in result.reasoning_trace],
                tools_used=result.tools_used,
                latency_ms=result.latency_ms,
                est_cost_usd=result.est_cost_usd,
            )
            session.add(run)
    except SQLAlchemyError as exc:
        logger.error("Failed to persist triage run %s: %s", result.run_id, exc)


def fetch_triage_run(run_id: UUID) -> TriageRun | None:
    """Fetch a single triage run by id.

    Args:
        run_id: Unique identifier of the triage run.

    Returns:
        TriageRun | None: The ORM record if found, otherwise `None`.
    """
    try:
        with get_session() as session:
            return session.get(TriageRun, run_id)
    except SQLAlchemyError as exc:
        logger.error("Failed to fetch triage run %s: %s", run_id, exc)
        return None


def list_triage_runs(limit: int = 50, offset: int = 0) -> list[TriageRun]:
    """List recent triage runs ordered by creation time, descending.

    Args:
        limit: Maximum number of runs to return.
        offset: Number of runs to skip.

    Returns:
        list[TriageRun]: A list of triage run records, possibly empty.
    """
    try:
        with get_session() as session:
            return (
                session.query(TriageRun)
                .order_by(TriageRun.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
    except SQLAlchemyError as exc:
        logger.error("Failed to list triage runs: %s", exc)
        return []
