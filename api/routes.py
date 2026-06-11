"""API route definitions for the Arabic customer support agent."""

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException

from agent.models import Entities, HealthResponse, ReasoningStep, TriageRequest, TriageResponse, TriageRunRecord
from agent.orchestrator import TriageOrchestrator
from observability.db_logger import fetch_triage_run, list_triage_runs, persist_triage_run
from observability.logger import write_trajectory

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health-check endpoint.

    Returns:
        HealthResponse: A simple status payload indicating the service is up.
    """
    return HealthResponse()


@router.post("/triage", response_model=TriageResponse)
def triage(request: TriageRequest) -> TriageResponse:
    """Run the triage pipeline for an incoming customer message.

    Args:
        request: The incoming triage request containing `message` and
            `customer_id`.

    Returns:
        TriageResponse: The structured triage result.

    Raises:
        HTTPException: With status code 500 if the orchestrator fails
            unexpectedly.
    """
    try:
        orchestrator = TriageOrchestrator()
        result = orchestrator.run(message=request.message, customer_id=request.customer_id)

        persist_triage_run(customer_id=request.customer_id, message=request.message, result=result)
        write_trajectory(
            run_id=result.run_id,
            trajectory={
                "run_id": str(result.run_id),
                "customer_id": request.customer_id,
                "message": request.message,
                "reasoning_trace": [step.model_dump() for step in result.reasoning_trace],
                "result": result.model_dump(mode="json"),
            },
        )
        return result
    except Exception as exc:
        logger.exception("Triage pipeline failed: %s", exc)
        raise HTTPException(status_code=500, detail="Triage pipeline failed.") from exc


@router.get("/triage/{run_id}", response_model=TriageRunRecord)
def get_triage_run(run_id: UUID) -> TriageRunRecord:
    """Fetch a single triage run by its identifier.

    Args:
        run_id: The unique identifier of the triage run.

    Returns:
        TriageRunRecord: The persisted triage run.

    Raises:
        HTTPException: With status code 404 if the run does not exist.
    """
    run = fetch_triage_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Triage run not found.")

    return TriageRunRecord(
        run_id=run.id,
        customer_id=run.customer_id,
        message=run.message,
        intent=run.intent,
        intent_confidence=run.intent_confidence,
        urgency=run.urgency,
        sentiment=run.sentiment,
        dialect=run.dialect,
        entities=Entities(**run.entities),
        requires_human=run.requires_human,
        routed_team=run.routed_team,
        draft_response_ar=run.draft_response_ar,
        reasoning_trace=[ReasoningStep(**step) for step in run.reasoning_trace],
        tools_used=list(run.tools_used),
        latency_ms=run.latency_ms,
        est_cost_usd=run.est_cost_usd,
        created_at=run.created_at,
    )


@router.get("/runs", response_model=list[TriageRunRecord])
def get_runs(limit: int = 50, offset: int = 0) -> list[TriageRunRecord]:
    """List recent triage runs.

    Args:
        limit: Maximum number of runs to return (default 50).
        offset: Number of runs to skip for pagination (default 0).

    Returns:
        list[TriageRunRecord]: A list of recent triage runs, most recent first.
    """
    runs = list_triage_runs(limit=limit, offset=offset)
    return [
        TriageRunRecord(
            run_id=run.id,
            customer_id=run.customer_id,
            message=run.message,
            intent=run.intent,
            intent_confidence=run.intent_confidence,
            urgency=run.urgency,
            sentiment=run.sentiment,
            dialect=run.dialect,
            entities=Entities(**run.entities),
            requires_human=run.requires_human,
            routed_team=run.routed_team,
            draft_response_ar=run.draft_response_ar,
            reasoning_trace=[ReasoningStep(**step) for step in run.reasoning_trace],
            tools_used=list(run.tools_used),
            latency_ms=run.latency_ms,
            est_cost_usd=run.est_cost_usd,
            created_at=run.created_at,
        )
        for run in runs
    ]
