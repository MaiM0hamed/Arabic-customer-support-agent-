"""API route definitions for the Arabic customer support agent."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from agent.models import Entities, HealthResponse, ReasoningStep, TriageRequest, TriageResponse, TriageRunRecord
from agent.orchestrator import TriageOrchestrator
from api.security import require_api_key
from database.models import TriageRun
from observability.db_logger import fetch_triage_run, list_triage_runs, persist_escalation, persist_triage_run
from observability.logger import write_trajectory

logger = logging.getLogger(__name__)
router = APIRouter()


def _to_record(run: TriageRun) -> TriageRunRecord:
    """Convert a `TriageRun` ORM object into a `TriageRunRecord` response model.

    Args:
        run: The persisted triage run.

    Returns:
        TriageRunRecord: The API representation of the run.
    """
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


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health-check endpoint.

    Returns:
        HealthResponse: A simple status payload indicating the service is up.
    """
    return HealthResponse()


@router.post("/triage", response_model=TriageResponse, dependencies=[Depends(require_api_key)])
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
    orchestrator = TriageOrchestrator()
    try:
        result = orchestrator.run(message=request.message, customer_id=request.customer_id)

        persist_triage_run(customer_id=request.customer_id, message=request.message, result=result)

        if result.requires_human:
            escalation_step = next(
                (step for step in result.reasoning_trace if step.tool == "escalate_to_human"), None
            )
            reason = escalation_step.output.get("reason", "") if escalation_step else ""
            persist_escalation(
                triage_run_id=result.run_id,
                customer_id=request.customer_id,
                intent=result.intent,
                priority=result.urgency,
                reason=reason,
            )

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
    finally:
        orchestrator.llm_client.close()


@router.get("/triage/{run_id}", response_model=TriageRunRecord, dependencies=[Depends(require_api_key)])
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

    return _to_record(run)


@router.get("/runs", response_model=list[TriageRunRecord], dependencies=[Depends(require_api_key)])
def get_runs(limit: int = 50, offset: int = 0) -> list[TriageRunRecord]:
    """List recent triage runs.

    Args:
        limit: Maximum number of runs to return (default 50, capped at 200).
        offset: Number of runs to skip for pagination (default 0).

    Returns:
        list[TriageRunRecord]: A list of recent triage runs, most recent first.
    """
    limit = max(1, min(limit, 200))
    runs = list_triage_runs(limit=limit, offset=offset)
    return [_to_record(run) for run in runs]
