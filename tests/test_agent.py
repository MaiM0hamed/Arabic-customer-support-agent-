"""Integration tests for the TriageOrchestrator pipeline."""

from agent.models import TriageResponse
from agent.orchestrator import TriageOrchestrator
from llm.client import OpenRouterClient


def _orchestrator() -> TriageOrchestrator:
    """Build an orchestrator with an unconfigured LLM client (forces fallbacks)."""
    return TriageOrchestrator(llm_client=OpenRouterClient(api_key=""))


def test_orchestrator_run_returns_complete_response() -> None:
    """A full run should return a populated TriageResponse with a reasoning trace."""
    orchestrator = _orchestrator()
    result = orchestrator.run(message="وين طلبي رقم ORD-1001؟ تأخر كثير", customer_id="CUST-001")

    assert isinstance(result, TriageResponse)
    assert result.intent
    assert result.dialect
    assert result.sentiment
    assert result.urgency in {"low", "medium", "high", "critical"}
    assert result.routed_team
    assert result.draft_response_ar
    assert len(result.reasoning_trace) >= 10
    assert "classify_intent" in result.tools_used
    assert "draft_response" in result.tools_used
    assert result.entities.order_ids == ["ORD-1001"]


def test_orchestrator_high_urgency_triggers_escalation() -> None:
    """A message expressing legal threats should require human escalation."""
    orchestrator = _orchestrator()
    result = orchestrator.run(message="سأرفع قضية عليكم إن لم تحلوا مشكلتي الآن", customer_id="CUST-002")

    assert result.urgency == "critical"
    assert result.requires_human is True
    assert result.routed_team == "escalations_management"
    assert "escalate_to_human" in result.tools_used


def test_orchestrator_handles_empty_message() -> None:
    """An empty message should not crash the pipeline."""
    orchestrator = _orchestrator()
    result = orchestrator.run(message="", customer_id="CUST-003")

    assert result.intent == "general_inquiry"
    assert result.draft_response_ar
