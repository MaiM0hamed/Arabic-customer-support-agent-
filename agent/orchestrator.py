"""Main orchestration workflow: a raw, sequential tool-calling pipeline."""

import logging
import time
import uuid
from typing import Any

from agent.models import Entities, ReasoningStep, TriageResponse
from agent.nlp import (
    analyze_sentiment,
    detect_dialect,
    extract_entities,
    sanitize_text,
)
from agent.nlp.urgency import assess_urgency
from agent.tools import (
    classify_intent,
    draft_response,
    escalate_to_human,
    lookup_order,
    route_team,
    search_kb,
)
from llm.client import OpenRouterClient

logger = logging.getLogger(__name__)


class TriageOrchestrator:
    """Runs the end-to-end triage pipeline for an incoming customer message."""

    def __init__(self, llm_client: OpenRouterClient | None = None) -> None:
        """Initialize the orchestrator.

        Args:
            llm_client: Optional `OpenRouterClient` used for LLM-backed
                steps (intent classification and response drafting). If
                `None`, a new client is created from application settings.
        """
        self.llm_client = llm_client if llm_client is not None else OpenRouterClient()

    def run(self, message: str, customer_id: str) -> TriageResponse:
        """Execute the full triage pipeline for a customer message.

        Pipeline: sanitizer -> dialect detection -> entity extraction ->
        sentiment -> classify intent -> lookup order -> search kb ->
        urgency -> route team -> draft response -> optional escalation.

        Args:
            message: Raw customer message text.
            customer_id: Unique identifier of the customer.

        Returns:
            TriageResponse: The structured result of the triage run,
                including a full reasoning trace, tools used, latency,
                and estimated LLM cost.
        """
        start_time = time.perf_counter()
        run_id = uuid.uuid4()
        trace: list[ReasoningStep] = []
        tools_used: list[str] = []

        def record(step_index: int, tool: str, step_input: dict[str, Any], step_output: dict[str, Any], step_start: float) -> None:
            tools_used.append(tool)
            trace.append(
                ReasoningStep(
                    step=step_index,
                    tool=tool,
                    input=step_input,
                    output=step_output,
                    latency_ms=int((time.perf_counter() - step_start) * 1000),
                )
            )

        # 1. Sanitize
        step_start = time.perf_counter()
        clean_message = sanitize_text(message)
        record(1, "sanitizer", {"message": message}, {"clean_message": clean_message}, step_start)

        # 2. Dialect detection
        step_start = time.perf_counter()
        dialect = detect_dialect(clean_message)
        record(2, "detect_dialect", {"text": clean_message}, {"dialect": dialect}, step_start)

        # 3. Entity extraction
        step_start = time.perf_counter()
        entities_raw = extract_entities(clean_message)
        record(3, "extract_entities", {"text": clean_message}, entities_raw, step_start)

        # 4. Sentiment
        step_start = time.perf_counter()
        sentiment, sentiment_score = analyze_sentiment(clean_message)
        record(
            4,
            "analyze_sentiment",
            {"text": clean_message},
            {"sentiment": sentiment, "score": sentiment_score},
            step_start,
        )

        # 5. Classify intent
        step_start = time.perf_counter()
        intent, intent_confidence = classify_intent(clean_message, llm_client=self.llm_client)
        record(
            5,
            "classify_intent",
            {"text": clean_message},
            {"intent": intent, "confidence": intent_confidence},
            step_start,
        )

        # 6. Lookup order (if an order id was found)
        step_start = time.perf_counter()
        order_data = None
        if entities_raw["order_ids"]:
            order_data = lookup_order(entities_raw["order_ids"][0])
        record(
            6,
            "lookup_order",
            {"order_ids": entities_raw["order_ids"]},
            {"order": order_data},
            step_start,
        )

        # 7. Search knowledge base
        step_start = time.perf_counter()
        kb_results = search_kb(clean_message, top_k=3)
        record(
            7,
            "search_kb",
            {"query": clean_message},
            {"results": [{"id": doc.get("id"), "score": doc.get("score")} for doc in kb_results]},
            step_start,
        )

        # 8. Assess urgency
        step_start = time.perf_counter()
        urgency = assess_urgency(clean_message, sentiment, intent)
        record(8, "assess_urgency", {"sentiment": sentiment, "intent": intent}, {"urgency": urgency}, step_start)

        # 9. Route team
        step_start = time.perf_counter()
        routed_team, requires_human = route_team(intent, urgency)
        record(
            9,
            "route_team",
            {"intent": intent, "urgency": urgency},
            {"team": routed_team, "requires_human": requires_human},
            step_start,
        )

        # 10. Draft response
        step_start = time.perf_counter()
        draft = draft_response(
            message=clean_message,
            intent=intent,
            dialect=dialect,
            context=kb_results,
            llm_client=self.llm_client,
        )
        record(10, "draft_response", {"intent": intent, "dialect": dialect}, {"draft_response_ar": draft}, step_start)

        # 11. Optional escalation
        if requires_human:
            step_start = time.perf_counter()
            escalation = escalate_to_human(
                reason=f"تم التصعيد تلقائيًا بسبب مستوى الإلحاح: {urgency}",
                priority=urgency,
                customer_id=customer_id,
                intent=intent,
            )
            record(11, "escalate_to_human", {"urgency": urgency}, escalation, step_start)

        total_latency_ms = int((time.perf_counter() - start_time) * 1000)

        return TriageResponse(
            run_id=run_id,
            intent=intent,
            intent_confidence=intent_confidence,
            urgency=urgency,
            sentiment=sentiment,
            dialect=dialect,
            entities=Entities(**entities_raw),
            requires_human=requires_human,
            routed_team=routed_team,
            draft_response_ar=draft,
            reasoning_trace=trace,
            tools_used=tools_used,
            latency_ms=total_latency_ms,
            est_cost_usd=self.llm_client.total_cost_usd,
        )
