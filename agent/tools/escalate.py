"""Escalation tool: produces a structured escalation record for human review."""

from datetime import datetime, timezone
from typing import Any


def escalate_to_human(reason: str, priority: str, customer_id: str = "", intent: str = "") -> dict[str, Any]:
    """Build an escalation record for routing to a human agent.

    Args:
        reason: Free-text explanation of why escalation is required.
        priority: Escalation priority (`low`, `medium`, `high`, `critical`).
        customer_id: Identifier of the customer associated with the case.
        intent: Classified intent identifier for the case.

    Returns:
        dict[str, Any]: A dictionary describing the escalation, including
            `reason`, `priority`, `customer_id`, `intent`, and a UTC
            `created_at` ISO-8601 timestamp.
    """
    valid_priorities = {"low", "medium", "high", "critical"}
    normalized_priority = priority if priority in valid_priorities else "medium"

    return {
        "reason": reason or "تم التصعيد تلقائيًا بناءً على قواعد النظام.",
        "priority": normalized_priority,
        "customer_id": customer_id,
        "intent": intent,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
