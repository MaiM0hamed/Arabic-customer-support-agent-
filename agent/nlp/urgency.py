"""Rule-based urgency assessment for customer messages."""

_CRITICAL_KEYWORDS: set[str] = {
    "قضية", "محامي", "بلاغ", "أرفع قضية", "حقوق المستهلك", "احتيال", "fraud",
}

_HIGH_KEYWORDS: set[str] = {
    "حالًا", "حالا", "فورًا", "فورا", "عاجل", "urgent", "الآن", "مستحيل",
    "أسوأ", "غير مقبول", "ألغي", "إلغاء نهائي",
}

_HIGH_URGENCY_INTENTS: set[str] = {"damaged_product", "payment_issue"}

_VALID_LEVELS = ("low", "medium", "high", "critical")


def assess_urgency(text: str, sentiment: str, intent: str) -> str:
    """Assess the urgency level of a customer message.

    Args:
        text: Sanitized customer message text.
        sentiment: Sentiment label produced by `analyze_sentiment`
            (`positive`, `neutral`, or `negative`).
        intent: Classified intent identifier.

    Returns:
        str: One of `low`, `medium`, `high`, `critical`.
    """
    if not text:
        return "low"

    if any(keyword in text for keyword in _CRITICAL_KEYWORDS):
        return "critical"

    if any(keyword in text for keyword in _HIGH_KEYWORDS):
        return "high"

    if sentiment == "negative" and intent in _HIGH_URGENCY_INTENTS:
        return "high"

    if sentiment == "negative":
        return "medium"

    return "low"


def normalize_urgency(level: str) -> str:
    """Normalize an arbitrary urgency string to a known level.

    Args:
        level: Urgency level string, possibly from an LLM response.

    Returns:
        str: A valid urgency level from `low`, `medium`, `high`, `critical`.
            Defaults to `low` if `level` is not recognized.
    """
    normalized = (level or "").strip().lower()
    return normalized if normalized in _VALID_LEVELS else "low"
