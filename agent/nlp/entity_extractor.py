"""Regex-based entity extraction from customer messages."""

import re

_ORDER_ID_RE = re.compile(r"\bORD-\d{3,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?0\d{8,9}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extract structured entities (order IDs, phone numbers, emails) from text.

    Args:
        text: Sanitized customer message text.

    Returns:
        dict[str, list[str]]: A dictionary with keys `order_ids`,
            `phone_numbers`, and `emails`, each mapping to a list of
            unique matches found in the text. Lists are empty if nothing
            is found.
    """
    if not text:
        return {"order_ids": [], "phone_numbers": [], "emails": []}

    order_ids = sorted(set(match.upper() for match in _ORDER_ID_RE.findall(text)))
    phone_numbers = sorted(set(_PHONE_RE.findall(text)))
    emails = sorted(set(_EMAIL_RE.findall(text)))

    return {
        "order_ids": order_ids,
        "phone_numbers": phone_numbers,
        "emails": emails,
    }
