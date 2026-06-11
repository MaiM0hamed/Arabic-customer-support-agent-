"""Text sanitization utilities for incoming Arabic customer messages."""

import re
import unicodedata

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")
_TATWEEL_RE = re.compile("ـ")
_DIACRITICS_RE = re.compile(r"[ً-ٰٟ]")


def sanitize_text(text: str) -> str:
    """Clean and normalize raw customer message text.

    Performs Unicode normalization, removes control characters, Arabic
    diacritics and tatweel (kashida), and collapses repeated whitespace.

    Args:
        text: Raw customer message text.

    Returns:
        str: The sanitized, UTF-8-safe text. Returns an empty string for
            `None` or non-string input.

    Raises:
        TypeError: Never raised; invalid input is coerced to an empty string.
    """
    if not text or not isinstance(text, str):
        return ""

    try:
        normalized = unicodedata.normalize("NFKC", text)
        normalized = _CONTROL_CHARS_RE.sub("", normalized)
        normalized = _DIACRITICS_RE.sub("", normalized)
        normalized = _TATWEEL_RE.sub("", normalized)
        normalized = _WHITESPACE_RE.sub(" ", normalized)
        return normalized.strip()
    except (TypeError, ValueError):
        return ""
