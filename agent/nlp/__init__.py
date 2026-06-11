"""NLP utilities: sanitization, dialect detection, sentiment, entities, urgency."""

from agent.nlp.dialect_detector import detect_dialect
from agent.nlp.entity_extractor import extract_entities
from agent.nlp.sanitizer import sanitize_text
from agent.nlp.sentiment import analyze_sentiment
from agent.nlp.urgency import assess_urgency, normalize_urgency

__all__ = [
    "sanitize_text",
    "detect_dialect",
    "analyze_sentiment",
    "extract_entities",
    "assess_urgency",
    "normalize_urgency",
]
