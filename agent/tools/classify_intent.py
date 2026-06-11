"""Intent classification tool: LLM-based with a keyword fallback."""

import json
import logging
from pathlib import Path

from config import settings
from llm.client import OpenRouterClient, OpenRouterError

logger = logging.getLogger(__name__)

_TAXONOMY_PATH = Path(settings.knowledge_base_dir) / "intent_taxonomy.json"


def _load_taxonomy() -> list[dict]:
    """Load the intent taxonomy from disk.

    Returns:
        list[dict]: A list of intent definitions, each with `id`, `label_ar`,
            and `keywords`. Returns an empty list if the file is missing or
            malformed.
    """
    try:
        with open(_TAXONOMY_PATH, encoding="utf-8") as handle:
            data = json.load(handle)
        return data.get("intents", [])
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load intent taxonomy from %s: %s", _TAXONOMY_PATH, exc)
        return []


_TAXONOMY = _load_taxonomy()
_VALID_INTENT_IDS = {intent["id"] for intent in _TAXONOMY} or {"general_inquiry"}
_DEFAULT_INTENT = "general_inquiry"


def _keyword_fallback(text: str) -> tuple[str, float]:
    """Classify intent using simple keyword matching against the taxonomy.

    Args:
        text: Sanitized customer message text.

    Returns:
        tuple[str, float]: The matched intent id and a confidence score.
            Falls back to `general_inquiry` with confidence `0.3` if no
            keyword matches.
    """
    best_intent = _DEFAULT_INTENT
    best_score = 0
    for intent in _TAXONOMY:
        score = sum(1 for keyword in intent.get("keywords", []) if keyword in text)
        if score > best_score:
            best_score = score
            best_intent = intent["id"]

    if best_score == 0:
        return _DEFAULT_INTENT, 0.3
    confidence = min(0.5 + 0.15 * best_score, 0.95)
    return best_intent, confidence


def classify_intent(text: str, llm_client: OpenRouterClient | None = None) -> tuple[str, float]:
    """Classify the intent of a customer message.

    Attempts an LLM-based classification first, falling back to keyword
    matching if the LLM call fails or returns an invalid result.

    Args:
        text: Sanitized customer message text.
        llm_client: Optional `OpenRouterClient` instance. If `None`, only
            the keyword fallback is used.

    Returns:
        tuple[str, float]: The classified intent id (guaranteed to be a
            valid taxonomy id) and a confidence score in `[0, 1]`.
    """
    if not text:
        return _DEFAULT_INTENT, 0.0

    if llm_client is not None:
        try:
            prompt_path = Path("llm/prompts/classify_intent.txt")
            template = prompt_path.read_text(encoding="utf-8")
            intents_str = "\n".join(f"- {i['id']}: {i['label_ar']}" for i in _TAXONOMY)
            prompt = template.format(intents=intents_str, message=text)

            response = llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            result = llm_client.extract_json(response)
            intent = result.get("intent")
            confidence = float(result.get("confidence", 0.0))

            if intent in _VALID_INTENT_IDS:
                return intent, max(0.0, min(confidence, 1.0))
            logger.info("LLM returned invalid intent '%s', falling back to keywords.", intent)
        except (OpenRouterError, ValueError, KeyError, OSError) as exc:
            logger.warning("LLM intent classification failed: %s", exc)

    return _keyword_fallback(text)
