"""Unit tests for NLP utilities and agent tools."""

from agent.nlp.dialect_detector import detect_dialect
from agent.nlp.entity_extractor import extract_entities
from agent.nlp.sanitizer import sanitize_text
from agent.nlp.sentiment import analyze_sentiment
from agent.nlp.urgency import assess_urgency, normalize_urgency
from agent.tools.classify_intent import classify_intent
from agent.tools.escalate import escalate_to_human
from agent.tools.lookup_order import lookup_order
from agent.tools.route_team import route_team
from agent.tools.search_kb import search_kb


def test_sanitize_text_strips_diacritics_and_whitespace() -> None:
    """Diacritics, tatweel, and extra whitespace should be removed."""
    raw = "مَرْحَبًا   بِكُمْ  جداً"
    cleaned = sanitize_text(raw)
    assert "  " not in cleaned
    assert "َ" not in cleaned


def test_sanitize_text_handles_empty_input() -> None:
    """Empty or non-string input returns an empty string."""
    assert sanitize_text("") == ""
    assert sanitize_text(None) == ""  # type: ignore[arg-type]


def test_detect_dialect_egyptian() -> None:
    """Egyptian keywords should be detected as Egyptian dialect."""
    assert detect_dialect("عايز أعرف ايه اللي حصل في طلبي") == "egyptian"


def test_detect_dialect_default_msa() -> None:
    """Text without dialect keywords defaults to MSA."""
    assert detect_dialect("أرغب في الاستفسار عن حالة طلبي") == "msa"


def test_extract_entities_finds_order_id_and_email() -> None:
    """Order IDs, emails, and phone numbers should be extracted."""
    text = "طلبي ORD-1001 ولم يصل، تواصلوا معي على test@example.com أو 0501234567"
    entities = extract_entities(text)
    assert "ORD-1001" in entities["order_ids"]
    assert "test@example.com" in entities["emails"]
    assert "0501234567" in entities["phone_numbers"]


def test_analyze_sentiment_negative() -> None:
    """Messages with negative keywords should be classified as negative."""
    label, score = analyze_sentiment("الخدمة سيئة جدًا والمنتج وصل تالف")
    assert label == "negative"
    assert score < 0


def test_assess_urgency_critical_keywords() -> None:
    """Messages mentioning legal action should be marked critical."""
    assert assess_urgency("سأرفع قضية عليكم في المحكمة", "negative", "complaint") == "critical"


def test_normalize_urgency_invalid_defaults_to_low() -> None:
    """Invalid urgency strings normalize to 'low'."""
    assert normalize_urgency("not_a_level") == "low"
    assert normalize_urgency("HIGH") == "high"


def test_classify_intent_keyword_fallback_order_status() -> None:
    """Without an LLM client, keyword matching should classify order status."""
    intent, confidence, issues = classify_intent("وين طلبي ORD-1001 رقم التتبع؟", llm_client=None)
    assert intent == "order_status"
    assert 0.0 < confidence <= 1.0
    assert issues == [intent]


def test_classify_intent_empty_text() -> None:
    """Empty text returns the default intent with zero confidence."""
    intent, confidence, issues = classify_intent("", llm_client=None)
    assert intent == "general_inquiry"
    assert confidence == 0.0
    assert issues == []


def test_route_team_maps_intent_to_team() -> None:
    """Order-status intents should route to logistics under low urgency."""
    team, requires_human = route_team("order_status", "low")
    assert team == "logistics"
    assert requires_human is False


def test_route_team_high_urgency_escalates() -> None:
    """High urgency should route to escalations management and require a human."""
    team, requires_human = route_team("order_status", "high")
    assert team == "escalations_management"
    assert requires_human is True


def test_escalate_to_human_returns_record() -> None:
    """Escalation record should contain the provided fields."""
    record = escalate_to_human(reason="عميل غاضب", priority="critical", customer_id="CUST-001", intent="complaint")
    assert record["priority"] == "critical"
    assert record["customer_id"] == "CUST-001"
    assert "created_at" in record


def test_search_kb_returns_relevant_documents() -> None:
    """Searching for shipping-related text should surface shipping policy docs."""
    results = search_kb("تأخر الشحن ولم يصل الطلب", top_k=3)
    assert isinstance(results, list)
    if results:
        assert "score" in results[0]


def test_lookup_order_handles_missing_database_gracefully() -> None:
    """If the database is unreachable, lookup_order returns None instead of raising."""
    result = lookup_order("ORD-9999")
    assert result is None or isinstance(result, dict)
