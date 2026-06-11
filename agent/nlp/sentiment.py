"""Lexicon-based sentiment analysis for Arabic customer messages."""

_NEGATIVE_WORDS: set[str] = {
    "سيء", "سيئة", "مستاء", "غاضب", "زعلان", "خايب", "مشكلة", "تعبت", "تعبان",
    "مكسور", "تالف", "معطوب", "متأخر", "تأخر", "خصم مرتين", "غير راضي",
    "قضية", "بلاغ", "احتجاج", "أسوأ", "محبط", "مزعج", "خايس", "وحش",
}

_POSITIVE_WORDS: set[str] = {
    "شكرا", "شكرًا", "ممتاز", "رائع", "جميل", "تمام", "مشكور", "حلو",
    "سعيد", "راضي", "ممتازة", "تسلم", "يعطيك العافية", "كويس",
}


def analyze_sentiment(text: str) -> tuple[str, float]:
    """Estimate the sentiment polarity of a customer message.

    Args:
        text: Sanitized customer message text.

    Returns:
        tuple[str, float]: A tuple of (`label`, `score`) where `label` is
            one of `positive`, `neutral`, `negative`, and `score` is a
            value in `[-1.0, 1.0]` (negative means negative sentiment).
    """
    if not text:
        return "neutral", 0.0

    negative_hits = sum(1 for word in _NEGATIVE_WORDS if word in text)
    positive_hits = sum(1 for word in _POSITIVE_WORDS if word in text)

    if negative_hits == 0 and positive_hits == 0:
        return "neutral", 0.0

    total = negative_hits + positive_hits
    score = (positive_hits - negative_hits) / total

    if score > 0.2:
        return "positive", score
    if score < -0.2:
        return "negative", score
    return "neutral", score
