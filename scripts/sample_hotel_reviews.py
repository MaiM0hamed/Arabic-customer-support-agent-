"""Sample HARD Arabic hotel reviews and reframe them as incoming SoukAI support messages.

Each sampled review is treated as a customer message. The 1-5 star rating is
mapped to a rough sentiment/urgency proxy, and a heuristic keyword classifier
assigns one of the existing intents from `data/knowledge_base/intent_taxonomy.json`.
These proxy labels are NOT gold labels -- the hand-labeled gold test set lives
separately in `hotel_labeled_test_set.json`.
"""

import csv
import json
import logging
import random
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

_SOURCE_PATH = Path(
    "/mnt/c/Users/FreeComp/Downloads/HARD-Arabic-Dataset-master/"
    "HARD-Arabic-Dataset-master/data/balanced-reviews/balanced-reviews.txt"
)
_OUTPUT_PATH = Path(settings.knowledge_base_dir) / "test_set" / "hotel_reviews_sample.json"
_SAMPLE_SIZE = 500
_SEED = 42

# Checked in order against the message text; first match wins.
_INTENT_KEYWORDS: list[tuple[str, list[str]]] = [
    ("damaged_product", [
        "مكسور", "تالف", "معطوب", "خراب", "متهالك", "صراصير", "فئران",
        "حشرات", "غير نظيف", "قذارة", "عفن", "وسخ",
    ]),
    ("app_bug", [
        "واي فاي", "الواي فاي", "وايفاي", "الانترنت", "الإنترنت",
        "التكييف", "تكييف", "المصعد", "السخان", "لا يعمل", "معطل",
    ]),
    ("payment_issue", [
        "تأمين", "خصم", "رسوم اضافية", "رسوم إضافية", "فاتورة",
        "مبالغ فيه", "مبالغ فيها", "غالي", "السعر مرتفع", "زياده بالسعر",
        "زيادة بالسعر", "دفع مبلغ",
    ]),
    ("refund_request", [
        "استرداد", "ارجاع المبلغ", "ارجاع فلوسي", "استرجاع المبلغ",
        "لم يرجع المبلغ", "رد المبلغ", "سرقة فلوسي",
    ]),
    ("exchange_request", [
        "تغيير الغرفة", "غرفة غير مطل", "غير مطله", "ترقية الحجز",
        "ترقية الغرفة", "الحجز يختلف", "ليس كما حجزت", "تم اسكاننا بغرفتين",
    ]),
    ("contact_support_request", [
        "المدير", "الادارة", "الإدارة", "اشتكي", "بلاغ", "شكوى رسمية",
        "أرفع قضية", "سأقاضي",
    ]),
    ("shipping_delay", [
        "تأخير", "تأخر", "انتظرت اكثر من ساعة", "انتظرت أكثر من ساعة",
        "انتظار طويل",
    ]),
    ("account_help", [
        "تسجيل الدخول", "كلمة المرور", "حسابي", "البوكينج", "الحجز عن طريق",
    ]),
    ("complaint", [
        "سيء", "سيئة", "سيئ", "مخيب للأمل", "مخيب للامل", "ازعاج", "إزعاج",
        "سوء التعامل", "سوء تعامل", "موظف الاستقبال", "تعامل غير جيد",
        "نظافة سيئة", "ضعيف جداً", "ضعيف جدا", "اسوأ", "أسوأ",
    ]),
]

_GULF_MARKERS = ["وايد", "شلون", "يبا ", "ابغى", "أبغى", "جيك أوت", "ماشاء الله عليكم"]
_EGYPTIAN_MARKERS = ["مفيش", "اوي", "أوي", "كده", "عايز", "بتاع", "خالص", "ازاي", "إزاي"]
_LEVANTINE_MARKERS = ["هيك", "منيح", "كتير", "هلق"]


def _sentiment_proxy(rating: int) -> str:
    """Map a 1-5 star rating to a coarse sentiment label.

    Args:
        rating: The original 1-5 star rating.

    Returns:
        str: One of "negative", "neutral", "positive".
    """
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def _urgency_proxy(rating: int) -> str:
    """Map a 1-5 star rating to a coarse urgency label.

    Args:
        rating: The original 1-5 star rating.

    Returns:
        str: One of "high", "medium", "low".
    """
    if rating == 1:
        return "high"
    if rating == 2:
        return "medium"
    return "low"


def _intent_proxy(text: str, rating: int) -> str:
    """Heuristically assign one of the existing taxonomy intents.

    Args:
        text: The review text (treated as the customer message).
        rating: The original 1-5 star rating, used as a fallback signal.

    Returns:
        str: An intent id from `intent_taxonomy.json`.
    """
    for intent_id, keywords in _INTENT_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return intent_id
    return "complaint" if rating <= 2 else "general_inquiry"


def _dialect_proxy(text: str) -> str:
    """Heuristically guess the dialect of the message text.

    Args:
        text: The review text.

    Returns:
        str: One of "gulf", "egyptian", "levantine", "msa".
    """
    if any(marker in text for marker in _GULF_MARKERS):
        return "gulf"
    if any(marker in text for marker in _EGYPTIAN_MARKERS):
        return "egyptian"
    if any(marker in text for marker in _LEVANTINE_MARKERS):
        return "levantine"
    return "msa"


def build_sample() -> list[dict]:
    """Sample reviews from the HARD dataset and reframe them as support messages.

    Returns:
        list[dict]: Sampled messages with rating-derived sentiment/urgency
            proxies and a heuristic intent/dialect label.

    Raises:
        FileNotFoundError: If the source HARD dataset file is missing.
    """
    if not _SOURCE_PATH.exists():
        raise FileNotFoundError(f"HARD dataset not found: {_SOURCE_PATH}")

    with open(_SOURCE_PATH, encoding="utf-16") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [row for row in reader if row.get("review", "").strip()]

    rng = random.Random(_SEED)
    sampled = rng.sample(rows, _SAMPLE_SIZE)

    messages = []
    for idx, row in enumerate(sampled, start=1):
        text = row["review"].strip()
        rating = int(row["rating"])
        messages.append({
            "id": f"hard_{idx:04d}",
            "message": text,
            "customer_id": f"CUST-HARD-{idx:04d}",
            "rating": rating,
            "sentiment_proxy": _sentiment_proxy(rating),
            "urgency_proxy": _urgency_proxy(rating),
            "intent_proxy": _intent_proxy(text, rating),
            "dialect_proxy": _dialect_proxy(text),
            "source_hotel": row["Hotel name"].strip(),
        })

    return messages


def main() -> None:
    """Build the sampled dataset and write it to disk."""
    logging.basicConfig(level=logging.INFO)
    messages = build_sample()
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(messages, handle, ensure_ascii=False, indent=2)
    logger.info("Wrote %d sampled messages to %s", len(messages), _OUTPUT_PATH)


if __name__ == "__main__":
    main()
