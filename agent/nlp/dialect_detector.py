"""Rule-based Arabic dialect detection."""

import re

# Keywords without spaces are matched as whole words (token-level) to avoid
# false positives from short strings that occur as substrings of unrelated
# MSA words (e.g. "مش" inside "مشكلة", "حق" inside "يستحق"). Keywords that
# contain a space are matched as substrings against the raw text.
_DIALECT_KEYWORDS: dict[str, list[str]] = {
    "egyptian": [
        "عايز", "عاوز", "عايزة", "عاوزة", "ازيك", "ازيكم", "ايه", "إيه", "مش",
        "كده", "كدا", "بقى", "هو ده", "علشان", "خالص", "دلوقتي", "إزاي", "ازاي",
        "فين", "ليه", "كمان", "اللي", "بيقفل", "بتاع", "بتاعة", "جدًا خالص",
        "دايخ", "تعبانة", "حاجة", "مفيش", "هيّه", "لسه", "لسة", "جامد", "يلا",
    ],
    "gulf": [
        "وش", "شلونك", "شلونكم", "أبغى", "ابغى", "ابغا", "أبغا", "وايد", "اشلون",
        "ليش", "حق", "كذا", "زين", "زينة", "مو", "ودي", "وديت", "أبد", "تراني",
        "ترى", "عساك", "يبه", "خوش", "صج", "أكيد كذا", "هلبت", "شوي", "حيل",
        "تكفى", "تكفون", "ولدي",
    ],
    "levantine": [
        "شو", "هلق", "هلأ", "كتير", "منيح", "منيحة", "بدي", "بدك", "بدو", "هيك",
        "ليش", "تمام كتير", "إيمتى", "إيمتا", "لحالي", "كمان", "تبعي", "تبعو",
        "هلأة", "مبارح", "ولك", "يعطيك العافية",
    ],
    "maghrebi": [
        "بزاف", "واش", "كيداير", "كيفاش", "دابا", "زعما", "غادي", "نتا", "نتي",
        "بغيت", "ماشي هكا", "هاد", "هادي", "راه", "واخا", "بصح", "شحال", "علاش",
        "دروك",
    ],
}

DEFAULT_DIALECT = "msa"

_WORD_RE = re.compile(r"[؀-ۿ]+")


def detect_dialect(text: str) -> str:
    """Detect the Arabic dialect of a message using keyword matching.

    Single-word keywords are matched against whole word tokens to avoid
    false positives from short substrings that appear inside unrelated MSA
    words. Multi-word phrases are matched as substrings of the raw text.

    Args:
        text: Sanitized customer message text.

    Returns:
        str: One of `msa`, `egyptian`, `gulf`, `levantine`, `maghrebi`.
            Defaults to `msa` if no dialect-specific keywords are found
            or the input is empty.
    """
    if not text:
        return DEFAULT_DIALECT

    tokens = set(_WORD_RE.findall(text))

    scores: dict[str, int] = {dialect: 0 for dialect in _DIALECT_KEYWORDS}
    for dialect, keywords in _DIALECT_KEYWORDS.items():
        for keyword in keywords:
            if " " in keyword:
                if keyword in text:
                    scores[dialect] += 1
            elif keyword in tokens:
                scores[dialect] += 1

    best_dialect = max(scores, key=lambda key: scores[key])
    if scores[best_dialect] == 0:
        return DEFAULT_DIALECT
    return best_dialect
