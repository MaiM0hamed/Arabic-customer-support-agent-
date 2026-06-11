"""Rule-based Arabic dialect detection."""

_DIALECT_KEYWORDS: dict[str, list[str]] = {
    "egyptian": ["عايز", "عاوز", "ازيك", "ايه", "مش", "كده", "بقى", "هو ده", "علشان", "خالص"],
    "gulf": ["وش", "شلونك", "أبغى", "ابغى", "وايد", "اشلون", "ليش", "حق", "كذا", "زين"],
    "levantine": ["شو", "هلق", "كتير", "منيح", "بدي", "هيك", "ليش", "تمام كتير"],
    "maghrebi": ["بزاف", "واش", "كيداير", "دابا", "زعما", "غادي", "نتا"],
}

DEFAULT_DIALECT = "msa"


def detect_dialect(text: str) -> str:
    """Detect the Arabic dialect of a message using keyword matching.

    Args:
        text: Sanitized customer message text.

    Returns:
        str: One of `msa`, `egyptian`, `gulf`, `levantine`, `maghrebi`.
            Defaults to `msa` if no dialect-specific keywords are found
            or the input is empty.
    """
    if not text:
        return DEFAULT_DIALECT

    scores: dict[str, int] = {dialect: 0 for dialect in _DIALECT_KEYWORDS}
    for dialect, keywords in _DIALECT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                scores[dialect] += 1

    best_dialect = max(scores, key=lambda key: scores[key])
    if scores[best_dialect] == 0:
        return DEFAULT_DIALECT
    return best_dialect
