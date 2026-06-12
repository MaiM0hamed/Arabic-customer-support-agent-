"""Generate `data/knowledge_base/test_set/gold_test.csv` from the labeled test set."""

import csv
import json
import logging
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

_TEST_SET_DIR = Path(settings.knowledge_base_dir) / "test_set"
_LABELED_PATHS = [
    _TEST_SET_DIR / "labeled_test_set.json",
    _TEST_SET_DIR / "hotel_labeled_test_set.json",
]
_OUTPUT_PATH = _TEST_SET_DIR / "gold_test.csv"

_FIELDNAMES = ["id", "message", "customer_id", "expected_intent", "expected_team", "expected_dialect"]


def build_gold_test_csv() -> int:
    """Convert the labeled JSON test sets into a single CSV gold test file.

    Merges every file in `_LABELED_PATHS` that exists on disk so both the
    hand-written gold cases and the hotel-review-derived gold cases are
    included in the evaluation run.

    Returns:
        int: The number of rows written to `gold_test.csv`.

    Raises:
        FileNotFoundError: If none of `_LABELED_PATHS` exist.
    """
    cases: list[dict] = []
    for path in _LABELED_PATHS:
        if not path.exists():
            logger.warning("Labeled test set not found, skipping: %s", path)
            continue
        with open(path, encoding="utf-8") as handle:
            cases.extend(json.load(handle))

    if not cases:
        raise FileNotFoundError(f"No labeled test sets found in {_LABELED_PATHS}")

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_PATH, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for case in cases:
            writer.writerow({key: case.get(key, "") for key in _FIELDNAMES})

    return len(cases)


def main() -> None:
    """Build the gold test CSV and report the row count."""
    logging.basicConfig(level=logging.INFO)
    count = build_gold_test_csv()
    logger.info("Wrote %d rows to %s", count, _OUTPUT_PATH)


if __name__ == "__main__":
    main()
