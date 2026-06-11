"""Generate `data/knowledge_base/test_set/gold_test.csv` from the labeled test set."""

import csv
import json
import logging
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

_LABELED_PATH = Path(settings.knowledge_base_dir) / "test_set" / "labeled_test_set.json"
_OUTPUT_PATH = Path(settings.knowledge_base_dir) / "test_set" / "gold_test.csv"

_FIELDNAMES = ["id", "message", "customer_id", "expected_intent", "expected_team", "expected_dialect"]


def build_gold_test_csv() -> int:
    """Convert the labeled JSON test set into a CSV gold test file.

    Returns:
        int: The number of rows written to `gold_test.csv`.

    Raises:
        FileNotFoundError: If `labeled_test_set.json` does not exist.
    """
    if not _LABELED_PATH.exists():
        raise FileNotFoundError(f"Labeled test set not found: {_LABELED_PATH}")

    with open(_LABELED_PATH, encoding="utf-8") as handle:
        cases = json.load(handle)

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
