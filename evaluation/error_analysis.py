"""Error analysis for evaluation runs: identifies and reports mismatches."""

import json
import logging
from pathlib import Path
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


def find_misclassifications(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Identify cases where predicted intent or routed team differ from gold labels.

    Args:
        results: A list of per-case dictionaries, each expected to contain
            `id`, `message`, `expected_intent`, `predicted_intent`,
            `expected_team`, and `predicted_team`.

    Returns:
        list[dict[str, Any]]: The subset of `results` where the predicted
            intent or team does not match the expected value.
    """
    errors = []
    for case in results:
        intent_mismatch = case.get("expected_intent") != case.get("predicted_intent")
        team_mismatch = case.get("expected_team") != case.get("predicted_team")
        if intent_mismatch or team_mismatch:
            errors.append({**case, "intent_mismatch": intent_mismatch, "team_mismatch": team_mismatch})
    return errors


def write_error_report(errors: list[dict[str, Any]], filename: str = "error_analysis.json") -> Path:
    """Write the list of misclassified cases to a JSON report.

    Args:
        errors: List of error case dictionaries from `find_misclassifications`.
        filename: Name of the output file under `logs/evaluations/`.

    Returns:
        Path: The path to the written report file.

    Raises:
        OSError: If the output directory or file cannot be written.
    """
    output_dir = Path(settings.logs_dir) / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(errors, handle, ensure_ascii=False, indent=2)

    logger.info("Wrote %d error cases to %s", len(errors), output_path)
    return output_path
