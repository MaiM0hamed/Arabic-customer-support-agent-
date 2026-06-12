"""Error analysis for evaluation runs: identifies and reports mismatches."""

import json
import logging
from collections import Counter
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
            intent or team does not match the expected value. Each entry is
            annotated with `intent_mismatch`, `team_mismatch`, and
            `dialect_mismatch` boolean flags.
    """
    errors = []
    for case in results:
        intent_mismatch = case.get("expected_intent") != case.get("predicted_intent")
        team_mismatch = case.get("expected_team") != case.get("predicted_team")
        dialect_mismatch = case.get("expected_dialect") != case.get("predicted_dialect")
        if intent_mismatch or team_mismatch:
            errors.append({
                **case,
                "intent_mismatch": intent_mismatch,
                "team_mismatch": team_mismatch,
                "dialect_mismatch": dialect_mismatch,
            })
    return errors


def build_error_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a breakdown of failures by category and confusion pairs.

    Args:
        results: The full list of per-case evaluation results.

    Returns:
        dict[str, Any]: A summary containing counts of intent, team, and
            dialect mismatches, plus the most common
            (expected -> predicted) confusion pairs for intent and dialect.
    """
    intent_errors = 0
    team_errors = 0
    dialect_errors = 0
    intent_confusion: Counter[str] = Counter()
    dialect_confusion: Counter[str] = Counter()

    for case in results:
        expected_intent = case.get("expected_intent")
        predicted_intent = case.get("predicted_intent")
        expected_team = case.get("expected_team")
        predicted_team = case.get("predicted_team")
        expected_dialect = case.get("expected_dialect")
        predicted_dialect = case.get("predicted_dialect")

        if expected_intent != predicted_intent:
            intent_errors += 1
            intent_confusion[f"{expected_intent} -> {predicted_intent}"] += 1
        if expected_team != predicted_team:
            team_errors += 1
        if expected_dialect != predicted_dialect:
            dialect_errors += 1
            dialect_confusion[f"{expected_dialect} -> {predicted_dialect}"] += 1

    return {
        "total_cases": len(results),
        "intent_errors": intent_errors,
        "team_errors": team_errors,
        "dialect_errors": dialect_errors,
        "intent_confusion_pairs": dict(intent_confusion.most_common()),
        "dialect_confusion_pairs": dict(dialect_confusion.most_common()),
    }


def write_error_report(
    errors: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
    filename: str = "error_analysis.json",
) -> Path:
    """Write the list of misclassified cases (and optional summary) to a JSON report.

    Args:
        errors: List of error case dictionaries from `find_misclassifications`.
        summary: Optional failure breakdown from `build_error_summary`.
        filename: Name of the output file under `logs/evaluations/`.

    Returns:
        Path: The path to the written report file.

    Raises:
        OSError: If the output directory or file cannot be written.
    """
    output_dir = Path(settings.logs_dir) / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    report: dict[str, Any] = {"errors": errors}
    if summary is not None:
        report["summary"] = summary

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    logger.info("Wrote %d error cases to %s", len(errors), output_path)
    return output_path
